//! Whisper speech-to-text node
//!
//! Provides speech-to-text transcription using OpenAI's Whisper models
//! via the Candle ML framework.
//!
//! Defined with the `#[node]` attribute macro so the `loadable-export`
//! cargo feature on this crate auto-emits the FFI plumbing
//! (`WhisperNodeLoadableFactory` + `inventory::submit!` entry +
//! `LoadableFactoryEntry`). Same source compiles two ways:
//!
//! * `cargo build` — only the in-tree `AsyncStreamingNode` impl plus
//!   `WhisperNodeFactory` (used by `CandleNodesProvider`) ship.
//! * `cargo build --features loadable-export` — additionally compiles
//!   the loadable-bundle FFI surface so a wrapper cdylib can re-export
//!   `WhisperNode` through `plugin_export!()`.

mod model;

pub use model::{WhisperModel, WhisperModelExt};

use crate::cache::ModelCache;
use crate::convert::{AudioData, RuntimeDataConverter};
use crate::device::{DeviceSelector, InferenceDevice};
use crate::error::CandleNodeError;
// Local alias so the helper methods below keep their old `Result<T>`
// shape (= `Result<T, CandleNodeError>`) without shadowing std's
// generic two-arg `Result` at the `#[node]` macro expansion site
// (the macro emits `Result<RuntimeData, Error>` in absolute paths
// and a 1-arg shadowed `Result` makes that look like a misuse).
use crate::error::Result as CResult;

use async_trait::async_trait;
use remotemedia_core::capabilities::{
    AudioConstraints, CapabilityBehavior, ConstraintValue, MediaCapabilities, MediaConstraints,
};
use remotemedia_core::data::RuntimeData;
use remotemedia_core::nodes::streaming_node::{
    AsyncStreamingNode, InitializeContextRead, NodeRuntimeContextRead, StreamingNode,
    StreamingNodeFactory,
};
use remotemedia_core::Error;
use remotemedia_core::node;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

#[cfg(feature = "whisper")]
use candle_core::{Device, IndexOp, Tensor};
#[cfg(feature = "whisper")]
use candle_nn::VarBuilder;
#[cfg(feature = "whisper")]
use candle_transformers::models::whisper::{self as m, audio, Config};
#[cfg(feature = "whisper")]
use tokenizers::Tokenizer;

const NODE_TYPE: &str = "Whisper";

// ---------------------------------------------------------------------------
// Internal lazy-loaded model state
// ---------------------------------------------------------------------------

/// Internal model state after loading
#[cfg(feature = "whisper")]
pub(crate) struct WhisperModelState {
    /// Candle device
    candle_device: Device,
    /// Whisper model
    model: m::model::Whisper,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Model config
    config: Config,
    /// Mel filters for audio processing
    mel_filters: Vec<f32>,
}

#[cfg(not(feature = "whisper"))]
pub(crate) struct WhisperModelState {
    _placeholder: (),
}

// ---------------------------------------------------------------------------
// #[node]-defined Whisper struct
// ---------------------------------------------------------------------------

/// Whisper speech-to-text node.
///
/// Config fields are flattened directly into the node — `#[node]`
/// promotes them into a generated `WhisperNodeConfig` and rewrites
/// `WhisperNode` to hold a `pub config: WhisperNodeConfig` field plus
/// the explicit `#[state]` slots below.
#[node(
    node_type = "Whisper",
    category = "ml",
    description = "Speech-to-text transcription via OpenAI Whisper (Candle backend)",
    accepts = "audio",
    produces = "text",
    capabilities = "static",
    input_caps = "audio(sample_rate=16000, channels=1, format=F32)",
    output_caps = "text"
)]
pub struct WhisperNode {
    /// Model variant to use (Tiny / Base / Small / Medium / LargeV3).
    #[config]
    pub model: WhisperModel,

    /// Target language code (ISO 639-1) or "auto" for detection.
    #[config(default = "auto".to_string())]
    pub language: String,

    /// Inference device ("auto", "cpu", "cuda", "metal").
    #[config(default = "auto".to_string())]
    pub device: String,

    /// Enable streaming partial transcriptions.
    #[config(default = true)]
    pub streaming: bool,

    /// Task: "transcribe" or "translate".
    #[config(default = "transcribe".to_string())]
    pub task: String,

    /// Lazily-loaded model state (model + tokenizer + mel filters).
    /// Populated on first call to `initialize()` or `process_impl()`.
    #[state(default = RwLock::new(None))]
    model_state: RwLock<Option<WhisperModelState>>,
}

// ---------------------------------------------------------------------------
// Backwards-compatibility alias: pre-macro callers reach for `WhisperConfig`.
// ---------------------------------------------------------------------------

/// Legacy alias for the macro-generated `WhisperNodeConfig`.
///
/// Kept so existing call sites that wrote
/// `let cfg: WhisperConfig = serde_json::from_value(...)?;` keep
/// compiling.
pub type WhisperConfig = WhisperNodeConfig;

impl WhisperNodeConfig {
    /// Create config from JSON value (back-compat shim).
    pub fn from_json(value: &serde_json::Value) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_value(value.clone())
    }

    /// Validate configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.language != "auto" && self.language.len() != 2 {
            return Err(format!(
                "Invalid language code: {}. Use 'auto' or ISO 639-1 code (e.g., 'en', 'es')",
                self.language
            ));
        }
        if self.task != "transcribe" && self.task != "translate" {
            return Err(format!(
                "Invalid task: {}. Use 'transcribe' or 'translate'",
                self.task
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// User-supplied processing logic.
//
// `#[node]` auto-generates the `AsyncStreamingNode::process()` body
// that delegates here. We also expose a couple of helpers that used to
// live on the original hand-written struct.
// ---------------------------------------------------------------------------

impl WhisperNode {
    /// Build a node from a JSON params blob — preserves the legacy
    /// `(node_id, params)` call shape used by `WhisperNodeFactory`.
    pub fn from_params(_node_id: impl Into<String>, params: &Value) -> CResult<Self> {
        let cfg = WhisperConfig::from_json(params)
            .map_err(|e| CandleNodeError::configuration("candle-whisper", e.to_string()))?;
        cfg.validate()
            .map_err(|e| CandleNodeError::configuration("candle-whisper", e))?;
        Ok(Self::new(cfg))
    }

    /// Resolve the inference device on demand.
    fn select_device(&self) -> CResult<InferenceDevice> {
        DeviceSelector::from_config(&self.config.device)
    }

    /// User-supplied processing entry point. The `#[node]` macro
    /// generates the trait impl that delegates here.
    pub async fn process_impl(
        &self,
        data: RuntimeData,
    ) -> std::result::Result<RuntimeData, Error> {
        // Extract metadata from input
        let input_metadata = match &data {
            RuntimeData::Audio { metadata, .. } => metadata.clone(),
            _ => None,
        };

        // Extract audio from input
        let audio = RuntimeDataConverter::extract_audio(&data, NODE_TYPE)
            .map_err(|e| Error::Execution(e.to_string()))?;

        // Transcribe
        let transcription = self
            .transcribe(audio)
            .await
            .map_err(|e| Error::Execution(e.to_string()))?;

        match input_metadata {
            Some(ref meta) if meta.get("diarization").is_some() => {
                Ok(RuntimeData::Json(serde_json::json!({
                    "text": transcription,
                    "diarization": meta["diarization"],
                })))
            }
            _ => Ok(RuntimeData::Text(transcription)),
        }
    }

    /// Load model weights (lazy, idempotent).
    #[cfg(feature = "whisper")]
    async fn load_model(&self) -> CResult<()> {
        let mut state = self.model_state.write().await;

        if state.is_some() {
            return Ok(());
        }

        let device = self.select_device()?;
        info!("Loading Whisper model: {} on {}", self.config.model, device);

        let cache = ModelCache::new();

        // Download model files if not cached
        let model_id = self.config.model.model_id();

        let config_path = cache.download_model(model_id, "config.json", None).await?;
        let tokenizer_path = cache
            .download_model(model_id, "tokenizer.json", None)
            .await?;
        let weights_path = cache
            .download_model(model_id, "model.safetensors", None)
            .await?;

        // Initialize candle device
        let candle_device: Device = (&device).try_into()?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            CandleNodeError::model_load(config_path.display().to_string(), e.to_string())
        })?;
        let model_config: Config = serde_json::from_str(&config_str)
            .map_err(|e| CandleNodeError::model_load("config.json", e.to_string()))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            CandleNodeError::model_load(tokenizer_path.display().to_string(), e.to_string())
        })?;

        // Load mel filters
        let mel_filters = Self::get_mel_filters(model_config.num_mel_bins)?;

        // Load model weights
        info!("Loading model weights from {:?}", weights_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, &candle_device)
                .map_err(|e| CandleNodeError::model_load("weights", e.to_string()))?
        };

        let model = m::model::Whisper::load(&vb, model_config.clone())
            .map_err(|e| CandleNodeError::model_load("whisper", e.to_string()))?;

        info!("Whisper model loaded successfully");

        *state = Some(WhisperModelState {
            candle_device,
            model,
            tokenizer,
            config: model_config,
            mel_filters,
        });

        Ok(())
    }

    /// Get mel filter bank coefficients
    #[cfg(feature = "whisper")]
    fn get_mel_filters(num_mel_bins: usize) -> CResult<Vec<f32>> {
        let mel_bytes: &[u8] = match num_mel_bins {
            80 => include_bytes!("mel_filters_80.bin"),
            128 => include_bytes!("mel_filters_128.bin"),
            _ => {
                return Err(CandleNodeError::configuration(
                    "whisper",
                    format!("Unsupported num_mel_bins: {}", num_mel_bins),
                ))
            }
        };

        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        for (i, chunk) in mel_bytes.chunks(4).enumerate() {
            if chunk.len() == 4 {
                mel_filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        Ok(mel_filters)
    }

    /// Transcribe audio data
    #[cfg(feature = "whisper")]
    async fn transcribe(&self, audio: AudioData) -> CResult<String> {
        // Ensure model is loaded
        self.load_model().await?;

        let prepared = audio.prepare_for_whisper()?;

        debug!(
            "Transcribing {} samples at {}Hz",
            prepared.samples.len(),
            prepared.sample_rate
        );

        let mut state = self.model_state.write().await;
        let state = state
            .as_mut()
            .ok_or_else(|| CandleNodeError::inference(NODE_TYPE, "Model not loaded"))?;

        let mel = audio::pcm_to_mel(&state.config, &prepared.samples, &state.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                state.config.num_mel_bins,
                mel_len / state.config.num_mel_bins,
            ),
            &state.candle_device,
        )
        .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

        debug!("Mel spectrogram shape: {:?}", mel.dims());

        let audio_features = state
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

        let sot_token = Self::token_id(&state.tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = Self::token_id(&state.tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = Self::token_id(&state.tokenizer, m::EOT_TOKEN)?;
        let no_timestamps_token = Self::token_id(&state.tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        let language_token = if self.config.model.is_multilingual() {
            let lang_str = format!("<|{}|>", self.config.language);
            Self::token_id(&state.tokenizer, &lang_str).ok()
        } else {
            None
        };

        let mut tokens = vec![sot_token];
        if let Some(lang_token) = language_token {
            tokens.push(lang_token);
        }
        tokens.push(transcribe_token);
        tokens.push(no_timestamps_token);

        let sample_len = state.config.max_target_positions / 2;
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &state.candle_device)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

            let ys = state
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

            let (_, seq_len, _) = ys
                .dims3()
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

            let logits = state
                .model
                .decoder
                .final_linear(
                    &ys.i((..1, seq_len - 1..))
                        .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?,
                )
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?
                .i(0)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?
                .i(0)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

            let logits_v: Vec<f32> = logits
                .to_vec1()
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(eot_token);

            if next_token == eot_token || tokens.len() > state.config.max_target_positions {
                break;
            }
            tokens.push(next_token);
        }

        let text = state
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

        Ok(text)
    }

    /// Get token ID from tokenizer
    #[cfg(feature = "whisper")]
    fn token_id(tokenizer: &Tokenizer, token: &str) -> CResult<u32> {
        tokenizer.token_to_id(token).ok_or_else(|| {
            CandleNodeError::inference(NODE_TYPE, format!("Token not found: {}", token))
        })
    }

    #[cfg(not(feature = "whisper"))]
    async fn transcribe(&self, _audio: AudioData) -> CResult<String> {
        Err(CandleNodeError::configuration(
            "candle-whisper",
            "Whisper feature not enabled at compile time",
        ))
    }

    #[cfg(not(feature = "whisper"))]
    async fn load_model(&self) -> CResult<()> {
        Err(CandleNodeError::configuration(
            "candle-whisper",
            "Whisper feature not enabled at compile time",
        ))
    }
}

// ---------------------------------------------------------------------------
// In-tree integration: wrapper + factory.
//
// `#[node]` auto-generates `AsyncStreamingNode for WhisperNode` (used
// by both this wrapper and the loadable-export FFI adapter). The
// wrapper bridges to the unified `StreamingNode` trait the
// `StreamingNodeRegistry` consumes, and lets us forward
// node-instance-specific things (node_id, capability metadata) that
// the macro's static surface doesn't carry.
// ---------------------------------------------------------------------------

/// Wrapper to make WhisperNode a StreamingNode
pub struct WhisperNodeWrapper {
    node_id: String,
    inner: Arc<WhisperNode>,
}

impl WhisperNodeWrapper {
    pub fn new(node_id: impl Into<String>, inner: Arc<WhisperNode>) -> Self {
        Self {
            node_id: node_id.into(),
            inner,
        }
    }
}

#[async_trait]
impl StreamingNode for WhisperNodeWrapper {
    fn node_type(&self) -> &str {
        AsyncStreamingNode::node_type(&*self.inner)
    }

    fn node_id(&self) -> &str {
        &self.node_id
    }

    async fn initialize(
        &self,
        ctx: &dyn InitializeContextRead,
    ) -> std::result::Result<(), Error> {
        AsyncStreamingNode::initialize(&*self.inner, ctx).await
    }

    async fn process_async(
        &self,
        data: RuntimeData,
        _ctx: &dyn NodeRuntimeContextRead,
    ) -> std::result::Result<RuntimeData, Error> {
        AsyncStreamingNode::process(&*self.inner, data).await
    }

    async fn process_multi_async(
        &self,
        inputs: std::collections::HashMap<String, RuntimeData>,
        ctx: &dyn NodeRuntimeContextRead,
    ) -> std::result::Result<RuntimeData, Error> {
        if let Some((_, data)) = inputs.into_iter().next() {
            self.process_async(data, ctx).await
        } else {
            Err(Error::Execution("No input data provided".into()))
        }
    }

    fn is_multi_input(&self) -> bool {
        false
    }

    fn media_capabilities(&self) -> Option<MediaCapabilities> {
        Some(MediaCapabilities::with_input(MediaConstraints::Audio(
            AudioConstraints {
                sample_rate: Some(ConstraintValue::Exact(16000)),
                channels: Some(ConstraintValue::Exact(1)),
                ..Default::default()
            },
        )))
    }

    fn capability_behavior(&self) -> CapabilityBehavior {
        CapabilityBehavior::Static
    }
}

/// Factory for creating WhisperNode instances
pub struct WhisperNodeFactory;

impl WhisperNodeFactory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WhisperNodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingNodeFactory for WhisperNodeFactory {
    fn create(
        &self,
        node_id: String,
        params: &Value,
        _session_id: Option<String>,
    ) -> std::result::Result<Box<dyn StreamingNode>, Error> {
        let node = WhisperNode::from_params(&node_id, params)
            .map_err(|e| Error::Execution(e.to_string()))?;
        Ok(Box::new(WhisperNodeWrapper::new(node_id, Arc::new(node))))
    }

    fn node_type(&self) -> &str {
        // Historical type tag kept for in-tree pipelines built against
        // the v0.4 registry; the loadable-export factory uses the
        // macro-derived "Whisper" tag instead.
        "candle-whisper"
    }

    fn media_capabilities(&self, _params: &Value) -> Option<MediaCapabilities> {
        Some(MediaCapabilities::with_input(MediaConstraints::Audio(
            AudioConstraints {
                sample_rate: Some(ConstraintValue::Exact(16000)),
                channels: Some(ConstraintValue::Exact(1)),
                ..Default::default()
            },
        )))
    }

    fn capability_behavior(&self) -> CapabilityBehavior {
        CapabilityBehavior::Static
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_config_default() {
        let config = WhisperConfig::default();
        assert_eq!(config.model, WhisperModel::Base);
    }

    #[test]
    fn test_whisper_node_creation() {
        let config = WhisperConfig::default();
        let _node = WhisperNode::new(config);
    }

    #[test]
    fn test_factory_node_type() {
        let factory = WhisperNodeFactory::new();
        assert_eq!(factory.node_type(), "candle-whisper");
    }
}
