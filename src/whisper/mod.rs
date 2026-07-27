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
mod streaming;

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
use self::streaming::{DecodedSegment, StreamingOptions, WhisperStreamingState};
use crate::error::Result as CResult;

use async_trait::async_trait;
use remotemedia_core::capabilities::{
    AudioConstraints, CapabilityBehavior, ConstraintValue, MediaCapabilities, MediaConstraints,
};
use remotemedia_core::data::RuntimeData;
use remotemedia_core::node;
use remotemedia_core::nodes::streaming_node::{
    AsyncStreamingNode, InitializeContextRead, NodeRuntimeContextRead, StreamingNode,
    StreamingNodeFactory,
};
use remotemedia_core::Error;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
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
    accepts = "audio,json",
    produces = "text,json",
    capabilities = "static",
    input_caps = "audio(sample_rate=16000, channels=1, format=F32)",
    output_caps = "text",
    streaming = true
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

    /// New normalized audio required between streaming inference passes.
    #[config(default = 2000)]
    pub streaming_decode_interval_ms: u32,

    /// Audio retained before the last committed timestamp for decoder context.
    #[config(default = 750)]
    pub streaming_overlap_ms: u32,

    /// Hard upper bound for audio decoded by any streaming inference pass.
    #[config(default = 30000)]
    pub streaming_max_window_ms: u32,

    /// Consecutive hypotheses required before text becomes stable.
    #[config(default = 2)]
    pub streaming_agreement_updates: usize,

    /// Agreed trailing words kept revisable to avoid premature commitment.
    #[config(default = 2)]
    pub streaming_trailing_guard_words: usize,

    /// Task: "transcribe" or "translate".
    #[config(default = "transcribe".to_string())]
    pub task: String,

    /// Optional initial prompt prepended to the decode context. Biases the
    /// model toward specific vocabulary (e.g. a user's Steam library titles)
    /// so names are transcribed correctly instead of phonetically. Empty/None
    /// disables the bias. Tokens are appended right after the SOT/language/
    /// task prefix (whisper's canonical initial-prompt slot) and before the
    /// first decode step.
    #[config(default = None)]
    pub initial_prompt: Option<String>,

    /// Lazily-loaded model state (model + tokenizer + mel filters).
    /// Populated on first call to `initialize()` or `process_impl()`.
    #[state(default = RwLock::new(None))]
    model_state: RwLock<Option<WhisperModelState>>,

    /// Optional runtime config override. When set (by `apply_param_update`,
    /// driven by a `__param_update__` control frame), all live `self.config.*`
    /// reads consult this instead of the statically-built `config`. Lets a
    /// caller hot-swap model/language/device/task at runtime without a
    /// full pipeline-graph rebuild — we just drop `model_state` and the
    /// next frame reloads the model lazily.
    #[state(default = RwLock::new(None))]
    config_override: RwLock<Option<WhisperNodeConfig>>,

    /// Incremental audio, agreement, and revision state for this node instance.
    #[state(default = Mutex::new(WhisperStreamingState::default()))]
    streaming_state: Mutex<WhisperStreamingState>,

    /// Serializes streaming state transitions across concurrent input arrivals.
    #[state(default = Mutex::new(()))]
    streaming_gate: Mutex<()>,
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
        if self.streaming_decode_interval_ms == 0 {
            return Err("streaming_decode_interval_ms must be greater than zero".to_string());
        }
        if !(1000..=30000).contains(&self.streaming_max_window_ms) {
            return Err("streaming_max_window_ms must be between 1000 and 30000".to_string());
        }
        if self.streaming_overlap_ms >= self.streaming_max_window_ms {
            return Err(
                "streaming_overlap_ms must be smaller than streaming_max_window_ms".to_string(),
            );
        }
        if self.streaming_decode_interval_ms > self.streaming_max_window_ms {
            return Err(
                "streaming_decode_interval_ms must not exceed streaming_max_window_ms".to_string(),
            );
        }
        if self.streaming_agreement_updates < 2 {
            return Err("streaming_agreement_updates must be at least 2".to_string());
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
    pub async fn process_impl(&self, data: RuntimeData) -> std::result::Result<RuntimeData, Error> {
        tracing::info!("[Whisper] process_impl entered, type={:?}", data.data_type());
        // Runtime param-override control frame: consume it before the
        // audio passthrough. On the single-output path there's no
        // "consume" semantics, so echo an ack so the frame isn't
        // silently dropped; downstream nodes will passthrough the ack.
        // (The real consumption happens in process_streaming_impl.)
        if let RuntimeData::Json(value) = &data {
            if value.get("__param_update__").is_some() {
                self.apply_param_update(value.get("__param_update__").unwrap())
                    .await
                    .map_err(|e| Error::Execution(e.to_string()))?;
                return Ok(RuntimeData::Json(serde_json::json!({ "param_update": "ok" })));
            }
        }

        // Passthrough anything that isn't audio (e.g. ControlMessage emitted
        // by upstream nodes such as the AudioBufferAccumulator). The streaming
        // path already does this; mirror it here so the non-streaming path
        // (used when config.streaming == false) doesn't reject the
        // control_message the accumulator routes to this node.
        if !matches!(data, RuntimeData::Audio { .. }) {
            return Ok(data);
        }

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

    /// Resolve the effective config: the runtime override if set,
    /// otherwise the statically-built `config`.
    async fn resolved_config(&self) -> WhisperNodeConfig {
        self.config_override
            .read()
            .await
            .clone()
            .unwrap_or_else(|| self.config.clone())
    }

    /// Apply a runtime parameter update pushed as a `__param_update__` JSON
    /// control frame. Parses model/language/device/task from the patch,
    /// validates, commits the override, and drops `model_state` so the
    /// next frame lazily reloads the (possibly new) model.
    pub async fn apply_param_update(
        &self,
        patch: &Value,
    ) -> std::result::Result<(), CandleNodeError> {
        let mut base = self.resolved_config().await;

        if let Some(v) = patch.get("model") {
            base.model = serde_json::from_value(v.clone())
                .map_err(|e| CandleNodeError::configuration("candle-whisper", format!("invalid model: {e}")))?;
        }
        if let Some(v) = patch.get("language") {
            base.language = serde_json::from_value(v.clone())
                .map_err(|e| CandleNodeError::configuration("candle-whisper", format!("invalid language: {e}")))?;
        }
        if let Some(v) = patch.get("device") {
            base.device = serde_json::from_value(v.clone())
                .map_err(|e| CandleNodeError::configuration("candle-whisper", format!("invalid device: {e}")))?;
        }
        if let Some(v) = patch.get("task") {
            base.task = serde_json::from_value(v.clone())
                .map_err(|e| CandleNodeError::configuration("candle-whisper", format!("invalid task: {e}")))?;
        }

        base.validate()
            .map_err(|e| CandleNodeError::configuration("candle-whisper", e))?;

        *self.config_override.write().await = Some(base);
        *self.model_state.write().await = None;

        info!("[Whisper] runtime param override applied; model will reload on next frame");
        Ok(())
    }

    fn streaming_options(&self) -> StreamingOptions {
        let cfg = self.resolved_config_blocking();
        StreamingOptions {
            decode_interval_ms: cfg.streaming_decode_interval_ms,
            overlap_ms: cfg.streaming_overlap_ms,
            max_window_ms: cfg.streaming_max_window_ms,
            agreement_updates: cfg.streaming_agreement_updates,
            trailing_guard_words: cfg.streaming_trailing_guard_words,
        }
    }

    /// Synchronous resolve for call sites that already hold no async
    /// borrow (e.g. the sync `streaming_options`). Reads the override
    /// if present, else the static config.
    fn resolved_config_blocking(&self) -> WhisperNodeConfig {
        // RwLock read is safe to await in an async fn, but
        // `streaming_options` is sync; use a best-effort try-read and
        // fall back to the static config if the lock is contended.
        // tokio::sync::RwLock::try_read returns Result<Guard, TryLockError>,
        // so map it to Option first.
        self.config_override
            .try_read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_else(|| self.config.clone())
    }

    /// Native incremental Whisper path used by in-tree and loadable runtimes.
    /// Audio inputs contain only new PCM; JSON `{ "type": "flush" }`
    /// finalizes the latest hypothesis.
    pub async fn process_streaming_impl<F>(
        &self,
        data: RuntimeData,
        _session_id: Option<String>,
        mut callback: F,
    ) -> std::result::Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> std::result::Result<(), Error> + Send,
    {
        tracing::info!(
            "[Whisper] process_streaming_impl entered, streaming_config={}, type={:?}",
            self.config.streaming,
            data.data_type()
        );
        if !self.config.streaming {
            callback(self.process_impl(data).await?)?;
            return Ok(1);
        }

        // The trait permits concurrent arrivals. Keep audio mutation, decode,
        // and hypothesis application in stream order for deterministic
        // revision numbers and local-agreement history.
        let _gate = self.streaming_gate.lock().await;
        let options = self.streaming_options();

        // Runtime param-override control frame: consume it here (ahead
        // of the audio arm and the `other =>` passthrough) so it
        // never reaches downstream. Drops model_state; the next frame
        // reloads the (possibly new) model lazily.
        if let RuntimeData::Json(value) = &data {
            if value.get("__param_update__").is_some() {
                self.apply_param_update(value.get("__param_update__").unwrap())
                    .await
                    .map_err(|e| Error::Execution(e.to_string()))?;
                return Ok(0);
            }
        }

        let (window, final_result) = match data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                channels,
                ..
            } => {
                let mut stream = self.streaming_state.lock().await;
                stream
                    .push_audio(&samples, sample_rate, channels)
                    .map_err(|e| Error::Execution(e.to_string()))?;
                (stream.decode_window(options, false), false)
            }
            RuntimeData::Json(value)
                if value
                    .get("type")
                    .or_else(|| value.get("command"))
                    .and_then(Value::as_str)
                    == Some("flush") =>
            {
                let mut stream = self.streaming_state.lock().await;
                (stream.decode_window(options, true), true)
            }
            // Pass through unexpected data types (e.g. control messages)
            // unchanged instead of erroring.
            other => {
                callback(other)?;
                return Ok(1);
            }
        };

        let Some(window) = window else {
            if final_result {
                return Err(Error::Execution(
                    "Whisper flush received before any audio".to_string(),
                ));
            }
            return Ok(0);
        };

        let audio = AudioData {
            samples: window.samples,
            sample_rate: streaming::WHISPER_SAMPLE_RATE,
            channels: 1,
        };
        let segments = self
            .transcribe_segments(audio, true)
            .await
            .map_err(|e| Error::Execution(e.to_string()))?;

        let event = self.streaming_state.lock().await.apply_hypothesis(
            segments,
            window.start_ms,
            options,
            final_result,
        );
        if let Some(event) = event {
            callback(RuntimeData::Json(serde_json::to_value(event).map_err(
                |e| Error::Execution(format!("serialize transcript event: {e}")),
            )?))?;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Load model weights (lazy, idempotent).
    #[cfg(feature = "whisper")]
    async fn load_model(&self) -> CResult<()> {
        if self.model_state.read().await.is_some() {
            return Ok(());
        }
        let resolved = self.resolved_config().await;

        let device = DeviceSelector::from_config(&resolved.device)?;
        info!("Loading Whisper model: {} on {}", resolved.model, device);

        let cache = ModelCache::new();

        // Download model files if not cached
        let model_id = resolved.model.model_id();

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

        *self.model_state.write().await = Some(WhisperModelState {
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

    /// Transcribe audio data through the backwards-compatible batch API.
    #[cfg(feature = "whisper")]
    async fn transcribe(&self, audio: AudioData) -> CResult<String> {
        Ok(self
            .transcribe_segments(audio, false)
            .await?
            .into_iter()
            .map(|segment| segment.text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    /// Decode one bounded audio window. Streaming mode enables timestamp
    /// tokens so agreement and audio trimming operate on model-derived
    /// boundaries rather than guessed wall-clock positions.
    #[cfg(feature = "whisper")]
    async fn transcribe_segments(
        &self,
        audio: AudioData,
        timestamps: bool,
    ) -> CResult<Vec<DecodedSegment>> {
        self.load_model().await?;

        let prepared = audio.prepare_for_whisper()?;
        if prepared.samples.is_empty() {
            return Ok(Vec::new());
        }
        let audio_duration_ms =
            prepared.samples.len() as u64 * 1000 / streaming::WHISPER_SAMPLE_RATE as u64;

        debug!(
            "Transcribing {} samples at {}Hz (timestamps={})",
            prepared.samples.len(),
            prepared.sample_rate,
            timestamps,
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
        // Candle's Whisper mel helper appends at least one 15-second zero
        // chunk. Narrow to Whisper's fixed 30-second encoder input just like
        // the upstream Candle examples; otherwise a full 30-second PCM window
        // becomes 45 seconds of mel frames and exceeds the positional table.
        let mel_frames = mel
            .dim(2)
            .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
        let mel = mel
            .narrow(2, 0, mel_frames.min(m::N_FRAMES))
            .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

        debug!("Mel spectrogram shape: {:?}", mel.dims());

        let audio_features = state
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;

        let resolved = self.resolved_config().await;
        let sot_token = Self::token_id(&state.tokenizer, m::SOT_TOKEN)?;
        let task_token = if resolved.task == "translate" {
            Self::token_id(&state.tokenizer, m::TRANSLATE_TOKEN)?
        } else {
            Self::token_id(&state.tokenizer, m::TRANSCRIBE_TOKEN)?
        };
        let eot_token = Self::token_id(&state.tokenizer, m::EOT_TOKEN)?;
        let no_timestamps_token = Self::token_id(&state.tokenizer, m::NO_TIMESTAMPS_TOKEN)?;

        let language_token = if resolved.model.is_multilingual() {
            let lang_str = format!("<|{}|>", resolved.language);
            Self::token_id(&state.tokenizer, &lang_str).ok()
        } else {
            None
        };

        let mut tokens = vec![sot_token];
        if let Some(lang_token) = language_token {
            tokens.push(lang_token);
        }
        tokens.push(task_token);
        if !timestamps {
            tokens.push(no_timestamps_token);
        }
        // Initial-prompt bias: encode the prompt string into token ids and
        // prepend them after the SOT/lang/task prefix (whisper's canonical
        // initial-prompt slot) so the model is conditioned on the known
        // vocabulary (e.g. a user's Steam library titles) before it begins
        // decoding the audio. Truncate to a sane bound to avoid blowing the
        // 448-token decoder context.
        if let Some(prompt) = &self.config.initial_prompt {
            let prompt = prompt.trim();
            if !prompt.is_empty() {
                match state.tokenizer.encode(prompt, false) {
                    Ok(enc) => {
                        const MAX_PROMPT_TOKENS: usize = 384;
                        let mut ptoks = enc.get_ids().to_vec();
                        if ptoks.len() > MAX_PROMPT_TOKENS {
                            ptoks.truncate(MAX_PROMPT_TOKENS);
                        }
                        tokens.extend_from_slice(&ptoks);
                    }
                    Err(e) => {
                        debug!("Failed to tokenize initial_prompt (ignored): {e}");
                    }
                }
            }
        }
        let sample_begin = tokens.len();

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

            let mut logits_v: Vec<f32> = logits
                .to_vec1()
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
            for &token in &state.config.suppress_tokens {
                if let Some(logit) = logits_v.get_mut(token as usize) {
                    *logit = f32::NEG_INFINITY;
                }
            }
            if timestamps {
                if let Some(logit) = logits_v.get_mut(no_timestamps_token as usize) {
                    *logit = f32::NEG_INFINITY;
                }
                Self::apply_timestamp_rules(
                    &mut logits_v,
                    &tokens,
                    sample_begin,
                    no_timestamps_token,
                    eot_token,
                );
            }
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(eot_token);

            tokens.push(next_token);
            if next_token == eot_token || tokens.len() >= state.config.max_target_positions {
                break;
            }
        }

        if !timestamps {
            let text = state
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
            return Ok(vec![DecodedSegment {
                text,
                start_ms: 0,
                end_ms: audio_duration_ms,
            }]);
        }

        let timestamp_begin = no_timestamps_token + 1;
        let mut segments = Vec::new();
        let mut text_tokens = Vec::new();
        let mut segment_start_ms = 0u64;
        for &token in &tokens[sample_begin..] {
            if token == eot_token {
                break;
            }
            if token >= timestamp_begin {
                let timestamp_ms = ((token - timestamp_begin) as u64 * 20).min(audio_duration_ms);
                if !text_tokens.is_empty() && timestamp_ms >= segment_start_ms {
                    let text = state
                        .tokenizer
                        .decode(&text_tokens, true)
                        .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
                    if !text.trim().is_empty() {
                        segments.push(DecodedSegment {
                            text,
                            start_ms: segment_start_ms,
                            end_ms: timestamp_ms,
                        });
                    }
                    text_tokens.clear();
                }
                segment_start_ms = timestamp_ms;
            } else {
                text_tokens.push(token);
            }
        }
        if !text_tokens.is_empty() {
            let text = state
                .tokenizer
                .decode(&text_tokens, true)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
            if !text.trim().is_empty() {
                segments.push(DecodedSegment {
                    text,
                    start_ms: segment_start_ms,
                    end_ms: audio_duration_ms,
                });
            }
        }

        // A malformed/older tokenizer can occasionally produce text without a
        // timestamp pair. Preserve the hypothesis, but keep it unstable until
        // agreement or final flush rather than dropping useful speech.
        if segments.is_empty() {
            let text = state
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| CandleNodeError::inference(NODE_TYPE, e.to_string()))?;
            if !text.trim().is_empty() {
                segments.push(DecodedSegment {
                    text,
                    start_ms: 0,
                    end_ms: audio_duration_ms,
                });
            }
        }
        Ok(segments)
    }

    #[cfg(feature = "whisper")]
    fn apply_timestamp_rules(
        logits: &mut [f32],
        tokens: &[u32],
        sample_begin: usize,
        no_timestamps_token: u32,
        eot_token: u32,
    ) {
        let timestamp_begin = no_timestamps_token + 1;
        let vocab_size = logits.len() as u32;
        let sampled = tokens.get(sample_begin..).unwrap_or_default();

        if let Some(&last) = sampled.last() {
            let last_was_timestamp = last >= timestamp_begin;
            let penultimate_was_timestamp =
                sampled.len() >= 2 && sampled[sampled.len() - 2] >= timestamp_begin;
            if last_was_timestamp && penultimate_was_timestamp {
                for token in timestamp_begin..vocab_size {
                    logits[token as usize] = f32::NEG_INFINITY;
                }
            } else if last_was_timestamp {
                for token in 0..eot_token {
                    logits[token as usize] = f32::NEG_INFINITY;
                }
            }

            if let Some(&last_timestamp) = sampled
                .iter()
                .rev()
                .find(|&&token| token >= timestamp_begin)
            {
                let minimum = if last_was_timestamp && !penultimate_was_timestamp {
                    last_timestamp
                } else {
                    last_timestamp.saturating_add(1)
                };
                for token in timestamp_begin..minimum.min(vocab_size) {
                    logits[token as usize] = f32::NEG_INFINITY;
                }
            }
        }

        if tokens.len() == sample_begin {
            for token in 0..timestamp_begin.min(vocab_size) {
                logits[token as usize] = f32::NEG_INFINITY;
            }
        }

        if timestamp_begin < vocab_size {
            let timestamp_logits = &logits[timestamp_begin as usize..];
            let max_timestamp = timestamp_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let timestamp_logsumexp = if max_timestamp.is_finite() {
                max_timestamp
                    + timestamp_logits
                        .iter()
                        .map(|value| (*value - max_timestamp).exp())
                        .sum::<f32>()
                        .ln()
            } else {
                f32::NEG_INFINITY
            };
            let max_text = logits[..timestamp_begin as usize]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if timestamp_logsumexp > max_text {
                for token in 0..timestamp_begin {
                    logits[token as usize] = f32::NEG_INFINITY;
                }
            }
        }
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
    async fn transcribe_segments(
        &self,
        _audio: AudioData,
        _timestamps: bool,
    ) -> CResult<Vec<DecodedSegment>> {
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

    async fn initialize(&self, ctx: &dyn InitializeContextRead) -> std::result::Result<(), Error> {
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

    async fn process_streaming_async(
        &self,
        data: RuntimeData,
        ctx: &dyn NodeRuntimeContextRead,
        callback: Box<dyn FnMut(RuntimeData) -> std::result::Result<(), Error> + Send>,
    ) -> std::result::Result<usize, Error> {
        AsyncStreamingNode::process_streaming_with_ctx(&*self.inner, data, ctx, callback).await
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

    fn is_multi_output_streaming(&self) -> bool {
        true
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
        assert!(factory.is_multi_output_streaming());
    }

    #[test]
    fn test_streaming_config_validation() {
        let mut config = WhisperConfig::default();
        assert!(config.validate().is_ok());

        config.streaming_agreement_updates = 1;
        assert!(config.validate().unwrap_err().contains("at least 2"));
        config.streaming_agreement_updates = 2;
        config.streaming_overlap_ms = config.streaming_max_window_ms;
        assert!(config.validate().unwrap_err().contains("smaller"));
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn timestamp_rules_force_an_initial_timestamp() {
        let no_timestamps = 10;
        let mut logits = vec![1.0; 20];
        WhisperNode::apply_timestamp_rules(&mut logits, &[1, 2], 2, no_timestamps, 9);
        assert!(logits[..=no_timestamps as usize]
            .iter()
            .all(|value| value.is_infinite() && value.is_sign_negative()));
        assert!(logits[(no_timestamps + 1) as usize..]
            .iter()
            .any(|value| value.is_finite()));
    }
}
