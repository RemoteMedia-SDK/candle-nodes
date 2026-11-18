//! Parakeet EOU (End-of-Utterance) detection node
//!
//! Provides streaming speech-to-text with end-of-utterance detection using
//! NVIDIA's Parakeet Realtime EOU 120M model via Candle ONNX runtime.
//!
//! The model emits an `<EOU>` token at the end of each utterance, enabling
//! voice agent pipelines to detect when the speaker has finished talking.
//!
//! # Model
//!
//! - Source: nvidia/parakeet_realtime_eou_120m-v1
//! - ONNX export: altunenes/parakeet-rs (encoder.onnx + decoder_joint.onnx)
//! - Architecture: FastConformer encoder + LSTM decoder
//! - Input: 16kHz mono audio
//! - Output: Text transcription + EOU flag

mod config;

pub use config::{
    EouConfig, EOU_SAMPLE_RATE, BUFFER_SIZE_SECONDS, HOP_LENGTH, LOG_ZERO_GUARD, N_FFT, N_MELS,
    PREEMPHASIS, PRE_ENCODE_CACHE, WIN_LENGTH, FMAX, FRAMES_PER_CHUNK, MIN_BUFFER_SAMPLES,
    SLICE_LEN, MAX_SYMBOLS_PER_STEP,
};

use crate::cache::ModelCache;
use crate::convert::{AudioData, RuntimeDataConverter};
use crate::device::{DeviceSelector, InferenceDevice};
use crate::error::{CandleNodeError, Result};

use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3, Array4, s};
use remotemedia_traits::capabilities::{
    AudioConstraints, CapabilityBehavior, ConstraintValue, MediaCapabilities, MediaConstraints,
};
use remotemedia_types::RuntimeData;
use remotemedia_traits::{
    AsyncStreamingNode, InitializeContextRead, NodeRuntimeContextRead, StreamingNode,
    StreamingNodeFactory,
};
use remotemedia_types::Error;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[cfg(feature = "eou")]
use candle_core::{Device, Tensor};

const NODE_TYPE: &str = "parakeet-eou";

/// HuggingFace repo containing the ONNX export
const ONNX_REPO: &str = "altunenes/parakeet-rs";
/// Subdirectory containing ONNX model files
const ONNX_SUBDIR: &str = "realtime_eou_120m-v1-onnx";

// ---------------------------------------------------------------------------
// Encoder cache state for streaming inference
// ---------------------------------------------------------------------------

/// Encoder cache maintains temporal context across audio chunks.
///
/// The cache has two components:
/// - `cache_last_channel`: [17, 1, 70, 512] - 17 layers, 70 frame lookback
/// - `cache_last_time`: [17, 1, 512, 8] - 17 layers, fixed 8 time steps
/// - `cache_last_channel_len`: [1] - current cache length
#[derive(Debug, Clone)]
pub struct EncoderCache {
    /// Channel cache: [17, 1, 70, 512]
    pub cache_last_channel: Vec<f32>,
    /// Channel cache shape
    pub cache_last_channel_shape: [usize; 4],
    /// Time cache: [17, 1, 512, 8]
    pub cache_last_time: Vec<f32>,
    /// Time cache shape
    pub cache_last_time_shape: [usize; 4],
    /// Cache length: [1]
    pub cache_last_channel_len: Vec<i64>,
}

impl EncoderCache {
    /// Initialize zero-filled encoder cache
    pub fn new() -> Self {
        Self {
            cache_last_channel: vec![0.0; 17 * 1 * 70 * 512],
            cache_last_channel_shape: [17, 1, 70, 512],
            cache_last_time: vec![0.0; 17 * 1 * 512 * 8],
            cache_last_time_shape: [17, 1, 512, 8],
            cache_last_channel_len: vec![0i64],
        }
    }

    /// Create from raw tensor data
    #[cfg(feature = "eou")]
    fn from_tensors(
        channel: &Tensor,
        time: &Tensor,
        channel_len: &Tensor,
    ) -> candle_core::Result<Self> {
        let ch_shape: Vec<usize> = channel.shape().dims().iter().map(|d| *d as usize).collect();
        let tm_shape: Vec<usize> = time.shape().dims().iter().map(|d| *d as usize).collect();

        let ch_data: Vec<f32> = channel.flatten_all()?.to_vec1()?;
        let tm_data: Vec<f32> = time.flatten_all()?.to_vec1()?;
        let len_data: Vec<i64> = channel_len.flatten_all()?.to_vec1()?;

        Ok(Self {
            cache_last_channel: ch_data,
            cache_last_channel_shape: [
                ch_shape[0],
                ch_shape.get(1).copied().unwrap_or(1),
                ch_shape.get(2).copied().unwrap_or(70),
                ch_shape.get(3).copied().unwrap_or(512),
            ],
            cache_last_time: tm_data,
            cache_last_time_shape: [
                tm_shape[0],
                tm_shape.get(1).copied().unwrap_or(1),
                tm_shape.get(2).copied().unwrap_or(512),
                tm_shape.get(3).copied().unwrap_or(8),
            ],
            cache_last_channel_len: len_data,
        })
    }

    /// Convert cache to ONNX input tensors
    #[cfg(feature = "eou")]
    fn to_tensors(&self, device: &Device) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let channel = Tensor::from_slice(
            &self.cache_last_channel,
            self.cache_last_channel_shape.to_vec(),
            device,
        )?;
        let time = Tensor::from_slice(
            &self.cache_last_time,
            self.cache_last_time_shape.to_vec(),
            device,
        )?;
        let channel_len = Tensor::from_slice(
            &self.cache_last_channel_len,
            self.cache_last_channel_len.len(),
            device,
        )?;
        Ok((channel, time, channel_len))
    }
}

impl Default for EncoderCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Decoder LSTM state
// ---------------------------------------------------------------------------

/// Decoder LSTM hidden state and cell state for streaming autoregressive decoding
#[derive(Debug, Clone)]
pub struct DecoderState {
    /// Hidden state: [1, 1, 640]
    pub state_h: Vec<f32>,
    /// Cell state: [1, 1, 640]
    pub state_c: Vec<f32>,
    /// Last predicted token
    pub last_token: i32,
}

impl DecoderState {
    /// Initialize with blank token
    pub fn new(blank_id: i32) -> Self {
        Self {
            state_h: vec![0.0; 1 * 1 * 640],
            state_c: vec![0.0; 1 * 1 * 640],
            last_token: blank_id,
        }
    }

    /// Reset decoder state (soft reset - preserves encoder cache)
    pub fn reset(&mut self, blank_id: i32) {
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.last_token = blank_id;
    }

    /// Convert to ONNX input tensors
    #[cfg(feature = "eou")]
    fn to_tensors(&self, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
        let h = Tensor::from_slice(&self.state_h, (1, 1, 640), device)?;
        let c = Tensor::from_slice(&self.state_c, (1, 1, 640), device)?;
        Ok((h, c))
    }

    /// Update from ONNX output tensors
    #[cfg(feature = "eou")]
    fn update_from_tensors(&mut self, h: &Tensor, c: &Tensor) -> candle_core::Result<()> {
        self.state_h = h.flatten_all()?.to_vec1()?;
        self.state_c = c.flatten_all()?.to_vec1()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal model state after loading
// ---------------------------------------------------------------------------

/// Internal model state after loading ONNX models
#[cfg(feature = "eou")]
struct EouModelState {
    /// Candle device
    candle_device: Device,
    /// ONNX encoder model
    encoder: candle_onnx::onnx::ModelProto,
    /// ONNX decoder model
    decoder: candle_onnx::onnx::ModelProto,
    /// Tokenizer
    tokenizer: tokenizers::Tokenizer,
    /// Mel filterbank matrix [128, 257]
    mel_basis: Array2<f32>,
    /// Blank token ID
    blank_id: i32,
    /// EOU token ID
    eou_id: i32,
}

#[cfg(not(feature = "eou"))]
struct EouModelState {
    _placeholder: (),
}

// ---------------------------------------------------------------------------
// Parakeet EOU Node
// ---------------------------------------------------------------------------

/// Parakeet EOU node for streaming ASR with end-of-utterance detection
pub struct ParakeetEOUNode {
    /// Node identifier
    node_id: String,
    /// Node configuration
    config: EouConfig,
    /// Selected inference device
    device: InferenceDevice,
    /// Model cache
    cache: ModelCache,
    /// Loaded model state (lazy initialization)
    model_state: RwLock<Option<EouModelState>>,
    /// Encoder cache for streaming
    encoder_cache: RwLock<EncoderCache>,
    /// Decoder LSTM state
    decoder_state: RwLock<DecoderState>,
    /// Audio buffer for streaming context
    audio_buffer: RwLock<VecDeque<f32>>,
}

/// EOU detection output
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EouOutput {
    /// Transcribed text for this chunk
    pub text: String,
    /// Whether end-of-utterance was detected
    pub eou_detected: bool,
    /// Cumulative transcript (accumulated across chunks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cumulative_text: Option<String>,
}

impl ParakeetEOUNode {
    /// Create a new Parakeet EOU node
    pub fn new(node_id: impl Into<String>, config: &EouConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| CandleNodeError::configuration(NODE_TYPE, e))?;

        let device = DeviceSelector::from_config(&config.device)?;

        Ok(Self {
            node_id: node_id.into(),
            config: config.clone(),
            device,
            cache: ModelCache::new(),
            model_state: RwLock::new(None),
            encoder_cache: RwLock::new(EncoderCache::new()),
            decoder_state: RwLock::new(DecoderState::new(1026)), // Default blank ID
            audio_buffer: RwLock::new(VecDeque::with_capacity(
                EOU_SAMPLE_RATE as usize * BUFFER_SIZE_SECONDS,
            )),
        })
    }

    /// Create from JSON parameters
    pub fn from_params(node_id: impl Into<String>, params: &Value) -> Result<Self> {
        let config = EouConfig::from_json(params)
            .map_err(|e| CandleNodeError::configuration(NODE_TYPE, e.to_string()))?;
        Self::new(node_id, &config)
    }

    /// Load ONNX models and tokenizer
    #[cfg(feature = "eou")]
    async fn load_model(&self) -> Result<()> {
        let mut state = self.model_state.write().await;

        if state.is_some() {
            return Ok(());
        }

        info!("Loading Parakeet EOU model");

        let candle_device: Device = (&self.device).try_into()?;

        // Download model files if not cached
        let encoder_path = self
            .cache
            .download_model(ONNX_REPO, &format!("{}/encoder.onnx", ONNX_SUBDIR), None)
            .await?;

        let decoder_path = self
            .cache
            .download_model(ONNX_REPO, &format!("{}/decoder_joint.onnx", ONNX_SUBDIR), None)
            .await?;

        let tokenizer_path = self
            .cache
            .download_model(ONNX_REPO, &format!("{}/tokenizer.json", ONNX_SUBDIR), None)
            .await?;

        // Load ONNX models
        info!("Loading encoder from {:?}", encoder_path);
        let encoder = candle_onnx::read_file(&encoder_path)
            .map_err(|e| CandleNodeError::model_load("parakeet-eou-encoder", e.to_string()))?;

        info!("Loading decoder from {:?}", decoder_path);
        let decoder = candle_onnx::read_file(&decoder_path)
            .map_err(|e| CandleNodeError::model_load("parakeet-eou-decoder", e.to_string()))?;

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            CandleNodeError::model_load("parakeet-eou-tokenizer", e.to_string())
        })?;

        let vocab_size = tokenizer.get_vocab_size(true);
        let blank_id = if (vocab_size as i32) < 1000 {
            1026
        } else {
            (vocab_size - 1) as i32
        };
        let eou_id = tokenizer
            .token_to_id("<EOU>")
            .map(|id| id as i32)
            .unwrap_or(1024);

        // Create mel filterbank
        let mel_basis = create_mel_filterbank_htk();

        info!(
            "Parakeet EOU model loaded (blank_id={}, eou_id={})",
            blank_id, eou_id
        );

        // Initialize decoder state with correct blank ID
        *self.decoder_state.write().await = DecoderState::new(blank_id);

        *state = Some(EouModelState {
            candle_device,
            encoder,
            decoder,
            tokenizer,
            mel_basis,
            blank_id,
            eou_id,
        });

        Ok(())
    }

    /// Process audio chunk and return EOU output
    #[cfg(feature = "eou")]
    async fn process_audio(&self, audio: AudioData) -> Result<EouOutput> {
        // Ensure model is loaded
        self.load_model().await?;

        let state = self.model_state.read().await;
        let state = state
            .as_ref()
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Model not loaded"))?;

        // Resample to 16kHz if needed
        let target_rate = EOU_SAMPLE_RATE;
        let resampled = if audio.sample_rate != target_rate {
            debug!(
                "Resampling from {} to {} Hz",
                audio.sample_rate, target_rate
            );
            audio.resample(target_rate)?
        } else {
            audio.clone()
        };

        // Convert to mono
        let mono = if resampled.channels > 1 {
            resampled.to_mono()
        } else {
            resampled
        };

        // Add to audio buffer
        {
            let mut buffer = self.audio_buffer.write().await;
            buffer.extend(mono.samples.iter().copied());

            // Trim buffer
            let max_samples = EOU_SAMPLE_RATE as usize * BUFFER_SIZE_SECONDS;
            while buffer.len() > max_samples {
                buffer.pop_front();
            }
        }

        // Check minimum buffer
        {
            let buffer = self.audio_buffer.read().await;
            if buffer.len() < MIN_BUFFER_SAMPLES {
                return Ok(EouOutput {
                    text: String::new(),
                    eou_detected: false,
                    cumulative_text: None,
                });
            }
        }

        // Extract mel features from full buffer
        let buffer_slice: Vec<f32> = self.audio_buffer.read().await.iter().copied().collect();
        let full_features = self.extract_mel_features(&buffer_slice, &state.mel_basis)?;
        let total_frames = full_features.shape()[2];

        // Slice to take only (pre_encode_cache + new_frames) for encoder
        let start_frame = total_frames.saturating_sub(SLICE_LEN);
        // The slice is along the last axis, so it is not contiguous: collect
        // it in logical (row-major) order rather than via `as_slice()`.
        let features: Vec<f32> = full_features
            .slice(s![.., .., start_frame..])
            .iter()
            .copied()
            .collect();

        let feature_frames = (total_frames - start_frame).min(SLICE_LEN);
        if feature_frames == 0 {
            return Ok(EouOutput {
                text: String::new(),
                eou_detected: false,
                cumulative_text: None,
            });
        }

        let feature_shape = [1, N_MELS, feature_frames];

        // Run encoder with cache (the cache is updated in place)
        let (encoder_data, encoder_shape) = {
            let mut enc_cache = self.encoder_cache.write().await;
            self.run_encoder(
                &features,
                &feature_shape,
                feature_frames as i64,
                &mut enc_cache,
                state,
            )?
        };

        // TODO(eou): the encoder output is assumed to be [batch, dim, frames]
        // (NeMo's channels-first convention, dim = 512). This has not been
        // checked against the altunenes/parakeet-rs ONNX export; if it is
        // [batch, frames, dim] instead, `encoder_dim` and `enc_frames` swap
        // and the frame gather below becomes a contiguous chunk.
        let [_, encoder_dim, enc_frames] = encoder_shape;
        if enc_frames == 0 {
            return Ok(EouOutput {
                text: String::new(),
                eou_detected: false,
                cumulative_text: None,
            });
        }

        // Run decoder for each frame
        let mut text_output = String::new();
        let mut eou_detected = false;

        // One lock for the whole chunk: the decoder state is threaded through
        // every frame in order.
        let mut dec_state = self.decoder_state.write().await;

        for t in 0..enc_frames {
            // Frame t of a row-major [1, dim, frames] buffer: element (d, t)
            // lives at d * frames + t.
            let frame: Vec<f32> = (0..encoder_dim)
                .map(|d| encoder_data[d * enc_frames + t])
                .collect();

            // TODO(eou): RNN-T greedy decoding should re-run the joint on the
            // same frame until it emits blank (up to MAX_SYMBOLS_PER_STEP);
            // this emits at most one symbol per frame.
            let (chunk_text, chunk_eou) = self.run_decoder_step(
                &frame,
                &mut dec_state,
                state,
                self.config.reset_on_eou,
            )?;

            text_output.push_str(&chunk_text);
            if chunk_eou {
                eou_detected = true;
                break;
            }
        }

        Ok(EouOutput {
            text: text_output,
            eou_detected,
            cumulative_text: None,
        })
    }

    /// Run the stateful encoder with cache
    #[cfg(feature = "eou")]
    fn run_encoder(
        &self,
        features: &[f32],
        feature_shape: &[usize; 3],
        length: i64,
        cache: &mut EncoderCache,
        state: &EouModelState,
    ) -> Result<(Vec<f32>, [usize; 3])> {
        let (ch_tensor, tm_tensor, len_tensor) = cache.to_tensors(&state.candle_device)?;

        let audio_signal =
            Tensor::from_slice(features, feature_shape.to_vec(), &state.candle_device)?;
        let length_tensor = Tensor::from_slice(&[length], 1, &state.candle_device)?;

        let inputs: std::collections::HashMap<String, Tensor> = [
            ("audio_signal".to_string(), audio_signal),
            ("length".to_string(), length_tensor),
            ("cache_last_channel".to_string(), ch_tensor),
            ("cache_last_time".to_string(), tm_tensor),
            ("cache_last_channel_len".to_string(), len_tensor),
        ]
        .into_iter()
        .collect();

        let outputs = candle_onnx::simple_eval(&state.encoder, inputs)
            .map_err(|e| CandleNodeError::inference(&self.node_id, e.to_string()))?;

        // Get output names from model graph
        let out_names = state
            .encoder
            .graph
            .as_ref()
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "No graph in encoder"))?
            .output
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>();

        // Extract encoder output
        let encoder_out = outputs
            .get("outputs")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing encoder output"))?;

        let out_shape: Vec<usize> =
            encoder_out.shape().dims().iter().map(|d| *d as usize).collect();
        let out_data: Vec<f32> = encoder_out.flatten_all()?.to_vec1()?;

        // Update cache
        let new_ch = outputs
            .get("new_cache_last_channel")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing new_cache_last_channel"))?;
        let new_tm = outputs
            .get("new_cache_last_time")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing new_cache_last_time"))?;
        let new_len = outputs
            .get("new_cache_last_channel_len")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing new_cache_last_channel_len"))?;

        *cache = EncoderCache::from_tensors(new_ch, new_tm, new_len)?;

        debug!(
            "Encoder output shape: {:?}, output names: {:?}",
            out_shape, out_names
        );

        let out_shape: [usize; 3] = out_shape.try_into().map_err(|shape: Vec<usize>| {
            CandleNodeError::inference(
                &self.node_id,
                format!("Expected a rank-3 encoder output, got shape {:?}", shape),
            )
        })?;

        Ok((out_data, out_shape))
    }

    /// Run a single decoder step
    #[cfg(feature = "eou")]
    fn run_decoder_step(
        &self,
        encoder_frame: &[f32], // [encoder_dim]
        dec_state: &mut DecoderState,
        state: &EouModelState,
        reset_on_eou: bool,
    ) -> Result<(String, bool)> {
        let (h_tensor, c_tensor) = dec_state.to_tensors(&state.candle_device)?;

        // Reshape encoder frame to [1, encoder_dim, 1]
        let encoder_outputs = Tensor::from_slice(
            encoder_frame,
            (1, encoder_frame.len(), 1),
            &state.candle_device,
        )?;

        // Target token (last predicted token)
        // TODO(eou): the decoder input/output names and the i32 dtype of
        // `targets`/`target_length` come from the original draft and have not
        // been checked against the ONNX graph.
        let targets = Tensor::from_slice(&[dec_state.last_token], (1, 1), &state.candle_device)?;

        let target_length = Tensor::from_slice(&[1i32], 1, &state.candle_device)?;

        let inputs: std::collections::HashMap<String, Tensor> = [
            ("encoder_outputs".to_string(), encoder_outputs),
            ("targets".to_string(), targets),
            ("target_length".to_string(), target_length),
            ("input_states_1".to_string(), h_tensor),
            ("input_states_2".to_string(), c_tensor),
        ]
        .into_iter()
        .collect();

        let outputs = candle_onnx::simple_eval(&state.decoder, inputs)
            .map_err(|e| CandleNodeError::inference(&self.node_id, e.to_string()))?;

        // Extract logits
        let logits = outputs
            .get("outputs")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing decoder logits"))?;

        let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let vocab_size = logits_vec.len();

        // Argmax
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val.is_finite() && val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        let token_id = max_idx as i32;

        // Check for blank or EOU
        if token_id == state.blank_id || token_id == 0 {
            return Ok((String::new(), false));
        }

        if token_id == state.eou_id {
            if reset_on_eou {
                dec_state.reset(state.blank_id);
            }
            return Ok((String::new(), true));
        }

        // Update decoder state
        let new_h = outputs
            .get("output_states_1")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing output_states_1"))?;
        let new_c = outputs
            .get("output_states_2")
            .ok_or_else(|| CandleNodeError::inference(&self.node_id, "Missing output_states_2"))?;

        dec_state.update_from_tensors(new_h, new_c)?;
        dec_state.last_token = token_id;

        // Decode token
        let decoded = state
            .tokenizer
            .decode(&[token_id as u32], true)
            .map_err(|e| CandleNodeError::inference(&self.node_id, e.to_string()))?;

        Ok((decoded, false))
    }

    /// Extract mel spectrogram features from audio
    #[cfg(feature = "eou")]
    fn extract_mel_features(&self, audio: &[f32], mel_basis: &Array2<f32>) -> Result<Array3<f32>> {
        // Apply preemphasis
        let audio_pre = apply_preemphasis(audio, PREEMPHASIS);

        // Compute STFT magnitude spectrum
        let spec = compute_stft_magnitude(&audio_pre, N_FFT, HOP_LENGTH, WIN_LENGTH)
            .map_err(|e| CandleNodeError::inference(&self.node_id, e.to_string()))?;

        // Apply mel filterbank
        let mel = mel_basis.dot(&spec);

        // Log scale
        let mel_log = mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln());

        // Add batch dimension: [1, N_MELS, T]
        Ok(mel_log.insert_axis(ndarray::Axis(0)))
    }

    /// Reset all internal state (call between separate audio streams)
    pub async fn reset_state(&self) {
        let blank_id = {
            let state = self.model_state.read().await;
            state.as_ref().map(|s| s.blank_id).unwrap_or(1026)
        };
        *self.encoder_cache.write().await = EncoderCache::new();
        self.decoder_state.write().await.reset(blank_id);
        self.audio_buffer.write().await.clear();
    }

    #[cfg(not(feature = "eou"))]
    async fn load_model(&self) -> Result<()> {
        Err(CandleNodeError::configuration(
            NODE_TYPE,
            "EOU feature not enabled at compile time",
        ))
    }

    #[cfg(not(feature = "eou"))]
    async fn process_audio(&self, _audio: AudioData) -> Result<EouOutput> {
        Err(CandleNodeError::configuration(
            NODE_TYPE,
            "EOU feature not enabled at compile time",
        ))
    }
}

// ---------------------------------------------------------------------------
// Audio processing utilities
// ---------------------------------------------------------------------------

/// Apply preemphasis filter to audio
fn apply_preemphasis(audio: &[f32], coeff: f32) -> Vec<f32> {
    if coeff.abs() < 1e-6 {
        return audio.to_vec();
    }
    let mut output = Vec::with_capacity(audio.len());
    output.push(audio[0]);
    for i in 1..audio.len() {
        output.push(audio[i] - coeff * audio[i - 1]);
    }
    output
}

/// Compute STFT magnitude spectrum
fn compute_stft_magnitude(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> std::result::Result<Array2<f32>, String> {
    let num_frames = (audio.len() - win_length) / hop_length + 1;
    let num_freqs = n_fft / 2 + 1;

    // Hann window
    let mut window = vec![0.0f32; win_length];
    for i in 0..win_length {
        // TODO(eou): this is the periodic Hann window (divides by N). Check
        // which variant the model's NeMo preprocessor was trained with.
        window[i] =
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos());
    }

    let mut magnitude = Array2::zeros((num_freqs, num_frames));

    for frame in 0..num_frames {
        let start = frame * hop_length;

        // Apply window and compute FFT
        let mut framed = vec![0.0f32; n_fft];
        for i in 0..win_length {
            framed[i] = audio[start + i] * window[i];
        }

        // Simple DFT (for production, use a proper FFT library)
        for k in 0..num_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for n in 0..n_fft {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                real += framed[n] * angle.cos() as f32;
                imag += framed[n] * angle.sin() as f32;
            }
            magnitude[[k, frame]] = (real * real + imag * imag).sqrt();
        }
    }

    Ok(magnitude)
}

/// Create HTK mel filterbank matrix
fn create_mel_filterbank_htk() -> Array2<f32> {
    let num_freqs = N_FFT / 2 + 1;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(FMAX);

    let mel_points: Vec<f32> = (0..=N_MELS + 1)
        .map(|i| {
            mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
        })
        .collect();

    let fft_freqs: Vec<f32> = (0..num_freqs)
        .map(|i| (EOU_SAMPLE_RATE as f32 / N_FFT as f32) * i as f32)
        .collect();

    let mut weights = Array2::zeros((N_MELS, num_freqs));

    for i in 0..N_MELS {
        let left = mel_points[i];
        let center = mel_points[i + 1];
        let right = mel_points[i + 2];
        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= left && freq <= center {
                weights[[i, j]] = (freq - left) / (center - left);
            } else if freq > center && freq <= right {
                weights[[i, j]] = (right - freq) / (right - center);
            }
        }
    }

    // Normalize
    for i in 0..N_MELS {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for j in 0..num_freqs {
            weights[[i, j]] *= enorm;
        }
    }

    weights
}

// ---------------------------------------------------------------------------
// AsyncStreamingNode implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl AsyncStreamingNode for ParakeetEOUNode {
    fn node_type(&self) -> &str {
        NODE_TYPE
    }

    async fn initialize(&self, _ctx: &dyn InitializeContextRead) -> std::result::Result<(), Error> {
        self.load_model()
            .await
            .map_err(|e| Error::Execution(e.to_string()))
    }

    async fn process(&self, data: RuntimeData) -> std::result::Result<RuntimeData, Error> {
        // Extract audio from input
        let audio = RuntimeDataConverter::extract_audio(&data, &self.node_id)
            .map_err(|e| Error::Execution(e.to_string()))?;

        // Run EOU detection
        let output = self
            .process_audio(audio)
            .await
            .map_err(|e| Error::Execution(e.to_string()))?;

        // Return as JSON
        let json = serde_json::to_value(&output).map_err(|e| Error::Execution(e.to_string()))?;

        Ok(RuntimeData::Json(json))
    }
}

// ---------------------------------------------------------------------------
// StreamingNode wrapper
// ---------------------------------------------------------------------------

/// Wrapper to make ParakeetEOUNode a StreamingNode
pub struct ParakeetEOUNodeWrapper(pub Arc<ParakeetEOUNode>);

#[async_trait]
impl StreamingNode for ParakeetEOUNodeWrapper {
    fn node_type(&self) -> &str {
        self.0.node_type()
    }

    fn node_id(&self) -> &str {
        &self.0.node_id
    }

    async fn initialize(&self, ctx: &dyn InitializeContextRead) -> std::result::Result<(), Error> {
        AsyncStreamingNode::initialize(self.0.as_ref(), ctx).await
    }

    async fn process_async(
        &self,
        data: RuntimeData,
        _ctx: &dyn NodeRuntimeContextRead,
    ) -> std::result::Result<RuntimeData, Error> {
        self.0.process(data).await
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
                sample_rate: Some(ConstraintValue::Exact(EOU_SAMPLE_RATE)),
                channels: Some(ConstraintValue::Exact(1)),
                ..Default::default()
            },
        )))
    }

    fn capability_behavior(&self) -> CapabilityBehavior {
        CapabilityBehavior::Static
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Factory for creating ParakeetEOUNode instances
pub struct ParakeetEOUNodeFactory;

impl ParakeetEOUNodeFactory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ParakeetEOUNodeFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingNodeFactory for ParakeetEOUNodeFactory {
    fn create(
        &self,
        node_id: String,
        params: &Value,
        _session_id: Option<String>,
    ) -> std::result::Result<Box<dyn StreamingNode>, Error> {
        let node = ParakeetEOUNode::from_params(node_id, params)
            .map_err(|e| Error::Execution(e.to_string()))?;
        Ok(Box::new(ParakeetEOUNodeWrapper(Arc::new(node))))
    }

    fn node_type(&self) -> &str {
        NODE_TYPE
    }

    fn media_capabilities(&self, params: &Value) -> Option<MediaCapabilities> {
        let config = EouConfig::from_json(params).ok()?;
        Some(MediaCapabilities::with_input(MediaConstraints::Audio(
            AudioConstraints {
                sample_rate: Some(ConstraintValue::Exact(EOU_SAMPLE_RATE)),
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
    fn test_eou_config_default() {
        let config = EouConfig::default();
        assert_eq!(config.device, "auto");
        assert!(config.reset_on_eou);
        assert!(config.output_json);
    }

    #[test]
    fn test_eou_node_creation() {
        let config = EouConfig::default();
        let node = ParakeetEOUNode::new("test-eou", &config);
        assert!(node.is_ok());
    }

    #[test]
    fn test_factory_node_type() {
        let factory = ParakeetEOUNodeFactory::new();
        assert_eq!(factory.node_type(), NODE_TYPE);
    }

    #[test]
    fn test_encoder_cache_init() {
        let cache = EncoderCache::new();
        assert_eq!(cache.cache_last_channel.len(), 17 * 1 * 70 * 512);
        assert_eq!(cache.cache_last_time.len(), 17 * 1 * 512 * 8);
        assert_eq!(cache.cache_last_channel_len, vec![0i64]);
    }

    #[test]
    fn test_decoder_state_init() {
        let state = DecoderState::new(1026);
        assert_eq!(state.state_h.len(), 640);
        assert_eq!(state.state_c.len(), 640);
        assert_eq!(state.last_token, 1026);
    }

    #[test]
    fn test_preemphasis() {
        let audio = vec![1.0, 2.0, 3.0, 4.0];
        let result = apply_preemphasis(&audio, 0.97);
        assert_eq!(result.len(), audio.len());
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let mel_basis = create_mel_filterbank_htk();
        assert_eq!(mel_basis.shape(), &[N_MELS, N_FFT / 2 + 1]);
    }
}
