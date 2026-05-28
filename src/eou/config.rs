//! Configuration for Parakeet EOU (End-of-Utterance) detection node

use serde::{Deserialize, Serialize};

/// Sample rate for Parakeet EOU model
/// The model expects 16kHz mono audio
pub const EOU_SAMPLE_RATE: u32 = 16000;

/// FFT parameters for mel feature extraction
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const PREEMPHASIS: f32 = 0.97;
pub const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
pub const FMAX: f32 = 8000.0;

/// Pre-encode cache frames for streaming context
pub const PRE_ENCODE_CACHE: usize = 9;

/// Frames generated per 160ms chunk at 16kHz
pub const FRAMES_PER_CHUNK: usize = 16;

/// Total frames sliced for encoder input
pub const SLICE_LEN: usize = PRE_ENCODE_CACHE + FRAMES_PER_CHUNK;

/// Audio buffer size: 4 seconds for feature extraction context
pub const BUFFER_SIZE_SECONDS: usize = 4;

/// Minimum buffer samples before processing (1 second)
pub const MIN_BUFFER_SAMPLES: usize = EOU_SAMPLE_RATE as usize;

/// Maximum symbols added per decoder step
pub const MAX_SYMBOLS_PER_STEP: usize = 5;

/// Configuration for ParakeetEOUNode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EouConfig {
    /// Inference device ("auto", "cpu", "cuda", "metal")
    #[serde(default = "default_device")]
    pub device: String,

    /// Whether to reset decoder state when EOU is detected
    /// When true, decoder LSTM state is reset but encoder cache is preserved
    #[serde(default = "default_reset_on_eou")]
    pub reset_on_eou: bool,

    /// Whether to output structured JSON with EOU flag
    /// When false, outputs plain text with [EOU] suffix
    #[serde(default = "default_output_json")]
    pub output_json: bool,

    /// Model revision/tag for HuggingFace download (optional)
    /// Defaults to the ONNX export from altunenes/parakeet-rs
    #[serde(default)]
    pub model_revision: Option<String>,
}

fn default_device() -> String {
    "auto".to_string()
}

fn default_reset_on_eou() -> bool {
    true
}

fn default_output_json() -> bool {
    true
}

impl Default for EouConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            reset_on_eou: default_reset_on_eou(),
            output_json: default_output_json(),
            model_revision: None,
        }
    }
}

impl EouConfig {
    /// Create config from JSON value
    pub fn from_json(value: &serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value.clone())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Device validation is handled by DeviceSelector
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EouConfig::default();
        assert_eq!(config.device, "auto");
        assert!(config.reset_on_eou);
        assert!(config.output_json);
        assert!(config.model_revision.is_none());
    }

    #[test]
    fn test_config_from_json() {
        let json = serde_json::json!({
            "device": "cpu",
            "reset_on_eou": false,
            "output_json": false
        });
        let config = EouConfig::from_json(&json).unwrap();
        assert_eq!(config.device, "cpu");
        assert!(!config.reset_on_eou);
        assert!(!config.output_json);
    }

    #[test]
    fn test_config_constants() {
        assert_eq!(EOU_SAMPLE_RATE, 16000);
        assert_eq!(N_FFT, 512);
        assert_eq!(N_MELS, 128);
        assert_eq!(HOP_LENGTH, 160);
        assert_eq!(PRE_ENCODE_CACHE, 9);
        assert_eq!(FRAMES_PER_CHUNK, 16);
    }
}
