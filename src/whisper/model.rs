//! Configuration helper types for the Whisper speech-to-text node.
//!
//! After the `#[node]` migration the runtime config struct lives in
//! `mod.rs` (auto-generated as `WhisperNodeConfig`). This module just
//! holds the standalone `WhisperModel` enum that is referenced as a
//! field type, plus a small extension trait carrying model-id helpers.

use serde::{Deserialize, Serialize};

/// Whisper model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum WhisperModel {
    /// Tiny model (~39M parameters)
    Tiny,
    /// Base model (~74M parameters)
    Base,
    /// Small model (~244M parameters)
    Small,
    /// Medium model (~769M parameters)
    Medium,
    /// Large v3 model (~1.5B parameters)
    #[serde(rename = "large-v3")]
    LargeV3,
}

impl WhisperModel {
    /// Get the HuggingFace model ID
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::Base => "openai/whisper-base",
            Self::Small => "openai/whisper-small",
            Self::Medium => "openai/whisper-medium",
            Self::LargeV3 => "openai/whisper-large-v3",
        }
    }

    /// Get the config filename
    pub fn config_file(&self) -> &'static str {
        "config.json"
    }

    /// Get the model weights filename
    pub fn weights_file(&self) -> &'static str {
        "model.safetensors"
    }

    /// Get the tokenizer filename
    pub fn tokenizer_file(&self) -> &'static str {
        "tokenizer.json"
    }

    /// Get approximate model size in bytes
    pub fn approx_size(&self) -> u64 {
        match self {
            Self::Tiny => 39_000_000,
            Self::Base => 74_000_000,
            Self::Small => 244_000_000,
            Self::Medium => 769_000_000,
            Self::LargeV3 => 1_500_000_000,
        }
    }

    /// Check if model supports multiple languages
    pub fn is_multilingual(&self) -> bool {
        true
    }
}

impl Default for WhisperModel {
    fn default() -> Self {
        Self::Base
    }
}

impl std::fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::Base => write!(f, "base"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::LargeV3 => write!(f, "large-v3"),
        }
    }
}

impl std::str::FromStr for WhisperModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "base" => Ok(Self::Base),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large-v3" | "large" => Ok(Self::LargeV3),
            other => Err(format!("Unknown Whisper model: {}", other)),
        }
    }
}

/// Backwards-compatibility marker trait so callers can do
/// `use remotemedia_candle_nodes::whisper::WhisperModelExt;` and get
/// at the same helpers without changing import paths if we move the
/// enum again later.
pub trait WhisperModelExt {
    fn model_id(&self) -> &'static str;
}

impl WhisperModelExt for WhisperModel {
    fn model_id(&self) -> &'static str {
        WhisperModel::model_id(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id() {
        assert_eq!(WhisperModel::Base.model_id(), "openai/whisper-base");
        assert_eq!(WhisperModel::LargeV3.model_id(), "openai/whisper-large-v3");
    }
}
