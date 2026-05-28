//! Candle ML Inference Nodes for RemoteMedia Pipelines
//!
//! This crate provides native Rust-based machine learning inference nodes
//! powered by the Hugging Face Candle framework. Nodes implement the
//! `StreamingNode` trait for seamless pipeline integration.
//!
//! # Features
//!
//! - `whisper` - Speech-to-text transcription via Whisper models
//! - `yolo` - Object detection via YOLO models
//! - `llm` - Text generation via Phi and LLaMA models
//! - `cuda` - NVIDIA GPU acceleration
//! - `metal` - Apple GPU acceleration
//!
//! # Example
//!
//! ```ignore
//! use remotemedia_candle_nodes::WhisperNode;
//!
//! let node = WhisperNode::new("whisper-1", &config)?;
//! node.initialize().await?;
//! let result = node.process_async(audio_data).await?;
//! ```

pub mod error;

#[cfg(feature = "whisper")]
pub mod whisper;

#[cfg(feature = "yolo")]
pub mod yolo;

#[cfg(feature = "llm")]
pub mod llm;

#[cfg(feature = "vad")]
pub mod vad;
#[cfg(feature = "eou")]
pub mod eou;

/// Emotion vector extraction and steering nodes.
///
/// Implements the two-phase pipeline from Anthropic's 2026 emotion-vector
/// research: extraction (offline) and steering (runtime).
pub mod emotion;

mod cache;
mod cli;
mod convert;
mod device;
mod registry;
mod schema;

pub use cache::{CacheStats, CachedModel, ModelCache};
pub use cli::{list_available_models, ModelCli};
pub use convert::{RuntimeDataConverter, TensorExt};
pub use device::{DeviceSelector, InferenceDevice};
pub use error::{CandleNodeError, Result};
pub use registry::{register_candle_nodes, CandleNodeFactory, CandleNodesProvider};
pub use schema::{all_schemas, export_schemas_json, MediaType, NodeSchema};

#[cfg(feature = "whisper")]
pub use whisper::{WhisperConfig, WhisperNode, WhisperNodeFactory};

#[cfg(feature = "yolo")]
pub use yolo::{Detection, DetectionResult, YoloConfig, YoloNode, YoloNodeFactory};

#[cfg(feature = "llm")]
pub use llm::{GenerationConfig, LlamaNode, LlmConfig, PhiNode};

#[cfg(feature = "vad")]
pub use vad::{
    SileroVadNode, SileroVadNodeFactory, SpeechSegment, VadConfig, VadOutput, VadSampleRate,
};

#[cfg(feature = "eou")]
pub use eou::{
    EouConfig, EouOutput, ParakeetEOUNode, ParakeetEOUNodeFactory,
};

// Emotion vector extraction and steering
pub use emotion::{
    load_emotion_vector, save_emotion_vector, EmotionExtractConfig, EmotionExtractorNode,
    EmotionExtractorNodeFactory, EmotionSteerConfig, EmotionSteeringNode,
    EmotionSteeringNodeFactory, EmotionVectorMetadata, PoolingMode, SteeringVectorConfig,
};
