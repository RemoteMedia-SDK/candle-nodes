//! Node factory registration for Candle ML nodes
//!
//! Candle nodes reach a host two ways:
//!
//! - **Linked into the host.** `CandleNodesProvider` is submitted as a plugin
//!   SDK `NodeFactoryProvider`, so every registry the host builds from its
//!   defaults registers these nodes. That needs no dependency on
//!   `remotemedia-sdk-base`: the host collects SDK providers itself.
//! - **Loaded as a plugin.** The `whisper` plugin exports `WhisperNode`
//!   through `plugin_export!()` (the `loadable-export` feature); no provider
//!   is involved.
//!
//! Nodes are registered based on enabled feature flags.

use remotemedia_traits::provider::NodeFactoryProvider;
use remotemedia_traits::StreamingNodeFactory;
use std::sync::Arc;

/// Trait for Candle node factories
pub trait CandleNodeFactory: StreamingNodeFactory {
    /// Get the model family this factory creates nodes for
    fn model_family(&self) -> &'static str;

    /// Get supported model variants
    fn supported_variants(&self) -> &[&'static str];
}

/// The factories of every enabled Candle node.
pub fn candle_node_factories() -> Vec<Arc<dyn StreamingNodeFactory>> {
    let mut factories: Vec<Arc<dyn StreamingNodeFactory>> = Vec::new();

    #[cfg(feature = "whisper")]
    factories.push(Arc::new(crate::whisper::WhisperNodeFactory::new()));

    #[cfg(feature = "yolo")]
    factories.push(Arc::new(crate::yolo::YoloNodeFactory::new()));

    #[cfg(feature = "llm")]
    {
        factories.push(Arc::new(crate::llm::PhiNodeFactory::new()));
        factories.push(Arc::new(crate::llm::LlamaNodeFactory::new()));
    }

    #[cfg(feature = "vad")]
    factories.push(Arc::new(crate::vad::SileroVadNodeFactory::new()));

    #[cfg(feature = "eou")]
    factories.push(Arc::new(crate::eou::ParakeetEOUNodeFactory::new()));

    // Emotion vector nodes (always available, no feature flag)
    factories.push(Arc::new(crate::emotion::EmotionExtractorNodeFactory::new()));
    factories.push(Arc::new(crate::emotion::EmotionSteeringNodeFactory::new()));

    factories
}

/// Provider for Candle ML inference nodes.
///
/// Automatically registers nodes based on enabled feature flags:
/// - `whisper` - Speech-to-text via Whisper models
/// - `yolo` - Object detection via YOLO models
/// - `llm` - Text generation via Phi and LLaMA models
/// - `vad` - Voice activity detection via Silero VAD
/// - `eou` - End-of-utterance detection via Parakeet
///
/// Priority is 800 (below core nodes at 1000, above Python nodes at 500).
pub struct CandleNodesProvider;

impl NodeFactoryProvider for CandleNodesProvider {
    fn provider_name(&self) -> &'static str {
        "candle-nodes"
    }

    fn factories(&self) -> Vec<Arc<dyn StreamingNodeFactory>> {
        let factories = candle_node_factories();
        tracing::debug!(count = factories.len(), "Candle node factories");
        factories
    }

    fn priority(&self) -> i32 {
        // Below core nodes (1000), above Python nodes (500)
        800
    }
}

// Auto-register the Candle nodes provider with any host that links this crate.
remotemedia_traits::inventory::submit! {
    &CandleNodesProvider as &'static dyn NodeFactoryProvider
}

/// Get a list of all registered Candle node types
pub fn list_candle_node_types() -> Vec<&'static str> {
    let mut types = Vec::new();

    #[cfg(feature = "whisper")]
    types.push("candle-whisper");

    #[cfg(feature = "yolo")]
    types.push("candle-yolo");

    #[cfg(feature = "llm")]
    {
        types.push("candle-phi");
        types.push("candle-llama");
    }

    #[cfg(feature = "vad")]
    types.push("candle-silero-vad");

    #[cfg(feature = "eou")]
    types.push("parakeet-eou");

    // Emotion vector nodes (always available)
    types.push("EmotionExtractorNode");
    types.push("EmotionSteeringNode");

    types
}

/// Check if a node type is a Candle node
pub fn is_candle_node(node_type: &str) -> bool {
    matches!(
        node_type,
        "candle-whisper"
            | "candle-yolo"
            | "candle-phi"
            | "candle-llama"
            | "candle-silero-vad"
            | "EmotionExtractorNode"
            | "EmotionSteeringNode"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_node_types() {
        let types = list_candle_node_types();
        // At minimum, should be empty if no features enabled
        assert!(types.len() <= 5);
    }

    #[test]
    fn test_is_candle_node() {
        assert!(is_candle_node("candle-whisper"));
        assert!(is_candle_node("candle-yolo"));
        assert!(!is_candle_node("some-other-node"));
    }

    #[test]
    fn test_provider_metadata() {
        let provider = CandleNodesProvider;
        assert_eq!(provider.provider_name(), "candle-nodes");
        assert_eq!(provider.priority(), 800);
    }

    #[test]
    fn provider_is_submitted_to_the_plugin_sdk() {
        let provider = remotemedia_traits::iter_factory_providers()
            .find(|provider| provider.provider_name() == "candle-nodes")
            .expect("CandleNodesProvider is submitted");
        let types: Vec<String> = provider
            .factories()
            .iter()
            .map(|factory| factory.node_type().to_string())
            .collect();
        assert!(types.iter().any(|t| t == "EmotionExtractorNode"), "{types:?}");
        #[cfg(feature = "whisper")]
        assert!(types.iter().any(|t| t == "candle-whisper"), "{types:?}");
    }
}
