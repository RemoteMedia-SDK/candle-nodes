//! A host that links candle-nodes registers its nodes through the plugin
//! SDK's `NodeFactoryProvider`, without candle-nodes depending on
//! remotemedia-sdk-base. remotemedia-sdk-base is only a dev-dependency here,
//! standing in for the host.

// Name the crate so it is linked into this test binary, as a host would.
use remotemedia_candle_nodes as _;

use remotemedia_core::nodes::streaming_registry::create_default_streaming_registry;
use remotemedia_core::nodes::StreamingNodeRegistry;

fn has(types: &[String], node_type: &str) -> bool {
    types.iter().any(|t| t == node_type)
}

#[test]
fn default_registry_has_candle_nodes() {
    let types = create_default_streaming_registry().list_types();
    assert!(has(&types, "EmotionExtractorNode"), "{types:?}");
    assert!(has(&types, "EmotionSteeringNode"), "{types:?}");
    #[cfg(feature = "whisper")]
    assert!(has(&types, "candle-whisper"), "{types:?}");
}

#[test]
fn builder_defaults_have_candle_nodes() {
    let types = StreamingNodeRegistry::builder().with_defaults().build().list_types();
    assert!(has(&types, "EmotionExtractorNode"), "{types:?}");
    #[cfg(feature = "whisper")]
    assert!(has(&types, "candle-whisper"), "{types:?}");
}

#[test]
fn candle_provider_is_listed_by_the_host() {
    let provider = remotemedia_core::nodes::iter_providers()
        .find(|provider| provider.provider_name() == "candle-nodes")
        .expect("the host lists candle-nodes' SDK provider");
    assert_eq!(provider.priority(), 800);
}
