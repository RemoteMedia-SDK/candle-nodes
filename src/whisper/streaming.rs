use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::error::{CandleNodeError, Result};

pub(crate) const WHISPER_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodedSegment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamingOptions {
    pub decode_interval_ms: u32,
    pub overlap_ms: u32,
    pub max_window_ms: u32,
    pub agreement_updates: usize,
    pub trailing_guard_words: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DecodeWindow {
    pub samples: Vec<f32>,
    pub start_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TranscriptEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub revision: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replaces: Option<u64>,
    pub text: String,
    pub stable_text: String,
    pub unstable_text: String,
    pub audio_start_ms: u64,
    pub audio_end_ms: u64,
}

#[derive(Debug, Default)]
struct StreamingNormalizer {
    sample_rate: u32,
    channels: u32,
    pending: Vec<f32>,
    source_position: f64,
}

impl StreamingNormalizer {
    fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u32) -> Result<Vec<f32>> {
        if sample_rate == 0 || channels == 0 {
            return Err(CandleNodeError::invalid_input(
                "Whisper",
                "non-zero sample rate and channel count",
                format!("{sample_rate} Hz, {channels} channels"),
            ));
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(CandleNodeError::invalid_input(
                "Whisper",
                "complete interleaved audio frames",
                format!("{} samples for {channels} channels", samples.len()),
            ));
        }
        if self.sample_rate == 0 {
            self.sample_rate = sample_rate;
            self.channels = channels;
        } else if self.sample_rate != sample_rate || self.channels != channels {
            return Err(CandleNodeError::invalid_input(
                "Whisper",
                format!("{} Hz, {} channels", self.sample_rate, self.channels),
                format!("{sample_rate} Hz, {channels} channels"),
            ));
        }

        let mono: Vec<f32> = if channels == 1 {
            samples.to_vec()
        } else {
            samples
                .chunks_exact(channels as usize)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        if sample_rate == WHISPER_SAMPLE_RATE {
            return Ok(mono);
        }

        self.pending.extend(mono);
        let step = sample_rate as f64 / WHISPER_SAMPLE_RATE as f64;
        let mut output = Vec::new();
        while self.source_position + 1.0 < self.pending.len() as f64 {
            let left = self.source_position.floor() as usize;
            let fraction = (self.source_position - left as f64) as f32;
            output.push(self.pending[left] * (1.0 - fraction) + self.pending[left + 1] * fraction);
            self.source_position += step;
        }

        let consumed = (self.source_position.floor() as usize).saturating_sub(1);
        if consumed > 0 {
            self.pending.drain(..consumed);
            self.source_position -= consumed as f64;
        }
        Ok(output)
    }
}

#[derive(Debug, Default)]
pub(crate) struct WhisperStreamingState {
    normalizer: StreamingNormalizer,
    audio: VecDeque<f32>,
    buffer_start_sample: u64,
    total_samples: u64,
    new_samples_since_decode: usize,
    hypotheses: VecDeque<Vec<DecodedSegment>>,
    committed: Vec<DecodedSegment>,
    unstable: Vec<DecodedSegment>,
    committed_through_ms: u64,
    revision: u64,
    last_emitted_text: Option<String>,
    finalized: bool,
}

impl WhisperStreamingState {
    pub fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: u32,
    ) -> Result<usize> {
        if self.finalized {
            return Err(CandleNodeError::invalid_input(
                "Whisper",
                "audio before final flush",
                "audio after final flush",
            ));
        }
        let normalized = self.normalizer.push(samples, sample_rate, channels)?;
        let count = normalized.len();
        self.audio.extend(normalized);
        self.total_samples += count as u64;
        self.new_samples_since_decode += count;
        Ok(count)
    }

    pub fn decode_window(
        &mut self,
        options: StreamingOptions,
        force: bool,
    ) -> Option<DecodeWindow> {
        if self.audio.is_empty() {
            return None;
        }
        // Stabilize the part of the previous hypothesis that is about to
        // leave the rolling window before evicting its audio. Runtime
        // cancellation or decoder errors must still never allow queued audio
        // to grow beyond Whisper's encoder limit.
        self.force_commit_for_bound(options);
        self.trim_audio(options, false);
        let interval_samples = ms_to_samples(options.decode_interval_ms as u64) as usize;
        if !force && self.new_samples_since_decode < interval_samples {
            return None;
        }
        self.new_samples_since_decode = 0;
        Some(DecodeWindow {
            samples: self.audio.iter().copied().collect(),
            start_ms: samples_to_ms(self.buffer_start_sample),
        })
    }

    pub fn apply_hypothesis(
        &mut self,
        relative_segments: Vec<DecodedSegment>,
        window_start_ms: u64,
        options: StreamingOptions,
        final_result: bool,
    ) -> Option<TranscriptEvent> {
        let mut segments: Vec<DecodedSegment> = relative_segments
            .into_iter()
            .map(|mut segment| {
                segment.text = normalize_text(&segment.text);
                segment.start_ms += window_start_ms;
                segment.end_ms += window_start_ms;
                segment
            })
            .flat_map(split_segment_words)
            .filter(|segment| {
                segment.end_ms > self.committed_through_ms && !segment.text.is_empty()
            })
            .collect();
        segments.sort_by_key(|segment| (segment.start_ms, segment.end_ms));
        let overlap = committed_prefix_overlap(&self.committed, &segments);
        if overlap > 0 {
            segments.drain(..overlap);
        }

        if final_result {
            self.commit_segments(segments);
            self.unstable.clear();
            self.hypotheses.clear();
            self.finalized = true;
            let event = self.make_event("final", true);
            self.trim_audio(options, true);
            return Some(event);
        }

        self.hypotheses.push_back(segments.clone());
        while self.hypotheses.len() > options.agreement_updates.max(2) {
            self.hypotheses.pop_front();
        }

        if self.hypotheses.len() >= options.agreement_updates.max(2) {
            let common_words = common_prefix_words(self.hypotheses.iter());
            let eligible_words = common_words.saturating_sub(options.trailing_guard_words);
            let commit_count = complete_segments_within_words(&segments, eligible_words);
            if commit_count > 0 {
                self.commit_segments(segments.drain(..commit_count).collect());
                self.hypotheses.clear();
            }
        }

        self.unstable = segments;
        self.force_commit_for_bound(options);
        self.trim_audio(options, false);

        let text = self.full_text();
        if self.last_emitted_text.as_deref() == Some(text.as_str()) || text.is_empty() {
            return None;
        }
        let kind = if self.last_emitted_text.is_some() {
            "correction"
        } else {
            "partial"
        };
        Some(self.make_event(kind, false))
    }

    fn force_commit_for_bound(&mut self, options: StreamingOptions) {
        let max_window_start_ms = samples_to_ms(
            self.total_samples
                .saturating_sub(ms_to_samples(options.max_window_ms as u64)),
        );
        if max_window_start_ms <= samples_to_ms(self.buffer_start_sample) {
            return;
        }

        let required_commit_ms = max_window_start_ms + options.overlap_ms as u64;
        let mut count = self
            .unstable
            .iter()
            .take_while(|segment| segment.end_ms <= required_commit_ms)
            .count();
        if count == 0 && !self.unstable.is_empty() {
            count = 1;
        }
        if count > 0 {
            let forced: Vec<_> = self.unstable.drain(..count).collect();
            self.commit_segments(forced);
            self.hypotheses.clear();
        }
    }

    fn commit_segments(&mut self, segments: Vec<DecodedSegment>) {
        for segment in segments {
            self.committed_through_ms = self.committed_through_ms.max(segment.end_ms);
            self.committed.push(segment);
        }
    }

    fn trim_audio(&mut self, options: StreamingOptions, final_result: bool) {
        let target_ms = if final_result {
            samples_to_ms(self.total_samples)
        } else {
            self.committed_through_ms
                .saturating_sub(options.overlap_ms as u64)
        };
        let target_sample = ms_to_samples(target_ms).min(self.total_samples);
        let remove = target_sample.saturating_sub(self.buffer_start_sample) as usize;
        let remove = remove.min(self.audio.len());
        self.audio.drain(..remove);
        self.buffer_start_sample += remove as u64;

        self.enforce_audio_bound(options);
    }

    fn enforce_audio_bound(&mut self, options: StreamingOptions) {
        let max_samples = ms_to_samples(options.max_window_ms as u64) as usize;
        if self.audio.len() > max_samples {
            let extra = self.audio.len() - max_samples;
            self.audio.drain(..extra);
            self.buffer_start_sample += extra as u64;
        }
    }

    fn make_event(&mut self, kind: &str, final_result: bool) -> TranscriptEvent {
        let text = self.full_text();
        let changed = self.last_emitted_text.as_deref() != Some(text.as_str());
        let replaces = (changed && self.revision > 0).then_some(self.revision);
        if changed || self.revision == 0 {
            self.revision += 1;
        }
        self.last_emitted_text = Some(text.clone());

        TranscriptEvent {
            kind: kind.to_string(),
            revision: self.revision,
            replaces: if kind == "correction" || (final_result && changed) {
                replaces
            } else {
                None
            },
            text,
            stable_text: join_segments(&self.committed),
            unstable_text: if final_result {
                String::new()
            } else {
                join_segments(&self.unstable)
            },
            audio_start_ms: samples_to_ms(self.buffer_start_sample),
            audio_end_ms: samples_to_ms(self.total_samples),
        }
    }

    fn full_text(&self) -> String {
        join_nonempty(
            &join_segments(&self.committed),
            &join_segments(&self.unstable),
        )
    }

    #[cfg(test)]
    fn retained_samples(&self) -> usize {
        self.audio.len()
    }
}

fn common_prefix_words<'a>(hypotheses: impl Iterator<Item = &'a Vec<DecodedSegment>>) -> usize {
    let words: Vec<Vec<&str>> = hypotheses
        .map(|segments| {
            segments
                .iter()
                .flat_map(|segment| segment.text.split_whitespace())
                .collect()
        })
        .collect();
    let Some(first) = words.first() else {
        return 0;
    };
    (0..first.len())
        .take_while(|&index| {
            words
                .iter()
                .all(|candidate| candidate.get(index) == first.get(index))
        })
        .count()
}

fn complete_segments_within_words(segments: &[DecodedSegment], word_limit: usize) -> usize {
    let mut words = 0;
    segments
        .iter()
        .take_while(|segment| {
            words += segment.text.split_whitespace().count();
            words <= word_limit
        })
        .count()
}

fn committed_prefix_overlap(committed: &[DecodedSegment], candidate: &[DecodedSegment]) -> usize {
    let limit = committed.len().min(candidate.len());
    (1..=limit)
        .rev()
        .find(|&length| {
            committed[committed.len() - length..]
                .iter()
                .zip(&candidate[..length])
                .all(|(left, right)| comparison_word(&left.text) == comparison_word(&right.text))
        })
        .unwrap_or(0)
}

fn comparison_word(word: &str) -> String {
    word.chars()
        .filter(|character| character.is_alphanumeric() || *character == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

fn join_segments(segments: &[DecodedSegment]) -> String {
    segments
        .iter()
        .map(|segment| segment.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whisper timestamps delimit phrases rather than individual words. Local
/// agreement operates at word boundaries, so distribute each word across its
/// model-derived phrase interval. These interpolated times are used only after
/// the same word has appeared in multiple decodes; the phrase endpoints remain
/// the authoritative bounds.
fn split_segment_words(segment: DecodedSegment) -> Vec<DecodedSegment> {
    let words: Vec<_> = segment.text.split_whitespace().collect();
    if words.len() <= 1 {
        return vec![segment];
    }
    let word_count = words.len() as u64;
    let duration = segment.end_ms.saturating_sub(segment.start_ms);
    words
        .into_iter()
        .enumerate()
        .map(|(index, word)| DecodedSegment {
            text: word.to_string(),
            start_ms: segment.start_ms + duration * index as u64 / word_count,
            end_ms: segment.start_ms + duration * (index + 1) as u64 / word_count,
        })
        .collect()
}

fn join_nonempty(left: &str, right: &str) -> String {
    match (left.is_empty(), right.is_empty()) {
        (true, _) => right.to_string(),
        (_, true) => left.to_string(),
        _ => format!("{left} {right}"),
    }
}

fn normalize_text(text: &str) -> String {
    let mut visible = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<|") {
        visible.push_str(&remaining[..start]);
        let Some(end) = remaining[start + 2..].find("|>") else {
            visible.push_str(&remaining[start..]);
            remaining = "";
            break;
        };
        remaining = &remaining[start + 2 + end + 2..];
    }
    visible.push_str(remaining);
    visible.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn ms_to_samples(ms: u64) -> u64 {
    ms * WHISPER_SAMPLE_RATE as u64 / 1000
}

fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / WHISPER_SAMPLE_RATE as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> StreamingOptions {
        StreamingOptions {
            decode_interval_ms: 1_000,
            overlap_ms: 250,
            max_window_ms: 3_000,
            agreement_updates: 2,
            trailing_guard_words: 1,
        }
    }

    fn segment(text: &str, start_ms: u64, end_ms: u64) -> DecodedSegment {
        DecodedSegment {
            text: text.to_string(),
            start_ms,
            end_ms,
        }
    }

    #[test]
    fn schedules_only_after_decode_interval() {
        let mut state = WhisperStreamingState::default();
        state.push_audio(&vec![0.0; 8_000], 16_000, 1).unwrap();
        assert!(state.decode_window(options(), false).is_none());
        state.push_audio(&vec![0.0; 8_000], 16_000, 1).unwrap();
        assert_eq!(
            state.decode_window(options(), false).unwrap().samples.len(),
            16_000
        );
    }

    #[test]
    fn resamples_and_downmixes_incremental_chunks() {
        let mut state = WhisperStreamingState::default();
        let stereo = vec![0.5; 8_820];
        state.push_audio(&stereo, 44_100, 2).unwrap();
        let window = state.decode_window(options(), true).unwrap();
        assert!((window.samples.len() as isize - 1_600).abs() <= 2);
        assert!(window
            .samples
            .iter()
            .all(|sample| (*sample - 0.5).abs() < 1e-5));
    }

    #[test]
    fn agreement_commits_complete_segments_and_corrections_replace() {
        let mut state = WhisperStreamingState::default();
        state.push_audio(&vec![0.0; 32_000], 16_000, 1).unwrap();

        let first = state
            .apply_hypothesis(
                vec![segment("turn on", 0, 800), segment("the", 800, 1_200)],
                0,
                options(),
                false,
            )
            .unwrap();
        assert_eq!(first.kind, "partial");
        assert_eq!(first.revision, 1);

        let correction = state
            .apply_hypothesis(
                vec![
                    segment("turn on", 0, 800),
                    segment("the lights", 800, 1_700),
                ],
                0,
                options(),
                false,
            )
            .unwrap();
        assert_eq!(correction.kind, "correction");
        assert_eq!(correction.replaces, Some(1));
        assert_eq!(correction.stable_text, "turn on");
        assert_eq!(correction.unstable_text, "the lights");
    }

    #[test]
    fn unchanged_hypothesis_does_not_emit_redundant_revision() {
        let mut state = WhisperStreamingState::default();
        state.push_audio(&vec![0.0; 16_000], 16_000, 1).unwrap();
        let hypothesis = vec![segment("hello world", 0, 900)];
        assert!(state
            .apply_hypothesis(hypothesis.clone(), 0, options(), false)
            .is_some());
        assert!(state
            .apply_hypothesis(hypothesis, 0, options(), false)
            .is_none());
    }

    #[test]
    fn rolling_audio_is_strictly_bounded() {
        let mut state = WhisperStreamingState::default();
        state.push_audio(&vec![0.0; 64_000], 16_000, 1).unwrap();
        state.apply_hypothesis(
            vec![
                segment("one", 0, 1_000),
                segment("two", 1_000, 2_000),
                segment("three", 2_000, 3_000),
                segment("four", 3_000, 4_000),
            ],
            0,
            options(),
            false,
        );
        assert!(state.retained_samples() <= 48_000);
    }

    #[test]
    fn decode_window_is_bounded_without_a_successful_hypothesis() {
        let mut state = WhisperStreamingState::default();
        state
            .push_audio(&vec![0.0; 5 * WHISPER_SAMPLE_RATE as usize], 16_000, 1)
            .unwrap();

        let window = state.decode_window(options(), true).unwrap();
        assert_eq!(window.samples.len(), 3 * WHISPER_SAMPLE_RATE as usize);
        assert_eq!(window.start_ms, 2_000);
        assert_eq!(state.retained_samples(), 3 * WHISPER_SAMPLE_RATE as usize);
    }

    #[test]
    fn commits_previous_hypothesis_before_evicting_its_audio() {
        let mut state = WhisperStreamingState::default();
        state
            .push_audio(&vec![0.0; 3 * WHISPER_SAMPLE_RATE as usize], 16_000, 1)
            .unwrap();
        state.apply_hypothesis(
            vec![
                segment("early words", 0, 1_500),
                segment("later words", 1_500, 3_000),
            ],
            0,
            options(),
            false,
        );
        state
            .push_audio(&vec![0.0; 2 * WHISPER_SAMPLE_RATE as usize], 16_000, 1)
            .unwrap();

        let window = state.decode_window(options(), true).unwrap();
        assert_eq!(join_segments(&state.committed), "early words later");
        assert_eq!(window.start_ms, 2_000);
        assert_eq!(window.samples.len(), 3 * WHISPER_SAMPLE_RATE as usize);
    }

    #[test]
    fn removes_timestamp_control_tokens_from_visible_text() {
        assert_eq!(
            normalize_text("before <|0.00|><|27.48|> after"),
            "before after"
        );
        assert_eq!(normalize_text("<|0.00|><|27.48|>"), "");
    }

    #[test]
    fn removes_words_before_committed_cutoff_inside_a_phrase() {
        let mut state = WhisperStreamingState {
            committed: vec![segment("old", 0, 1_000)],
            committed_through_ms: 1_000,
            ..Default::default()
        };

        let event = state
            .apply_hypothesis(vec![segment("old new", 0, 2_000)], 0, options(), false)
            .unwrap();
        assert_eq!(event.stable_text, "old");
        assert_eq!(event.unstable_text, "new");
        assert_eq!(event.text, "old new");
    }

    #[test]
    fn final_event_commits_remaining_text_once() {
        let mut state = WhisperStreamingState::default();
        state.push_audio(&vec![0.0; 16_000], 16_000, 1).unwrap();
        state.apply_hypothesis(vec![segment("hello", 0, 900)], 0, options(), false);
        let final_event = state
            .apply_hypothesis(vec![segment("hello world", 0, 1_000)], 0, options(), true)
            .unwrap();
        assert_eq!(final_event.kind, "final");
        assert_eq!(final_event.text, "hello world");
        assert_eq!(final_event.stable_text, "hello world");
        assert!(final_event.unstable_text.is_empty());
        assert!(state.push_audio(&[0.0], 16_000, 1).is_err());
    }

    #[test]
    fn final_window_deduplicates_committed_overlap() {
        let mut state = WhisperStreamingState::default();
        state.push_audio(&vec![0.0; 32_000], 16_000, 1).unwrap();
        let first = vec![segment("this is a test tail", 0, 1_500)];
        state
            .apply_hypothesis(first.clone(), 0, options(), false)
            .unwrap();
        state.apply_hypothesis(first, 0, options(), false);

        let final_event = state
            .apply_hypothesis(
                vec![segment("test of streaming", 0, 1_500)],
                450,
                options(),
                true,
            )
            .unwrap();
        assert_eq!(final_event.text, "this is a test of streaming");
    }
}
