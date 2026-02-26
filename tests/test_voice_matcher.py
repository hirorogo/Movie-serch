"""声紋照合モジュールのテスト"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.audio.voice_matcher import VoiceMatcher


@pytest.fixture
def mock_encoder():
    """VoiceEncoder のモック"""
    with patch("src.audio.voice_matcher.VoiceEncoder") as MockEncoder:
        encoder_instance = MagicMock()
        MockEncoder.return_value = encoder_instance
        yield encoder_instance


@pytest.fixture
def matcher(mock_encoder):
    """テスト用 VoiceMatcher"""
    return VoiceMatcher(threshold=0.75)


class TestVoiceMatcher:
    def test_init(self, matcher):
        assert matcher.threshold == 0.75
        assert matcher.reference_embeddings == {}

    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_register_speaker(self, mock_preprocess, matcher, mock_encoder):
        mock_preprocess.return_value = np.zeros(16000)
        mock_encoder.embed_utterance.return_value = np.array([1.0, 0.0, 0.0])

        matcher.register_speaker("person_a", ["sample.wav"])

        assert "person_a" in matcher.reference_embeddings
        np.testing.assert_array_equal(
            matcher.reference_embeddings["person_a"],
            np.array([1.0, 0.0, 0.0])
        )

    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_compare_no_speakers_registered(self, mock_preprocess, matcher):
        with pytest.raises(RuntimeError, match="基準話者が登録されていません"):
            matcher.compare("test.wav")

    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_compare(self, mock_preprocess, matcher, mock_encoder):
        # 基準話者を手動で登録
        matcher.reference_embeddings = {
            "person_a": np.array([1.0, 0.0, 0.0]),
            "person_b": np.array([0.0, 1.0, 0.0]),
        }

        mock_preprocess.return_value = np.zeros(16000)
        # テスト音声は person_a に近いベクトル
        mock_encoder.embed_utterance.return_value = np.array([0.9, 0.1, 0.0])

        scores = matcher.compare("test.wav")

        assert "person_a" in scores
        assert "person_b" in scores
        assert scores["person_a"] > scores["person_b"]

    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_identify_above_threshold(self, mock_preprocess, matcher, mock_encoder):
        matcher.reference_embeddings = {
            "person_a": np.array([1.0, 0.0, 0.0]),
        }

        mock_preprocess.return_value = np.zeros(16000)
        mock_encoder.embed_utterance.return_value = np.array([0.95, 0.05, 0.0])

        speaker, score = matcher.identify("test.wav")

        assert speaker == "person_a"
        assert score > 0.75

    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_identify_below_threshold(self, mock_preprocess, matcher, mock_encoder):
        matcher.reference_embeddings = {
            "person_a": np.array([1.0, 0.0, 0.0]),
        }

        mock_preprocess.return_value = np.zeros(16000)
        # 全く異なるベクトル
        mock_encoder.embed_utterance.return_value = np.array([0.0, 0.0, 1.0])

        speaker, score = matcher.identify("test.wav")

        assert speaker is None

    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_compare_empty_audio(self, mock_preprocess, matcher, mock_encoder):
        matcher.reference_embeddings = {
            "person_a": np.array([1.0, 0.0, 0.0]),
        }

        mock_preprocess.return_value = np.array([])

        scores = matcher.compare("empty.wav")

        assert scores["person_a"] == 0.0


class TestCompareSegments:
    @patch("src.audio.voice_matcher.preprocess_wav")
    def test_compare_segments(self, mock_preprocess, matcher, mock_encoder):
        matcher.reference_embeddings = {
            "person_a": np.array([1.0, 0.0, 0.0]),
            "person_b": np.array([0.0, 1.0, 0.0]),
        }

        mock_preprocess.return_value = np.zeros(16000)
        mock_encoder.embed_utterance.return_value = np.array([0.9, 0.1, 0.0])

        segments = [
            {"start": 0.0, "end": 5.0, "audio_path": "seg1.wav"},
            {"start": 5.0, "end": 10.0, "audio_path": "seg2.wav"},
        ]

        results = matcher.compare_segments(segments)

        assert "person_a" in results
        assert "person_b" in results
        assert results["person_a"]["total_segments"] == 2
        assert results["person_b"]["total_segments"] == 2
