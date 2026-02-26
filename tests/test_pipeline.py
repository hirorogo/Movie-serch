"""パイプラインのテスト"""

import pytest

from src.pipeline import PerformerResult, VideoAnalysisResult, _format_time


class TestFormatTime:
    def test_zero(self):
        assert _format_time(0) == "0:00"

    def test_seconds_only(self):
        assert _format_time(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_time(125) == "2:05"

    def test_large_value(self):
        assert _format_time(3661) == "61:01"


class TestPerformerResult:
    def test_defaults(self):
        result = PerformerResult(
            person_id="person_a",
            name="出演者A",
            detected=True,
        )
        assert result.voice_score == 0.0
        assert result.visual_score == 0.0
        assert result.combined_score == 0.0
        assert result.speaking_time == 0.0


class TestVideoAnalysisResult:
    def test_to_dict_detected(self):
        result = VideoAnalysisResult(
            video_path="/tmp/test.mp4",
            video_name="test.mp4",
            duration=120.0,
            performers=[
                PerformerResult("person_a", "出演者A", True,
                                voice_score=0.9, combined_score=0.9),
                PerformerResult("person_b", "出演者B", False,
                                voice_score=0.3, combined_score=0.3),
            ],
            detected_count=1,
        )

        d = result.to_dict()

        assert d["video"] == "test.mp4"
        assert d["duration"] == "2:00"
        assert d["performers"]["person_a"]["detected"] is True
        assert d["performers"]["person_b"]["detected"] is False
        assert d["detected_count"] == 1
        assert "出演者A" in d["summary"]
        assert "1名" in d["summary"]

    def test_to_dict_no_performers(self):
        result = VideoAnalysisResult(
            video_path="/tmp/test.mp4",
            video_name="test.mp4",
            duration=60.0,
            performers=[
                PerformerResult("person_a", "出演者A", False),
            ],
            detected_count=0,
        )

        d = result.to_dict()
        assert d["summary"] == "出演者なし"

    def test_to_dict_two_performers(self):
        result = VideoAnalysisResult(
            video_path="/tmp/test.mp4",
            video_name="test.mp4",
            duration=300.0,
            performers=[
                PerformerResult("person_a", "出演者A", True,
                                voice_score=0.85, combined_score=0.85),
                PerformerResult("person_b", "出演者B", True,
                                voice_score=0.80, combined_score=0.80),
                PerformerResult("person_c", "出演者C", False,
                                voice_score=0.10, combined_score=0.10),
            ],
            detected_count=2,
        )

        d = result.to_dict()
        assert d["detected_count"] == 2
        assert "2名" in d["summary"]
        assert "出演者A" in d["summary"]
        assert "出演者B" in d["summary"]

    def test_errors_included(self):
        result = VideoAnalysisResult(
            video_path="/tmp/test.mp4",
            video_name="test.mp4",
            duration=0.0,
            errors=["音声抽出エラー: codec not found"],
        )

        d = result.to_dict()
        assert len(d["errors"]) == 1
