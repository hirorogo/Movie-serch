"""統計・分析モジュールのテスト"""

import pytest

from src.stats import ResultsAnalyzer


SAMPLE_RESULTS = [
    {
        "video": "video_1.mp4",
        "duration": "2:30",
        "performers": {
            "person_a": {
                "name": "A",
                "detected": True,
                "voice_score": 0.89,
                "visual_score": 0.62,
                "combined_score": 0.81,
                "speaking_time": "1:20",
                "matching_segments": 5,
            },
            "person_b": {
                "name": "B",
                "detected": False,
                "voice_score": 0.35,
                "visual_score": 0.21,
                "combined_score": 0.31,
                "speaking_time": "0:00",
                "matching_segments": 0,
            },
        },
        "detected_count": 1,
        "summary": "A",
        "errors": [],
    },
    {
        "video": "video_2.mp4",
        "duration": "3:00",
        "performers": {
            "person_a": {
                "name": "A",
                "detected": True,
                "voice_score": 0.92,
                "visual_score": 0.70,
                "combined_score": 0.85,
                "speaking_time": "2:00",
                "matching_segments": 8,
            },
            "person_b": {
                "name": "B",
                "detected": True,
                "voice_score": 0.80,
                "visual_score": 0.65,
                "combined_score": 0.76,
                "speaking_time": "1:30",
                "matching_segments": 4,
            },
        },
        "detected_count": 2,
        "summary": "A, B",
        "errors": [],
    },
]


class TestResultsAnalyzer:
    def test_empty_results(self):
        analyzer = ResultsAnalyzer(results_data=[])
        overview = analyzer.get_overview()
        assert overview["total_videos"] == 0

    def test_overview(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        overview = analyzer.get_overview()
        assert overview["total_videos"] == 2
        assert overview["total_performers_detected"] == 3
        assert overview["videos_with_errors"] == 0
        assert "person_a" in overview["performer_stats"]
        assert overview["performer_stats"]["person_a"]["detected_count"] == 2

    def test_performer_analysis(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        analysis = analyzer.get_performer_analysis()
        assert "person_a" in analysis
        assert "person_b" in analysis
        assert analysis["person_a"]["detection_rate"] == 1.0
        assert analysis["person_b"]["detection_rate"] == 0.5
        assert analysis["person_a"]["voice"]["mean"] > 0

    def test_video_details_all(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        details = analyzer.get_video_details()
        assert len(details) == 2

    def test_video_details_single(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        details = analyzer.get_video_details("video_1.mp4")
        assert len(details) == 1
        assert details[0]["video"] == "video_1.mp4"

    def test_score_trends(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        trends = analyzer.get_score_trends()
        assert "person_a" in trends
        assert len(trends["person_a"]) == 2
        assert trends["person_a"][0]["index"] == 0

    def test_detection_matrix(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        matrix = analyzer.get_detection_matrix()
        assert "performer_ids" in matrix
        assert "videos" in matrix
        assert len(matrix["videos"]) == 2
        assert matrix["videos"][0]["person_a"]["detected"] is True

    def test_confidence_analysis(self):
        analyzer = ResultsAnalyzer(results_data=SAMPLE_RESULTS)
        conf = analyzer.get_confidence_analysis()
        assert "high_confidence" in conf
        assert "medium_confidence" in conf
        assert "borderline" in conf
        assert "summary" in conf
        assert conf["summary"]["total_detections"] >= 0
