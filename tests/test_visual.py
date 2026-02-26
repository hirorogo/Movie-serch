"""Visual モジュールのテスト"""

from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from src.visual.body_analyzer import BodyAnalyzer, PersonDetection
from src.visual.appearance import AppearanceAnalyzer


# =========================================================================
# BodyAnalyzer
# =========================================================================


class TestBodyAnalyzer:
    def test_init_defaults(self):
        analyzer = BodyAnalyzer()
        assert analyzer.confidence_threshold == 0.5
        assert analyzer.model is None

    def test_init_custom_threshold(self):
        analyzer = BodyAnalyzer(confidence_threshold=0.8)
        assert analyzer.confidence_threshold == 0.8

    @patch("src.visual.body_analyzer.Image")
    def test_detect_persons(self, mock_image_module):
        analyzer = BodyAnalyzer(confidence_threshold=0.5)

        # モック YOLO モデル
        mock_model = MagicMock()
        analyzer.model = mock_model

        # モック画像
        mock_img = MagicMock()
        mock_img.size = (1920, 1080)
        mock_image_module.open.return_value = mock_img

        # モック検出結果
        mock_box = MagicMock()
        mock_box.conf = [np.array(0.9)]
        mock_box.xyxy = [np.array([100, 200, 500, 900])]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        detections = analyzer.detect_persons("test.jpg")

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, PersonDetection)
        assert det.confidence == 0.9
        assert det.bbox == (100, 200, 500, 900)

    @patch("src.visual.body_analyzer.Image")
    def test_detect_persons_below_threshold(self, mock_image_module):
        analyzer = BodyAnalyzer(confidence_threshold=0.8)
        mock_model = MagicMock()
        analyzer.model = mock_model

        mock_img = MagicMock()
        mock_img.size = (1920, 1080)
        mock_image_module.open.return_value = mock_img

        # 閾値以下の検出
        mock_box = MagicMock()
        mock_box.conf = [np.array(0.3)]
        mock_box.xyxy = [np.array([100, 200, 500, 900])]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        detections = analyzer.detect_persons("test.jpg")
        assert len(detections) == 0

    @patch("src.visual.body_analyzer.Image")
    def test_count_persons(self, mock_image_module):
        analyzer = BodyAnalyzer()
        mock_model = MagicMock()
        analyzer.model = mock_model

        mock_img = MagicMock()
        mock_img.size = (1920, 1080)
        mock_image_module.open.return_value = mock_img

        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        assert analyzer.count_persons("test.jpg") == 0


class TestPersonDetection:
    def test_dataclass_fields(self):
        det = PersonDetection(
            bbox=(10, 20, 110, 220),
            confidence=0.95,
            body_ratio=2.0,
            relative_height=0.5,
            area_ratio=0.1,
        )
        assert det.bbox == (10, 20, 110, 220)
        assert det.confidence == 0.95
        assert det.body_ratio == 2.0


# =========================================================================
# AppearanceAnalyzer
# =========================================================================


class TestAppearanceAnalyzer:
    def test_init_defaults(self):
        analyzer = AppearanceAnalyzer()
        assert analyzer.threshold == 0.60
        assert analyzer.model is None
        assert analyzer.reference_features == {}

    def test_compare_no_references(self):
        analyzer = AppearanceAnalyzer()
        result = analyzer.compare("test.jpg")
        assert result == {}

    def test_compare_crops_empty(self):
        analyzer = AppearanceAnalyzer()
        analyzer.reference_features = {"person_a": np.array([1.0, 0.0])}
        result = analyzer.compare_crops([])
        assert result["person_a"]["max_score"] == 0.0

    def test_compare_with_references(self):
        analyzer = AppearanceAnalyzer()
        analyzer.reference_features = {
            "person_a": np.array([1.0, 0.0, 0.0]),
            "person_b": np.array([0.0, 1.0, 0.0]),
        }

        # extract_features をモック
        with patch.object(analyzer, "extract_features") as mock_extract:
            mock_extract.return_value = np.array([0.9, 0.1, 0.0])
            scores = analyzer.compare("test.jpg")

        assert "person_a" in scores
        assert "person_b" in scores
        assert scores["person_a"] > scores["person_b"]
