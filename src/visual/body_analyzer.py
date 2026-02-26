"""体型・シルエット分析モジュール - YOLO による人物検出と体型特徴抽出"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """検出された人物の情報"""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    body_ratio: float                  # 身体の縦横比
    relative_height: float             # フレーム内での相対的な高さ
    area_ratio: float                  # フレーム面積に対する占有率


class BodyAnalyzer:
    """YOLO を使った人物検出・体型分析"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: 人物検出の信頼度閾値
        """
        self.confidence_threshold = confidence_threshold
        self.model = None

    def _ensure_model(self) -> None:
        """モデルの遅延ロード"""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")

    def detect_persons(self, image_path: str) -> list[PersonDetection]:
        """画像内の人物を検出する。

        Args:
            image_path: 入力画像のパス

        Returns:
            PersonDetection のリスト
        """
        self._ensure_model()
        image = Image.open(image_path)
        img_width, img_height = image.size
        img_area = img_width * img_height

        results = self.model(image, classes=[0], verbose=False)  # class 0 = person
        detections = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                box_width = x2 - x1
                box_height = y2 - y1

                detections.append(PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    body_ratio=box_height / max(box_width, 1),
                    relative_height=box_height / img_height,
                    area_ratio=(box_width * box_height) / img_area,
                ))

        return detections

    def count_persons(self, image_path: str) -> int:
        """画像内の人数を数える。"""
        return len(self.detect_persons(image_path))

    def extract_person_crops(self, image_path: str,
                             output_dir: str) -> list[dict]:
        """画像内の人物を切り出して保存する。

        Returns:
            [{"path": str, "detection": PersonDetection}, ...] のリスト
        """
        self._ensure_model()
        detections = self.detect_persons(image_path)
        image = Image.open(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        crops = []
        stem = Path(image_path).stem
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            crop = image.crop((x1, y1, x2, y2))
            crop_path = output_dir / f"{stem}_person_{i}.jpg"
            crop.save(str(crop_path))
            crops.append({"path": str(crop_path), "detection": det})

        return crops

    def get_body_features(self, image_path: str) -> list[dict]:
        """各人物の体型特徴量を抽出する。

        Returns:
            [{"body_ratio": float, "relative_height": float, "area_ratio": float}, ...]
        """
        detections = self.detect_persons(image_path)
        return [
            {
                "body_ratio": d.body_ratio,
                "relative_height": d.relative_height,
                "area_ratio": d.area_ratio,
            }
            for d in detections
        ]
