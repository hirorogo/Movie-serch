"""外見特徴分析モジュール - 髪型・髪色・服装などの視覚的特徴を抽出"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class AppearanceAnalyzer:
    """OpenCLIP を使った外見特徴のベクトル化と比較"""

    def __init__(self, threshold: float = 0.60):
        """
        Args:
            threshold: 視覚一致と判定する最低コサイン類似度
        """
        self.threshold = threshold
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.reference_features: dict[str, np.ndarray] = {}

    def _ensure_model(self) -> None:
        """モデルの遅延ロード"""
        if self.model is None:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model.eval()

    def extract_features(self, image_path: str) -> np.ndarray:
        """画像から視覚的特徴ベクトルを抽出する。

        Args:
            image_path: 入力画像のパス

        Returns:
            正規化された特徴ベクトル (512次元)
        """
        import torch

        self._ensure_model()
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    def register_reference(self, person_id: str, image_paths: list[str]) -> None:
        """基準画像から人物の視覚的特徴ベクトルを登録する。

        Args:
            person_id: 人物ID
            image_paths: 基準画像ファイルパスのリスト
        """
        features_list = []
        for path in image_paths:
            feat = self.extract_features(path)
            features_list.append(feat)

        self.reference_features[person_id] = np.mean(features_list, axis=0)

    def register_references_from_dir(self, reference_dir: str) -> None:
        """ディレクトリ構造から全人物の基準画像を一括登録する。

        ディレクトリ構成:
            reference_dir/
            ├── person_a/
            │   └── *.jpg / *.png
            └── ...
        """
        ref_path = Path(reference_dir)
        if not ref_path.exists():
            raise FileNotFoundError(f"基準画像ディレクトリが見つかりません: {ref_path}")

        for person_dir in sorted(ref_path.iterdir()):
            if not person_dir.is_dir():
                continue

            image_files = (
                list(person_dir.glob("*.jpg"))
                + list(person_dir.glob("*.png"))
                + list(person_dir.glob("*.jpeg"))
            )
            if not image_files:
                continue

            self.register_reference(person_dir.name, [str(f) for f in image_files])

    def compare(self, image_path: str) -> dict[str, float]:
        """画像を全登録人物と照合し、類似度スコアを返す。

        Args:
            image_path: 照合対象の画像パス

        Returns:
            {人物ID: コサイン類似度} の辞書
        """
        if not self.reference_features:
            return {}

        features = self.extract_features(image_path)
        scores = {}

        for person_id, ref_features in self.reference_features.items():
            similarity = float(np.dot(features, ref_features) / (
                np.linalg.norm(features) * np.linalg.norm(ref_features)
            ))
            scores[person_id] = similarity

        return scores

    def compare_crops(self, crop_paths: list[str]) -> dict[str, dict]:
        """複数の人物切り出し画像を照合し、最良スコアを返す。

        Args:
            crop_paths: 人物切り出し画像パスのリスト

        Returns:
            {人物ID: {"max_score": float, "avg_score": float}} の辞書
        """
        all_scores: dict[str, list[float]] = {
            pid: [] for pid in self.reference_features
        }

        for path in crop_paths:
            scores = self.compare(path)
            for pid, score in scores.items():
                all_scores[pid].append(score)

        results = {}
        for pid, scores in all_scores.items():
            if scores:
                results[pid] = {
                    "max_score": float(np.max(scores)),
                    "avg_score": float(np.mean(scores)),
                }
            else:
                results[pid] = {"max_score": 0.0, "avg_score": 0.0}

        return results
