"""閾値自動最適化モジュール - 解析結果から最適な閾値を推定"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """閾値最適化の結果"""
    optimal_voice: float
    optimal_visual: float
    voice_scores: list[float]
    visual_scores: list[float]
    score_distribution: dict


class ThresholdOptimizer:
    """解析結果に基づいて最適な閾値を自動推定するクラス。

    スコア分布のギャップ検出と統計的手法を組み合わせて、
    正例（出演あり）と負例（出演なし）を最もよく分離する閾値を推定する。
    """

    def __init__(self, results: list[dict]):
        """
        Args:
            results: VideoAnalysisResult.to_dict() 形式の結果リスト
        """
        self.results = results
        self._voice_scores: list[float] = []
        self._visual_scores: list[float] = []
        self._extract_scores()

    def _extract_scores(self) -> None:
        """全結果からスコアを抽出"""
        for result in self.results:
            performers = result.get("performers", {})
            for pid, pdata in performers.items():
                vs = pdata.get("voice_score", 0.0)
                vis = pdata.get("visual_score", 0.0)
                if isinstance(vs, (int, float)):
                    self._voice_scores.append(float(vs))
                if isinstance(vis, (int, float)) and vis > 0:
                    self._visual_scores.append(float(vis))

    def optimize(self) -> ThresholdResult:
        """スコア分布から最適な閾値を推定する。

        手法: bimodal分布のギャップ（谷）を検出。
        正例クラスタと負例クラスタの境界を閾値とする。

        Returns:
            ThresholdResult: 推定された最適閾値と分布情報
        """
        optimal_voice = self._find_optimal_threshold(self._voice_scores, default=0.75)
        optimal_visual = self._find_optimal_threshold(self._visual_scores, default=0.60)

        distribution = self._compute_distribution()

        logger.info("最適閾値推定: voice=%.3f, visual=%.3f", optimal_voice, optimal_visual)

        return ThresholdResult(
            optimal_voice=optimal_voice,
            optimal_visual=optimal_visual,
            voice_scores=self._voice_scores,
            visual_scores=self._visual_scores,
            score_distribution=distribution,
        )

    def _find_optimal_threshold(
        self, scores: list[float], default: float = 0.75
    ) -> float:
        """スコア分布のギャップから最適閾値を推定。

        ヒストグラムの谷（最小頻度）を閾値候補とし、
        分布が bimodal でない場合は Otsu 法を適用する。
        """
        if len(scores) < 4:
            return default

        arr = np.array(scores)
        bins = min(50, max(10, len(scores) // 3))
        hist, bin_edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))

        # ヒストグラムの谷（最小頻度ポイント）をスキャン
        # 0.3〜0.9 の範囲で探索
        search_start = int(bins * 0.3)
        search_end = int(bins * 0.9)
        if search_start >= search_end:
            return default

        search_hist = hist[search_start:search_end]
        if len(search_hist) == 0:
            return default

        # 最小頻度ポイントを閾値とする
        min_idx = int(np.argmin(search_hist)) + search_start
        threshold = float((bin_edges[min_idx] + bin_edges[min_idx + 1]) / 2)

        # 妥当性チェック: 閾値の上下にスコアが存在するか
        below = np.sum(arr < threshold)
        above = np.sum(arr >= threshold)
        if below == 0 or above == 0:
            # 分離できない場合、Otsu法（分散最大化）を使用
            threshold = self._otsu_threshold(arr, bins)

        return round(threshold, 3)

    @staticmethod
    def _otsu_threshold(scores: np.ndarray, bins: int = 50) -> float:
        """Otsu法によるクラス間分散最大化で閾値を推定"""
        hist, bin_edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total = hist.sum()

        if total == 0:
            return 0.75

        best_threshold = 0.75
        best_variance = 0.0

        cumsum = 0
        cum_mean = 0.0
        global_mean = float(np.sum(hist * bin_centers))

        for i in range(len(hist)):
            cumsum += hist[i]
            if cumsum == 0 or cumsum == total:
                continue

            cum_mean += hist[i] * bin_centers[i]
            w0 = cumsum / total
            w1 = 1.0 - w0
            mu0 = cum_mean / cumsum
            mu1 = (global_mean - cum_mean) / (total - cumsum)

            variance = w0 * w1 * (mu0 - mu1) ** 2
            if variance > best_variance:
                best_variance = variance
                best_threshold = float(bin_centers[i])

        return round(best_threshold, 3)

    def _compute_distribution(self) -> dict:
        """スコア分布の統計情報を計算"""
        result = {"voice": {}, "visual": {}}

        for key, scores in [("voice", self._voice_scores), ("visual", self._visual_scores)]:
            if not scores:
                result[key] = {
                    "count": 0, "mean": 0, "std": 0,
                    "min": 0, "max": 0, "median": 0,
                    "q25": 0, "q75": 0,
                    "histogram": {"bins": [], "counts": []},
                }
                continue

            arr = np.array(scores)
            hist, bin_edges = np.histogram(arr, bins=20, range=(0.0, 1.0))
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

            result[key] = {
                "count": len(scores),
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "q25": round(float(np.percentile(arr, 25)), 4),
                "q75": round(float(np.percentile(arr, 75)), 4),
                "histogram": {
                    "bins": [round(b, 3) for b in bin_centers],
                    "counts": hist.tolist(),
                },
            }

        return result

    def get_recommendation(self) -> dict:
        """現在の閾値設定に対する改善推奨を生成"""
        result = self.optimize()
        rec = {
            "current_voice": 0.75,
            "current_visual": 0.60,
            "recommended_voice": result.optimal_voice,
            "recommended_visual": result.optimal_visual,
            "voice_change": round(result.optimal_voice - 0.75, 3),
            "visual_change": round(result.optimal_visual - 0.60, 3),
            "data_points": len(self._voice_scores),
            "distribution": result.score_distribution,
        }

        if len(self._voice_scores) < 10:
            rec["confidence"] = "low"
            rec["message"] = "データが少ないため推定精度が低い可能性があります（10件以上推奨）"
        elif len(self._voice_scores) < 30:
            rec["confidence"] = "medium"
            rec["message"] = "中程度の信頼度です。より多くのデータで精度が向上します"
        else:
            rec["confidence"] = "high"
            rec["message"] = "十分なデータに基づく高精度な推定です"

        return rec
