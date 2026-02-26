"""閾値最適化モジュールのテスト"""

import numpy as np
import pytest

from src.optimizer import ThresholdOptimizer, ThresholdResult


def _make_results(voice_scores, visual_scores=None):
    """テスト用の結果データを生成"""
    results = []
    for i, vs in enumerate(voice_scores):
        performers = {
            "person_a": {
                "voice_score": vs,
                "visual_score": visual_scores[i] if visual_scores else 0.0,
                "detected": vs >= 0.75,
            }
        }
        results.append({"video": f"video_{i}.mp4", "performers": performers})
    return results


class TestThresholdOptimizer:
    def test_empty_results(self):
        optimizer = ThresholdOptimizer([])
        result = optimizer.optimize()
        assert isinstance(result, ThresholdResult)
        assert result.optimal_voice == 0.75  # default

    def test_few_results_returns_default(self):
        results = _make_results([0.8, 0.9])
        optimizer = ThresholdOptimizer(results)
        result = optimizer.optimize()
        assert result.optimal_voice == 0.75

    def test_bimodal_distribution(self):
        low = [0.2, 0.25, 0.3, 0.15, 0.22, 0.28, 0.18, 0.35]
        high = [0.85, 0.9, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89]
        scores = low + high
        results = _make_results(scores)
        optimizer = ThresholdOptimizer(results)
        result = optimizer.optimize()
        assert 0.3 < result.optimal_voice < 0.85

    def test_score_distribution_has_stats(self):
        scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = _make_results(scores)
        optimizer = ThresholdOptimizer(results)
        result = optimizer.optimize()
        dist = result.score_distribution
        assert "voice" in dist
        assert dist["voice"]["count"] == 7
        assert dist["voice"]["mean"] > 0
        assert "histogram" in dist["voice"]

    def test_get_recommendation(self):
        scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = _make_results(scores)
        optimizer = ThresholdOptimizer(results)
        rec = optimizer.get_recommendation()
        assert "recommended_voice" in rec
        assert "recommended_visual" in rec
        assert "confidence" in rec
        assert rec["confidence"] == "low"  # < 10 data points
        assert "data_points" in rec

    def test_recommendation_confidence_medium(self):
        scores = list(np.random.uniform(0.2, 0.9, 15))
        results = _make_results(scores)
        optimizer = ThresholdOptimizer(results)
        rec = optimizer.get_recommendation()
        assert rec["confidence"] == "medium"

    def test_recommendation_confidence_high(self):
        scores = list(np.random.uniform(0.2, 0.9, 35))
        results = _make_results(scores)
        optimizer = ThresholdOptimizer(results)
        rec = optimizer.get_recommendation()
        assert rec["confidence"] == "high"


class TestOtsuThreshold:
    def test_otsu_basic(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        threshold = ThresholdOptimizer._otsu_threshold(scores)
        assert 0.3 < threshold < 0.8

    def test_otsu_uniform(self):
        scores = np.linspace(0, 1, 50)
        threshold = ThresholdOptimizer._otsu_threshold(scores)
        assert 0.0 <= threshold <= 1.0
