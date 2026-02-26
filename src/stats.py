"""統計・分析モジュール - 解析結果の集計と可視化用データ生成"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """解析結果JSONを読み込み、統計情報を生成するクラス。"""

    def __init__(self, results_path: str | None = None, results_data: list[dict] | None = None):
        """
        Args:
            results_path: results.json のパス
            results_data: 直接結果データを渡す場合
        """
        if results_data is not None:
            self.results = results_data
        elif results_path:
            self.results = self._load_results(results_path)
        else:
            self.results = []

    @staticmethod
    def _load_results(path: str) -> list[dict]:
        """JSONファイルから結果を読み込む"""
        p = Path(path)
        if not p.exists():
            logger.warning("結果ファイルが見つかりません: %s", path)
            return []
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", []) if isinstance(data, dict) else data

    def get_overview(self) -> dict:
        """全体概要の統計情報を返す"""
        total = len(self.results)
        if total == 0:
            return {
                "total_videos": 0,
                "total_performers_detected": 0,
                "avg_detected_per_video": 0,
                "videos_with_errors": 0,
                "performer_stats": {},
            }

        detected_counts = []
        performer_detections: dict[str, int] = {}
        error_count = 0

        for r in self.results:
            detected_counts.append(r.get("detected_count", 0))
            if r.get("errors"):
                error_count += 1
            for pid, pdata in r.get("performers", {}).items():
                if pdata.get("detected"):
                    performer_detections[pid] = performer_detections.get(pid, 0) + 1

        return {
            "total_videos": total,
            "total_performers_detected": sum(detected_counts),
            "avg_detected_per_video": round(np.mean(detected_counts), 2) if detected_counts else 0,
            "videos_with_errors": error_count,
            "performer_stats": {
                pid: {"detected_count": count, "detection_rate": round(count / total, 3)}
                for pid, count in performer_detections.items()
            },
        }

    def get_performer_analysis(self) -> dict:
        """出演者ごとの詳細分析"""
        performers: dict[str, dict] = {}

        for r in self.results:
            for pid, pdata in r.get("performers", {}).items():
                if pid not in performers:
                    performers[pid] = {
                        "name": pdata.get("name", pid),
                        "voice_scores": [],
                        "visual_scores": [],
                        "combined_scores": [],
                        "detected_count": 0,
                        "total_appearances": 0,
                    }

                performers[pid]["total_appearances"] += 1
                vs = pdata.get("voice_score", 0)
                vis = pdata.get("visual_score", 0)
                cs = pdata.get("combined_score", 0)

                if isinstance(vs, (int, float)):
                    performers[pid]["voice_scores"].append(float(vs))
                if isinstance(vis, (int, float)):
                    performers[pid]["visual_scores"].append(float(vis))
                if isinstance(cs, (int, float)):
                    performers[pid]["combined_scores"].append(float(cs))
                if pdata.get("detected"):
                    performers[pid]["detected_count"] += 1

        result = {}
        for pid, data in performers.items():
            result[pid] = {
                "name": data["name"],
                "detected_count": data["detected_count"],
                "total_appearances": data["total_appearances"],
                "detection_rate": round(
                    data["detected_count"] / data["total_appearances"], 3
                ) if data["total_appearances"] > 0 else 0,
            }
            for score_type in ["voice_scores", "visual_scores", "combined_scores"]:
                scores = data[score_type]
                key = score_type.replace("_scores", "")
                if scores:
                    arr = np.array(scores)
                    result[pid][key] = {
                        "mean": round(float(np.mean(arr)), 4),
                        "std": round(float(np.std(arr)), 4),
                        "min": round(float(np.min(arr)), 4),
                        "max": round(float(np.max(arr)), 4),
                        "median": round(float(np.median(arr)), 4),
                    }
                else:
                    result[pid][key] = {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

        return result

    def get_video_details(self, video_name: str | None = None) -> list[dict]:
        """動画ごとの詳細結果。video_name指定時はその動画のみ返す。"""
        if video_name:
            return [r for r in self.results if r.get("video") == video_name]
        return self.results

    def get_score_trends(self) -> dict:
        """スコアの推移データ（動画解析順）"""
        trends: dict[str, list[dict]] = {}

        for i, r in enumerate(self.results):
            for pid, pdata in r.get("performers", {}).items():
                if pid not in trends:
                    trends[pid] = []
                trends[pid].append({
                    "index": i,
                    "video": r.get("video", ""),
                    "voice_score": pdata.get("voice_score", 0),
                    "visual_score": pdata.get("visual_score", 0),
                    "combined_score": pdata.get("combined_score", 0),
                    "detected": pdata.get("detected", False),
                })

        return trends

    def get_detection_matrix(self) -> dict:
        """出演マトリクスデータ（動画×出演者）"""
        videos = []
        performer_ids = set()

        for r in self.results:
            for pid in r.get("performers", {}):
                performer_ids.add(pid)

        performer_ids = sorted(performer_ids)

        for r in self.results:
            row = {"video": r.get("video", "")}
            for pid in performer_ids:
                pdata = r.get("performers", {}).get(pid, {})
                row[pid] = {
                    "detected": pdata.get("detected", False),
                    "combined_score": pdata.get("combined_score", 0),
                    "name": pdata.get("name", pid),
                }
            videos.append(row)

        return {"performer_ids": performer_ids, "videos": videos}

    def get_confidence_analysis(self) -> dict:
        """信頼度分析 - スコアの分布と確信度を分析"""
        high_confidence = []  # combined >= 0.85
        medium_confidence = []  # 0.65 <= combined < 0.85
        low_confidence = []  # combined < 0.65 but detected
        borderline = []  # 0.60 <= combined < 0.80 (閾値付近)

        for r in self.results:
            for pid, pdata in r.get("performers", {}).items():
                cs = pdata.get("combined_score", 0)
                entry = {
                    "video": r.get("video", ""),
                    "performer": pid,
                    "name": pdata.get("name", pid),
                    "combined_score": cs,
                    "voice_score": pdata.get("voice_score", 0),
                    "visual_score": pdata.get("visual_score", 0),
                    "detected": pdata.get("detected", False),
                }

                if cs >= 0.85:
                    high_confidence.append(entry)
                elif cs >= 0.65:
                    medium_confidence.append(entry)
                elif pdata.get("detected"):
                    low_confidence.append(entry)

                if 0.60 <= cs < 0.80:
                    borderline.append(entry)

        return {
            "high_confidence": {"count": len(high_confidence), "items": high_confidence},
            "medium_confidence": {"count": len(medium_confidence), "items": medium_confidence},
            "low_confidence": {"count": len(low_confidence), "items": low_confidence},
            "borderline": {"count": len(borderline), "items": borderline},
            "summary": {
                "total_detections": len(high_confidence) + len(medium_confidence) + len(low_confidence),
                "high_rate": round(
                    len(high_confidence) / max(1, len(high_confidence) + len(medium_confidence) + len(low_confidence)),
                    3,
                ),
                "borderline_count": len(borderline),
            },
        }
