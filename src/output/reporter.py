"""結果出力モジュール - 分析結果を JSON / CSV 形式で出力"""

import csv
import json
import logging
from pathlib import Path

from src.pipeline import VideoAnalysisResult

logger = logging.getLogger(__name__)


def save_json(results: list[VideoAnalysisResult], output_path: str) -> Path:
    """分析結果を JSON ファイルに出力する。

    Args:
        results: 分析結果のリスト
        output_path: 出力ファイルパス

    Returns:
        出力ファイルパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_videos": len(results),
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("JSON 出力完了: %s", output_path)
    return output_path


def save_csv(results: list[VideoAnalysisResult], output_path: str) -> Path:
    """分析結果を CSV ファイルに出力する。

    出力形式:
        動画名, 出演者A判定, 出演者Aスコア, 出演者B判定, 出演者Bスコア, ...

    Args:
        results: 分析結果のリスト
        output_path: 出力ファイルパス

    Returns:
        出力ファイルパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return output_path

    # ヘッダー構築
    performer_ids = [p.person_id for p in results[0].performers]
    header = ["動画名", "長さ"]
    for pid in performer_ids:
        header.extend([f"{pid}_出演", f"{pid}_声紋スコア", f"{pid}_視覚スコア",
                       f"{pid}_統合スコア", f"{pid}_発話時間"])
    header.append("出演者数")
    header.append("サマリー")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            result_dict = result.to_dict()
            row = [result.video_name, result_dict["duration"]]

            for pid in performer_ids:
                p = result_dict["performers"].get(pid, {})
                row.extend([
                    "○" if p.get("detected", False) else "×",
                    p.get("voice_score", 0.0),
                    p.get("visual_score", 0.0),
                    p.get("combined_score", 0.0),
                    p.get("speaking_time", "0:00"),
                ])

            row.append(result.detected_count)
            row.append(result_dict["summary"])
            writer.writerow(row)

    logger.info("CSV 出力完了: %s", output_path)
    return output_path


def save_results(results: list[VideoAnalysisResult], output_dir: str,
                 fmt: str = "json") -> list[Path]:
    """分析結果をファイルに保存する。

    Args:
        results: 分析結果のリスト
        output_dir: 出力ディレクトリ
        fmt: 出力形式 ("json", "csv", "both")

    Returns:
        出力ファイルパスのリスト
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if fmt in ("json", "both"):
        path = save_json(results, str(output_dir / "results.json"))
        saved.append(path)

    if fmt in ("csv", "both"):
        path = save_csv(results, str(output_dir / "results.csv"))
        saved.append(path)

    return saved


def print_summary(results: list[VideoAnalysisResult]) -> None:
    """分析結果のサマリーをコンソールに出力する。"""
    print(f"\n{'='*60}")
    print(f" 解析結果サマリー （{len(results)} 動画）")
    print(f"{'='*60}\n")

    for result in results:
        rd = result.to_dict()
        print(f"  {result.video_name}  [{rd['duration']}]")

        for p in result.performers:
            mark = "○" if p.detected else "×"
            print(f"    {mark} {p.name}: "
                  f"声紋={p.voice_score:.2f}  "
                  f"視覚={p.visual_score:.2f}  "
                  f"統合={p.combined_score:.2f}")

        print(f"    → {rd['summary']}")
        if result.errors:
            for err in result.errors:
                print(f"    ! {err}")
        print()

    # 出演マトリクス
    if results and results[0].performers:
        print(f"{'='*60}")
        print(" 出演マトリクス")
        print(f"{'='*60}\n")

        performer_names = [p.name for p in results[0].performers]
        header = f"  {'動画名':<30}" + "".join(f"{n:>10}" for n in performer_names)
        print(header)
        print("  " + "-" * (30 + 10 * len(performer_names)))

        for result in results:
            name = result.video_name[:28]
            marks = "".join(
                f"{'○':>10}" if p.detected else f"{'×':>10}"
                for p in result.performers
            )
            print(f"  {name:<30}{marks}")

        print()
