"""動画からフレーム画像を抽出するモジュール"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, output_dir: str,
                   interval_sec: float = 2.0) -> list[Path]:
    """動画から一定間隔でフレーム画像を抽出する。

    Args:
        video_path: 入力動画ファイルのパス
        output_dir: フレーム画像の出力ディレクトリ
        interval_sec: フレーム抽出間隔（秒）

    Returns:
        抽出されたフレーム画像パスのリスト
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video_path.stem}_frame_%06d.jpg"

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval_sec}",
        "-q:v", "2",    # JPEG品質（1=最高、31=最低）
        "-y",
        str(pattern)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

    frames = sorted(output_dir.glob(f"{video_path.stem}_frame_*.jpg"))
    return frames


def extract_frame_at(video_path: str, timestamp_sec: float,
                     output_path: str) -> Path:
    """指定時刻のフレーム画像を1枚抽出する。

    Args:
        video_path: 入力動画ファイルのパス
        timestamp_sec: 抽出する時刻（秒）
        output_path: 出力画像ファイルのパス

    Returns:
        出力画像ファイルのパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-ss", str(timestamp_sec),
        "-vframes", "1",
        "-q:v", "2",
        "-y",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

    return output_path


def get_frame_timestamps(video_path: str, interval_sec: float = 2.0) -> list[float]:
    """動画の長さに基づいてフレーム抽出タイムスタンプを計算する。"""
    from src.audio.extractor import get_video_duration

    duration = get_video_duration(video_path)
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += interval_sec

    return timestamps
