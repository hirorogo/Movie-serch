"""動画ファイルから音声を抽出するモジュール"""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str | None = None,
                  sample_rate: int = 16000) -> Path:
    """動画ファイルから音声(WAV)を抽出する。

    Args:
        video_path: 入力動画ファイルのパス
        output_path: 出力WAVファイルのパス（省略時は一時ファイル）
        sample_rate: サンプリングレート（デフォルト16kHz）

    Returns:
        出力WAVファイルのパス
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                      # 映像なし
        "-acodec", "pcm_s16le",     # 16bit PCM
        "-ar", str(sample_rate),    # サンプリングレート
        "-ac", "1",                 # モノラル
        "-y",                       # 上書き許可
        str(output_path)
    ]

    logger.debug("FFmpeg コマンド: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

    logger.info("音声抽出完了: %s", output_path)
    return output_path


def extract_audio_segment(video_path: str, start_sec: float, end_sec: float,
                          output_path: str | None = None,
                          sample_rate: int = 16000) -> Path:
    """動画ファイルから指定区間の音声を抽出する。

    Args:
        video_path: 入力動画ファイルのパス
        start_sec: 開始時刻（秒）
        end_sec: 終了時刻（秒）
        output_path: 出力WAVファイルのパス
        sample_rate: サンプリングレート

    Returns:
        出力WAVファイルのパス
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-ss", str(start_sec),
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y",
        str(output_path)
    ]

    logger.debug("FFmpeg セグメント抽出: %.1f-%.1f秒", start_sec, end_sec)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg エラー: {result.stderr}")

    return output_path


def get_video_duration(video_path: str) -> float:
    """動画の長さ（秒）を取得する。"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe エラー: {result.stderr}")

    return float(result.stdout.strip())
