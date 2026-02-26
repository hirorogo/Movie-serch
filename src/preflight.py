"""事前チェック - FFmpeg や依存ライブラリの存在確認"""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """事前チェックに失敗した場合のエラー"""


def check_ffmpeg() -> None:
    """FFmpeg と FFprobe がインストールされているか確認する。"""
    for cmd in ("ffmpeg", "ffprobe"):
        path = shutil.which(cmd)
        if path is None:
            raise PreflightError(
                f"{cmd} が見つかりません。インストールしてください: "
                "https://ffmpeg.org/download.html"
            )
        logger.debug("%s found: %s", cmd, path)


def check_ffmpeg_version() -> str | None:
    """FFmpeg のバージョン文字列を返す。取得できなければ None。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            logger.debug("FFmpeg version: %s", first_line)
            return first_line
    except FileNotFoundError:
        pass
    return None


def check_gpu() -> bool:
    """CUDA GPU が利用可能かチェックする。"""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            device_name = torch.cuda.get_device_name(0)
            logger.info("GPU 検出: %s", device_name)
        else:
            logger.info("GPU 未検出 - CPU モードで動作します")
        return available
    except ImportError:
        logger.debug("PyTorch 未インストール - GPU チェックをスキップ")
        return False


def run_preflight(check_gpu_available: bool = False) -> None:
    """全ての事前チェックを実行する。

    Args:
        check_gpu_available: GPU 可用性もチェックするか

    Raises:
        PreflightError: 必須の依存関係が見つからない場合
    """
    logger.info("事前チェック実行中...")
    check_ffmpeg()
    version = check_ffmpeg_version()
    if version:
        logger.info("FFmpeg: %s", version)

    if check_gpu_available:
        check_gpu()

    logger.info("事前チェック完了")
