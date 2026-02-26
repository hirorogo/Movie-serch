"""事前チェックモジュールのテスト"""

from unittest.mock import patch, MagicMock

import pytest

from src.preflight import check_ffmpeg, check_ffmpeg_version, check_gpu, PreflightError


class TestCheckFFmpeg:
    @patch("src.preflight.shutil.which")
    def test_ffmpeg_found(self, mock_which):
        mock_which.return_value = "/usr/bin/ffmpeg"
        # 例外が出なければOK
        check_ffmpeg()

    @patch("src.preflight.shutil.which")
    def test_ffmpeg_not_found(self, mock_which):
        mock_which.side_effect = lambda cmd: None if cmd == "ffmpeg" else "/usr/bin/ffprobe"
        with pytest.raises(PreflightError, match="ffmpeg が見つかりません"):
            check_ffmpeg()

    @patch("src.preflight.shutil.which")
    def test_ffprobe_not_found(self, mock_which):
        mock_which.side_effect = lambda cmd: "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None
        with pytest.raises(PreflightError, match="ffprobe が見つかりません"):
            check_ffmpeg()


class TestCheckFFmpegVersion:
    @patch("src.preflight.subprocess.run")
    def test_version_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ffmpeg version 6.1 Copyright\nmore info\n"
        )
        version = check_ffmpeg_version()
        assert "ffmpeg version 6.1" in version

    @patch("src.preflight.subprocess.run")
    def test_version_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert check_ffmpeg_version() is None

    @patch("src.preflight.subprocess.run", side_effect=FileNotFoundError)
    def test_version_not_installed(self, mock_run):
        assert check_ffmpeg_version() is None


class TestCheckGpu:
    @patch("src.preflight.torch", create=True)
    def test_gpu_available(self, mock_torch_module):
        # torch をモック化
        import src.preflight as pf
        with patch.dict("sys.modules", {"torch": mock_torch_module}):
            mock_torch_module.cuda.is_available.return_value = True
            mock_torch_module.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
            result = check_gpu()
            assert result is True

    def test_gpu_no_torch(self):
        # torch が未インストールのケース
        import sys
        with patch.dict(sys.modules, {"torch": None}):
            result = check_gpu()
            assert result is False
