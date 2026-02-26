"""音声抽出モジュールのテスト"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.audio.extractor import extract_audio, extract_audio_segment, get_video_duration


class TestExtractAudio:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="動画ファイルが見つかりません"):
            extract_audio("/nonexistent/video.mp4")

    @patch("src.audio.extractor.subprocess.run")
    def test_extract_audio_success(self, mock_run, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        output_file = tmp_path / "output.wav"

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = extract_audio(str(video_file), str(output_file))

        assert result == output_file
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-vn" in cmd
        assert "-ar" in cmd
        assert "16000" in cmd

    @patch("src.audio.extractor.subprocess.run")
    def test_extract_audio_ffmpeg_error(self, mock_run, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_run.return_value = MagicMock(returncode=1, stderr="codec error")

        with pytest.raises(RuntimeError, match="FFmpeg エラー"):
            extract_audio(str(video_file))

    @patch("src.audio.extractor.subprocess.run")
    def test_custom_sample_rate(self, mock_run, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        extract_audio(str(video_file), sample_rate=44100)

        cmd = mock_run.call_args[0][0]
        assert "44100" in cmd


class TestExtractAudioSegment:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_audio_segment("/nonexistent/video.mp4", 0.0, 5.0)

    @patch("src.audio.extractor.subprocess.run")
    def test_segment_extraction(self, mock_run, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = extract_audio_segment(str(video_file), 10.0, 20.0)

        cmd = mock_run.call_args[0][0]
        assert "-ss" in cmd
        assert "10.0" in cmd
        assert "-t" in cmd
        assert "10.0" in cmd  # duration = 20 - 10


class TestGetVideoDuration:
    @patch("src.audio.extractor.subprocess.run")
    def test_get_duration(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="123.45\n", stderr="")

        duration = get_video_duration("test.mp4")
        assert duration == pytest.approx(123.45)

    @patch("src.audio.extractor.subprocess.run")
    def test_ffprobe_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="probe error")

        with pytest.raises(RuntimeError, match="FFprobe エラー"):
            get_video_duration("test.mp4")
