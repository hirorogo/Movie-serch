"""話者ダイアライゼーション - 音声内の話者交代を検出し時間区間で分離"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """話者セグメント"""
    start: float      # 開始時刻（秒）
    end: float         # 終了時刻（秒）
    speaker_label: str # 話者ラベル（"speaker_0", "speaker_1", ...）

    @property
    def duration(self) -> float:
        return self.end - self.start


class Diarizer:
    """話者ダイアライゼーションを行うクラス。

    pyannote-audio が利用可能な場合はそれを使い、
    利用不可の場合は resemblyzer ベースの簡易実装にフォールバックする。
    """

    def __init__(self, max_speakers: int = 3, min_segment_duration: float = 1.0,
                 use_pyannote: bool = True, hf_token: str | None = None):
        """
        Args:
            max_speakers: 最大話者数
            min_segment_duration: 最短セグメント長（秒）
            use_pyannote: pyannote-audio を使用するか
            hf_token: HuggingFace トークン（pyannote使用時に必要）
        """
        self.max_speakers = max_speakers
        self.min_segment_duration = min_segment_duration
        self.pipeline = None

        if use_pyannote:
            try:
                self._init_pyannote(hf_token)
            except Exception as e:
                logger.warning("pyannote-audio の初期化に失敗。resemblyzer にフォールバックします: %s", e)
                self.pipeline = None

    def _init_pyannote(self, hf_token: str | None) -> None:
        """pyannote-audio パイプラインを初期化"""
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    def diarize(self, audio_path: str) -> list[SpeakerSegment]:
        """音声ファイルの話者ダイアライゼーションを実行。

        Args:
            audio_path: WAVファイルのパス

        Returns:
            SpeakerSegment のリスト（時系列順）
        """
        if self.pipeline is not None:
            logger.info("pyannote-audio でダイアライゼーション実行中...")
            return self._diarize_pyannote(audio_path)
        logger.info("resemblyzer でダイアライゼーション実行中（フォールバック）...")
        return self._diarize_resemblyzer(audio_path)

    def _diarize_pyannote(self, audio_path: str) -> list[SpeakerSegment]:
        """pyannote-audio による高精度ダイアライゼーション"""
        diarization = self.pipeline(
            audio_path,
            max_speakers=self.max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.duration >= self.min_segment_duration:
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker_label=speaker,
                ))

        return segments

    def _diarize_resemblyzer(self, audio_path: str) -> list[SpeakerSegment]:
        """resemblyzer ベースの簡易ダイアライゼーション（フォールバック）

        固定長ウィンドウで音声を分割し、声紋ベクトルのクラスタリングで
        話者を推定する簡易的な手法。
        """
        encoder = VoiceEncoder()
        wav = preprocess_wav(Path(audio_path))

        if len(wav) == 0:
            return []

        # 固定長ウィンドウで分割
        window_sec = 1.5
        step_sec = 0.75
        sr = 16000
        window_samples = int(window_sec * sr)
        step_samples = int(step_sec * sr)

        embeddings = []
        timestamps = []

        pos = 0
        while pos + window_samples <= len(wav):
            chunk = wav[pos:pos + window_samples]
            emb = encoder.embed_utterance(chunk)
            embeddings.append(emb)
            timestamps.append(pos / sr)
            pos += step_samples

        if not embeddings:
            return []

        embeddings_array = np.array(embeddings)

        # コサイン類似度に基づく簡易クラスタリング
        labels = self._cluster_embeddings(embeddings_array)

        # ラベルをセグメントに変換
        segments = []
        current_label = labels[0]
        segment_start = timestamps[0]

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                seg_end = timestamps[i]
                if seg_end - segment_start >= self.min_segment_duration:
                    segments.append(SpeakerSegment(
                        start=segment_start,
                        end=seg_end,
                        speaker_label=f"speaker_{current_label}",
                    ))
                current_label = labels[i]
                segment_start = timestamps[i]

        # 最後のセグメント
        final_end = (timestamps[-1] + window_sec)
        if final_end - segment_start >= self.min_segment_duration:
            segments.append(SpeakerSegment(
                start=segment_start,
                end=final_end,
                speaker_label=f"speaker_{current_label}",
            ))

        return segments

    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        """声紋ベクトルを簡易クラスタリングする（k-means 的手法）"""
        from scipy.cluster.hierarchy import fcluster, linkage

        if len(embeddings) <= 1:
            return [0] * len(embeddings)

        # 階層的クラスタリング
        linkage_matrix = linkage(embeddings, method="ward")
        labels = fcluster(linkage_matrix, t=self.max_speakers, criterion="maxclust")

        return [int(l) - 1 for l in labels]
