"""声紋照合モジュール - 基準音声と動画音声を比較して出演者を判定"""

import logging
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

logger = logging.getLogger(__name__)


class VoiceMatcher:
    """声紋ベクトルによる話者照合を行うクラス。"""

    def __init__(self, threshold: float = 0.75, cache=None):
        """
        Args:
            threshold: 声紋一致と判定する最低コサイン類似度
            cache: EmbeddingCache インスタンス（省略時はキャッシュ無効）
        """
        self.encoder = VoiceEncoder()
        self.threshold = threshold
        self.reference_embeddings: dict[str, np.ndarray] = {}
        self._cache = cache

    def register_speaker(self, speaker_id: str, audio_paths: list[str]) -> None:
        """基準音声から話者の声紋ベクトルを登録する。

        Args:
            speaker_id: 話者ID（例: "person_a"）
            audio_paths: 基準音声ファイルパスのリスト
        """
        embeddings = []
        for path in audio_paths:
            cached = self._cache.get(path, prefix="voice") if self._cache else None
            if cached is not None:
                embeddings.append(cached)
                continue
            wav = preprocess_wav(Path(path))
            embedding = self.encoder.embed_utterance(wav)
            if self._cache:
                self._cache.put(path, embedding, prefix="voice")
            embeddings.append(embedding)

        self.reference_embeddings[speaker_id] = np.mean(embeddings, axis=0)
        logger.info("話者登録完了: %s (%d ファイル)", speaker_id, len(audio_paths))

    def register_speakers_from_dir(self, reference_dir: str) -> None:
        """ディレクトリ構造から全話者を一括登録する。

        ディレクトリ構成:
            reference_dir/
            ├── person_a/
            │   └── *.wav
            ├── person_b/
            │   └── *.wav
            └── person_c/
                └── *.wav
        """
        ref_path = Path(reference_dir)
        if not ref_path.exists():
            raise FileNotFoundError(f"基準音声ディレクトリが見つかりません: {ref_path}")

        for speaker_dir in sorted(ref_path.iterdir()):
            if not speaker_dir.is_dir():
                continue

            audio_files = list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.mp3"))
            if not audio_files:
                logger.debug("音声ファイルなし: %s", speaker_dir)
                continue

            self.register_speaker(speaker_dir.name, [str(f) for f in audio_files])

    def compare(self, audio_path: str) -> dict[str, float]:
        """音声ファイルを全登録話者と照合し、類似度スコアを返す。

        Args:
            audio_path: 照合対象の音声ファイルパス

        Returns:
            {話者ID: コサイン類似度} の辞書
        """
        if not self.reference_embeddings:
            raise RuntimeError("基準話者が登録されていません。先にregister_speakerを呼んでください。")

        cached = self._cache.get(audio_path, prefix="voice") if self._cache else None
        if cached is not None:
            embedding = cached
        else:
            wav = preprocess_wav(Path(audio_path))
            if len(wav) == 0:
                return {sid: 0.0 for sid in self.reference_embeddings}
            embedding = self.encoder.embed_utterance(wav)
            if self._cache:
                self._cache.put(audio_path, embedding, prefix="voice")
        scores = {}
        for speaker_id, ref_embedding in self.reference_embeddings.items():
            similarity = np.dot(embedding, ref_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)
            )
            scores[speaker_id] = float(similarity)

        return scores

    def compare_segments(self, segments: list[dict]) -> dict[str, dict]:
        """複数の音声セグメントを一括照合する。

        Args:
            segments: [{"start": float, "end": float, "audio_path": str}, ...] のリスト

        Returns:
            {話者ID: {"max_score": float, "avg_score": float, "segments": int}} の辞書
        """
        all_scores: dict[str, list[float]] = {
            sid: [] for sid in self.reference_embeddings
        }

        for segment in segments:
            scores = self.compare(segment["audio_path"])
            for sid, score in scores.items():
                all_scores[sid].append(score)

        results = {}
        for sid, scores in all_scores.items():
            if scores:
                results[sid] = {
                    "max_score": float(np.max(scores)),
                    "avg_score": float(np.mean(scores)),
                    "matching_segments": int(np.sum(np.array(scores) >= self.threshold)),
                    "total_segments": len(scores),
                }
            else:
                results[sid] = {
                    "max_score": 0.0,
                    "avg_score": 0.0,
                    "matching_segments": 0,
                    "total_segments": 0,
                }

        return results

    def identify(self, audio_path: str) -> tuple[str | None, float]:
        """音声ファイルから最も一致する話者を特定する。

        Returns:
            (話者ID, 類似度) のタプル。閾値未満の場合は (None, 最大スコア)
        """
        scores = self.compare(audio_path)
        if not scores:
            return None, 0.0

        best_speaker = max(scores, key=scores.get)
        best_score = scores[best_speaker]

        if best_score >= self.threshold:
            return best_speaker, best_score
        return None, best_score
