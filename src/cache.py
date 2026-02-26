"""埋め込みキャッシュ - 声紋・視覚特徴ベクトルの再計算を回避"""

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".cache/embeddings")


class EmbeddingCache:
    """ファイルベースの埋め込みキャッシュ。

    音声ファイルの MD5 ハッシュをキーとして、
    計算済みの埋め込みベクトルを .npy 形式で保存する。
    """

    def __init__(self, cache_dir: Path | str = _DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _file_hash(self, file_path: str) -> str:
        """ファイルの MD5 ハッシュを計算する。"""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _cache_path(self, file_path: str, prefix: str) -> Path:
        """キャッシュファイルのパスを生成する。"""
        file_hash = self._file_hash(file_path)
        return self.cache_dir / f"{prefix}_{file_hash}.npy"

    def get(self, file_path: str, prefix: str = "emb") -> np.ndarray | None:
        """キャッシュから埋め込みベクトルを取得する。

        Args:
            file_path: 元の音声/画像ファイルパス
            prefix: キャッシュキーのプレフィックス

        Returns:
            キャッシュがあれば ndarray、なければ None
        """
        cache_file = self._cache_path(file_path, prefix)
        if cache_file.exists():
            self._hits += 1
            logger.debug("キャッシュヒット: %s", file_path)
            return np.load(cache_file)
        self._misses += 1
        return None

    def put(self, file_path: str, embedding: np.ndarray, prefix: str = "emb") -> None:
        """埋め込みベクトルをキャッシュに保存する。

        Args:
            file_path: 元の音声/画像ファイルパス
            embedding: 保存する埋め込みベクトル
            prefix: キャッシュキーのプレフィックス
        """
        cache_file = self._cache_path(file_path, prefix)
        np.save(cache_file, embedding)
        logger.debug("キャッシュ保存: %s", file_path)

    def clear(self) -> int:
        """キャッシュを全て削除する。

        Returns:
            削除したファイル数
        """
        count = 0
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
            count += 1
        logger.info("キャッシュクリア: %d ファイル削除", count)
        return count

    @property
    def stats(self) -> dict[str, int]:
        """キャッシュ統計を返す。"""
        return {"hits": self._hits, "misses": self._misses}
