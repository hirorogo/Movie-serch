"""埋め込みキャッシュのテスト"""

import numpy as np
import pytest

from src.cache import EmbeddingCache


class TestEmbeddingCache:
    def test_put_and_get(self, tmp_path):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        embedding = np.array([1.0, 2.0, 3.0])
        cache.put(str(test_file), embedding, prefix="voice")

        result = cache.get(str(test_file), prefix="voice")
        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding)

    def test_cache_miss(self, tmp_path):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        result = cache.get(str(test_file), prefix="voice")
        assert result is None

    def test_cache_stats(self, tmp_path):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        cache.get(str(test_file), prefix="voice")  # miss
        cache.put(str(test_file), np.array([1.0]), prefix="voice")
        cache.get(str(test_file), prefix="voice")  # hit

        assert cache.stats == {"hits": 1, "misses": 1}

    def test_clear(self, tmp_path):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"dummy audio data")

        cache.put(str(test_file), np.array([1.0, 2.0]), prefix="voice")
        count = cache.clear()

        assert count == 1
        assert cache.get(str(test_file), prefix="voice") is None

    def test_different_files_different_cache(self, tmp_path):
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        file_a = tmp_path / "a.wav"
        file_b = tmp_path / "b.wav"
        file_a.write_bytes(b"audio data a")
        file_b.write_bytes(b"audio data b")

        cache.put(str(file_a), np.array([1.0, 0.0]), prefix="voice")
        cache.put(str(file_b), np.array([0.0, 1.0]), prefix="voice")

        result_a = cache.get(str(file_a), prefix="voice")
        result_b = cache.get(str(file_b), prefix="voice")
        np.testing.assert_array_almost_equal(result_a, [1.0, 0.0])
        np.testing.assert_array_almost_equal(result_b, [0.0, 1.0])
