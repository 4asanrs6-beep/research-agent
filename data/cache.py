"""ローカルデータキャッシュ（Parquet形式）"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Parquet形式のローカルデータキャッシュ"""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours

    def _key_to_path(self, key: str) -> Path:
        safe_name = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_name}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        import time
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.ttl_hours:
            logger.debug("キャッシュ期限切れ: %s (%.1f時間)", key, age_hours)
            return None

        try:
            df = pd.read_parquet(path)
            logger.debug("キャッシュヒット: %s (%d行)", key, len(df))
            return df
        except Exception as e:
            logger.warning("キャッシュ読み込みエラー: %s - %s", key, e)
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        path = self._key_to_path(key)
        try:
            df.to_parquet(path, index=False)
            logger.debug("キャッシュ保存: %s (%d行)", key, len(df))
        except Exception as e:
            logger.warning("キャッシュ保存エラー: %s - %s", key, e)

    def invalidate(self, key: str) -> None:
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()

    def clear_all(self) -> int:
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        return count

    def get_stats(self) -> dict:
        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
