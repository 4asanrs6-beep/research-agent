"""スター株検出 + first-hit TTE計算 + 学習データ構築"""

import logging
import time as _time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import OnsetDetectorConfig

logger = logging.getLogger(__name__)

# 一般株式の市場区分
_STOCK_MARKET_SEGMENTS = {"プライム", "スタンダード", "グロース"}

# 推定時価総額マッピング
_SCALE_CAP_MAP = {
    "TOPIX Core30": 5000.0,
    "TOPIX Large70": 2000.0,
    "TOPIX Mid400": 500.0,
    "TOPIX Small 1": 100.0,
    "TOPIX Small 2": 50.0,
}


class OnsetLabeler:
    """スター株検出・first-hit計算・学習データ構築"""

    def __init__(self, data_provider, config: OnsetDetectorConfig):
        self.provider = data_provider
        self.config = config

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------
    def fetch_market_data(
        self,
        progress_callback=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """all_prices, topix, listed_stocksを取得

        月単位チャンクで429レート制限を回避する。
        """
        cfg = self.config

        # 日付範囲
        start_date = cfg.start_date or str(date.today() - timedelta(days=365))
        end_date = cfg.end_date or str(date.today())

        def _prog(msg):
            if progress_callback:
                progress_callback(msg)

        _prog("銘柄メタデータ取得中...")
        listed_stocks = self._api_call_with_retry(
            lambda: self.provider.get_listed_stocks()
        )
        # 一般株式のみ
        if "market_name" in listed_stocks.columns:
            listed_stocks = listed_stocks[
                listed_stocks["market_name"].isin(_STOCK_MARKET_SEGMENTS)
            ].copy()

        _prog("TOPIX取得中...")
        topix = self._api_call_with_retry(
            lambda: self.provider.get_index_prices(
                "0000", start_date, end_date,
            )
        )

        _prog("全銘柄株価取得中（月単位チャンク）...")
        all_prices = self._fetch_all_prices_chunked(
            start_date, end_date, _prog,
        )

        # adj_close 正規化
        if "adj_close" not in all_prices.columns and "close" in all_prices.columns:
            all_prices["adj_close"] = all_prices["close"]

        return all_prices, topix, listed_stocks

    @staticmethod
    def _api_call_with_retry(fn, max_retries: int = 5):
        """429エラー時にウェイト+リトライ"""
        last_err = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if "429" in str(e) or "too many" in str(e).lower():
                    wait = 15.0 * (attempt + 1)
                    logger.warning(
                        "429レート制限 (試行%d/%d)、%d秒待機...",
                        attempt + 1, max_retries, int(wait),
                    )
                    _time.sleep(wait)
                else:
                    raise
        raise last_err

    def _fetch_all_prices_chunked(self, start_date: str, end_date: str, _prog=None):
        """全銘柄株価を月単位チャンクで取得（429対策）"""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        chunks = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + pd.DateOffset(months=1) - pd.DateOffset(days=1), end)
            chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            cursor = cursor + pd.DateOffset(months=1)

        frames = []
        total = len(chunks)
        for i, (c_start, c_end) in enumerate(chunks):
            if _prog:
                _prog(f"株価取得中... ({i + 1}/{total}チャンク)")

            df = self._api_call_with_retry(
                lambda s=c_start, e=c_end: self.provider.get_price_daily(
                    code=None, start_date=s, end_date=e,
                )
            )
            if df is not None and not df.empty:
                frames.append(df)

            # チャンク間ウェイト（429予防）
            if i < total - 1:
                _time.sleep(4.0)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 信用取引データ取得・マージ
    # ------------------------------------------------------------------
    def _fetch_margin_data_bulk(
        self,
        start_date: str,
        end_date: str,
        _prog=None,
    ) -> pd.DataFrame:
        """金曜日リストで全銘柄の週次信用取引データを一括取得

        Returns
        -------
        pd.DataFrame
            週次信用取引データ（全銘柄）
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # 金曜日リスト生成
        fridays = pd.date_range(start, end, freq="W-FRI")
        if len(fridays) == 0:
            return pd.DataFrame()

        frames = []
        total = len(fridays)
        for i, friday in enumerate(fridays):
            date_str = friday.strftime("%Y-%m-%d")
            if _prog and i % 10 == 0:
                _prog(f"信用取引データ取得中... ({i + 1}/{total}週)")

            try:
                df = self._api_call_with_retry(
                    lambda d=date_str: self.provider.get_margin_trading_by_date(d),
                    max_retries=3,
                )
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning("信用取引データ取得失敗 (%s): %s", date_str, e)

            # APIウェイト（429予防）
            if i < total - 1:
                _time.sleep(2.0)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def merge_margin_into_prices(
        all_prices: pd.DataFrame,
        margin_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """信用取引データをall_pricesにマージ（週次→日次ffill展開）

        Parameters
        ----------
        all_prices : pd.DataFrame
            日次株価データ（code, date, ...）
        margin_df : pd.DataFrame
            週次信用取引データ（Code/code, Date/date, LongVol, ShrtVol, ...）

        Returns
        -------
        pd.DataFrame
            信用カラム追加済みの all_prices
        """
        if margin_df is None or margin_df.empty:
            return all_prices

        mdf = margin_df.copy()

        # カラム名正規化
        if "Date" in mdf.columns:
            mdf = mdf.rename(columns={"Date": "date"})
        if "Code" in mdf.columns:
            mdf = mdf.rename(columns={"Code": "code"})

        if "date" not in mdf.columns or "code" not in mdf.columns:
            logger.warning("信用取引データに date/code カラムがありません")
            return all_prices

        mdf["date"] = pd.to_datetime(mdf["date"])
        mdf["code"] = mdf["code"].astype(str)

        # 買残/売残カラム検出
        buy_col = None
        for c in ["LongVol", "LongMarginTradeVolume", "MarginBuyBalance"]:
            if c in mdf.columns:
                buy_col = c
                break
        sell_col = None
        for c in ["ShrtVol", "ShortMarginTradeVolume", "MarginSellBalance"]:
            if c in mdf.columns:
                sell_col = c
                break

        if buy_col is None or sell_col is None:
            logger.warning("信用買残/売残カラムが見つかりません: %s", mdf.columns.tolist())
            return all_prices

        # 基本指標算出
        mdf["margin_buy_balance"] = pd.to_numeric(mdf[buy_col], errors="coerce")
        mdf["margin_sell_balance"] = pd.to_numeric(mdf[sell_col], errors="coerce")
        mdf["margin_ratio"] = mdf["margin_buy_balance"] / mdf["margin_sell_balance"].replace(0, np.nan)

        # 前週比変化率
        mdf = mdf.sort_values(["code", "date"])
        mdf["margin_buy_change_pct"] = (
            mdf.groupby("code")["margin_buy_balance"].pct_change(fill_method=None)
        )
        mdf["margin_sell_change_pct"] = (
            mdf.groupby("code")["margin_sell_balance"].pct_change(fill_method=None)
        )
        mdf["margin_ratio_change_pct"] = (
            mdf.groupby("code")["margin_ratio"].pct_change(fill_method=None)
        )

        # マージ用カラム選択
        margin_cols = [
            "date", "code",
            "margin_buy_balance", "margin_sell_balance", "margin_ratio",
            "margin_buy_change_pct", "margin_sell_change_pct", "margin_ratio_change_pct",
        ]
        margin_merge = mdf[[c for c in margin_cols if c in mdf.columns]].copy()

        # all_prices にleft merge
        all_prices = all_prices.copy()
        all_prices["date"] = pd.to_datetime(all_prices["date"])
        all_prices["code"] = all_prices["code"].astype(str)

        merged = all_prices.merge(margin_merge, on=["date", "code"], how="left")

        # 銘柄別ffill（週次→日次展開）
        margin_fill_cols = [
            "margin_buy_balance", "margin_sell_balance", "margin_ratio",
            "margin_buy_change_pct", "margin_sell_change_pct", "margin_ratio_change_pct",
        ]
        fill_cols_present = [c for c in margin_fill_cols if c in merged.columns]
        if fill_cols_present:
            merged = merged.sort_values(["code", "date"])
            merged[fill_cols_present] = merged.groupby("code")[fill_cols_present].ffill()

        logger.info(
            "信用取引データマージ完了: %d銘柄に信用データ付与",
            merged["margin_ratio"].notna().groupby(merged["code"]).any().sum(),
        )
        return merged

    # ------------------------------------------------------------------
    # スター株検出（既存ロジック再利用）
    # ------------------------------------------------------------------
    def identify_star_stocks(
        self,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
    ) -> list[dict]:
        """スター株を取得する。

        優先順位:
        1. config.star_stocks_input が設定済み → そのまま使用（ページ4結果等）
        2. config.user_star_codes が指定 → 該当銘柄をスター株として扱う
        3. どちらもなし → 自動検出
        """
        cfg = self.config

        # 1. 外部注入（スター株分析ページの結果）
        if cfg.star_stocks_input:
            logger.info(f"外部スター株使用: {len(cfg.star_stocks_input)}件")
            return cfg.star_stocks_input

        # 2. 手動指定コード
        if cfg.user_star_codes:
            return self._resolve_user_codes(
                cfg.user_star_codes, all_prices, topix, listed_stocks
            )
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        # TOPIX期間リターン
        topix_close = "close" if "close" in topix.columns else topix.columns[-1]
        topix_sorted = topix.sort_values("date")
        topix_first = topix_sorted[topix_close].iloc[0]
        topix_last = topix_sorted[topix_close].iloc[-1]
        topix_ret = (topix_last / topix_first - 1) if topix_first > 0 else 0.0

        stock_stats = []
        for code, grp in all_prices.groupby("code"):
            grp = grp.sort_values("date")
            if len(grp) < 20:
                continue
            first_price = grp[close_col].iloc[0]
            last_price = grp[close_col].iloc[-1]
            if first_price <= 0:
                continue

            total_return = last_price / first_price - 1
            excess_return = total_return - topix_ret

            # 出来高変化率
            n_half = len(grp) // 2
            vol_first = grp["volume"].iloc[:n_half].mean()
            vol_second = grp["volume"].iloc[n_half:].mean()
            vol_ratio = vol_second / vol_first if vol_first > 0 else 1.0

            daily_rets = grp[close_col].pct_change(fill_method=None).dropna()
            peak = grp[close_col].cummax()
            drawdown = ((peak - grp[close_col]) / peak).max()
            max_daily = daily_rets.abs().max() if len(daily_rets) > 0 else 0
            up_ratio = (daily_rets > 0).mean() if len(daily_rets) > 0 else 0

            stock_stats.append({
                "code": str(code),
                "total_return": float(total_return),
                "excess_return": float(excess_return),
                "volume_change_ratio": float(vol_ratio),
                "max_drawdown": float(drawdown),
                "max_single_day_return": float(max_daily),
                "up_days_ratio": float(up_ratio),
                "first_price": float(first_price),
                "last_price": float(last_price),
                "n_days": len(grp),
            })

        # メタデータ結合
        meta_cols = ["code", "name", "sector_17_name", "market_name", "scale_category"]
        available = [c for c in meta_cols if c in listed_stocks.columns]
        meta_map = {}
        if available:
            for _, row in listed_stocks[available].iterrows():
                meta_map[str(row["code"])] = {c: row.get(c, "") for c in available}

        results = []
        for s in stock_stats:
            code = s["code"]
            meta = meta_map.get(code, {})
            s.update({
                "name": meta.get("name", ""),
                "sector": meta.get("sector_17_name", ""),
                "market": meta.get("market_name", ""),
                "scale_category": meta.get("scale_category", ""),
            })
            est_cap = _SCALE_CAP_MAP.get(s["scale_category"], 30.0)
            s["est_market_cap_billion"] = est_cap

            # 仕手株フィルタ
            flags = self._check_pump_dump(s)
            if flags:
                continue

            if (s["total_return"] >= cfg.star_min_total_return
                    and s["excess_return"] >= cfg.star_min_excess_return
                    and s["volume_change_ratio"] >= cfg.star_min_volume_ratio):
                s["pump_dump_flags"] = []
                results.append(s)

        results.sort(key=lambda x: x["excess_return"], reverse=True)
        results = results[:cfg.star_max_auto_detect]
        logger.info(f"スター株 {len(results)} 件検出")
        return results

    def _resolve_user_codes(
        self,
        user_codes: list[str],
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
    ) -> list[dict]:
        """手動指定コードからスター株情報を構築"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        topix_close = "close" if "close" in topix.columns else topix.columns[-1]
        topix_sorted = topix.sort_values("date")
        topix_first = topix_sorted[topix_close].iloc[0]
        topix_last = topix_sorted[topix_close].iloc[-1]
        topix_ret = (topix_last / topix_first - 1) if topix_first > 0 else 0.0

        meta_cols = ["code", "name", "sector_17_name", "market_name", "scale_category"]
        available = [c for c in meta_cols if c in listed_stocks.columns]
        meta_map = {}
        if available:
            for _, row in listed_stocks[available].iterrows():
                meta_map[str(row["code"])] = {c: row.get(c, "") for c in available}

        # 4桁→5桁正規化
        normalized = []
        for c in user_codes:
            c = c.strip()
            if len(c) == 4 and c.isalnum():
                c = c + "0"
            normalized.append(c)

        results = []
        for code in normalized:
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 20:
                grp = all_prices[all_prices["code"].astype(str) == code].sort_values("date")
            if len(grp) < 20:
                logger.warning(f"銘柄 {code}: データ不足（{len(grp)}行）")
                continue

            first_price = grp[close_col].iloc[0]
            last_price = grp[close_col].iloc[-1]
            if first_price <= 0:
                continue

            total_return = last_price / first_price - 1
            excess_return = total_return - topix_ret
            n_half = len(grp) // 2
            vol_first = grp["volume"].iloc[:n_half].mean()
            vol_second = grp["volume"].iloc[n_half:].mean()
            vol_ratio = vol_second / vol_first if vol_first > 0 else 1.0

            daily_rets = grp[close_col].pct_change(fill_method=None).dropna()
            peak = grp[close_col].cummax()
            drawdown = ((peak - grp[close_col]) / peak).max()
            max_daily = daily_rets.abs().max() if len(daily_rets) > 0 else 0
            up_ratio = (daily_rets > 0).mean() if len(daily_rets) > 0 else 0

            meta = meta_map.get(code, {})
            results.append({
                "code": code,
                "total_return": float(total_return),
                "excess_return": float(excess_return),
                "volume_change_ratio": float(vol_ratio),
                "max_drawdown": float(drawdown),
                "max_single_day_return": float(max_daily),
                "up_days_ratio": float(up_ratio),
                "first_price": float(first_price),
                "last_price": float(last_price),
                "n_days": len(grp),
                "name": meta.get("name", ""),
                "sector": meta.get("sector_17_name", ""),
                "market": meta.get("market_name", ""),
                "scale_category": meta.get("scale_category", ""),
                "est_market_cap_billion": _SCALE_CAP_MAP.get(meta.get("scale_category", ""), 30.0),
                "pump_dump_flags": [],
                "source": "user",
            })

        logger.info(f"手動指定スター株: {len(results)}件 / {len(normalized)}件指定")
        return results

    def _check_pump_dump(self, s: dict) -> list[str]:
        cfg = self.config
        flags = []
        if s["est_market_cap_billion"] < cfg.star_min_market_cap_billion:
            flags.append("時価総額小")
        if s["max_drawdown"] > cfg.star_max_drawdown:
            flags.append("高値からの下落率大")
        if s["max_single_day_return"] > cfg.star_max_single_day_return:
            flags.append("1日最大変動大")
        if s["up_days_ratio"] < cfg.star_min_up_days_ratio:
            flags.append("上昇日比率低")
        return flags

    # ------------------------------------------------------------------
    # 類型発見（K-means）
    # ------------------------------------------------------------------
    def discover_star_types(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
    ) -> dict:
        """スター株をK-meansで類型分類する"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        cfg = self.config
        n_clusters = min(cfg.n_star_types, len(star_stocks))
        if n_clusters < 2:
            for s in star_stocks:
                s["star_type"] = 0
            return {"n_types": 1, "labels": [0] * len(star_stocks), "profiles": []}

        feature_keys = [
            "total_return", "excess_return", "max_drawdown",
            "volume_change_ratio", "up_days_ratio",
        ]
        X = np.array([
            [float(s.get(k, 0) or 0) for k in feature_keys]
            for s in star_stocks
        ])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        for i, s in enumerate(star_stocks):
            s["star_type"] = int(labels[i])

        profiles = []
        for c in range(n_clusters):
            mask = labels == c
            members = [star_stocks[i]["code"] for i in range(len(star_stocks)) if mask[i]]
            centroid = kmeans.cluster_centers_[c]
            profiles.append({
                "type_id": c,
                "n_members": int(mask.sum()),
                "member_codes": members,
                "centroid": {feature_keys[j]: round(float(centroid[j]), 4) for j in range(len(feature_keys))},
            })

        return {
            "n_types": n_clusters,
            "labels": labels.tolist(),
            "profiles": profiles,
            "scaler": scaler,
            "kmeans": kmeans,
            "feature_keys": feature_keys,
        }

    # ------------------------------------------------------------------
    # First-hit TTE計算
    # ------------------------------------------------------------------
    @staticmethod
    def compute_first_hit(
        prices_series: pd.Series,
        topix_series: pd.Series,
        t_idx: int,
        excess_threshold: float = 0.30,
        horizon: int = 60,
    ) -> int | None:
        """日t_idxからの初回到達日数を計算

        Parameters
        ----------
        prices_series : pd.Series
            銘柄の終値系列（index=日付）
        topix_series : pd.Series
            TOPIXの終値系列（index=日付）
        t_idx : int
            計算開始の位置インデックス
        excess_threshold : float
            超過リターン閾値
        horizon : int
            到達判定地平

        Returns
        -------
        int | None
            初回到達日数。未到達ならNone
        """
        n = len(prices_series)
        if t_idx >= n - 1:
            return None

        p0 = prices_series.iloc[t_idx]
        if p0 <= 0:
            return None

        # TOPIX基準価格
        t0_topix = topix_series.iloc[t_idx] if t_idx < len(topix_series) else None

        for k in range(1, min(horizon + 1, n - t_idx)):
            pk = prices_series.iloc[t_idx + k]
            stock_ret = pk / p0 - 1

            if t0_topix is not None and t0_topix > 0 and (t_idx + k) < len(topix_series):
                topix_ret = topix_series.iloc[t_idx + k] / t0_topix - 1
                excess = stock_ret - topix_ret
            else:
                excess = stock_ret

            if excess >= excess_threshold:
                return k

        return None

    # ------------------------------------------------------------------
    # 学習データ構築
    # ------------------------------------------------------------------
    def build_training_dataset(
        self,
        star_stocks: list[dict],
        star_types: dict,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
    ) -> dict:
        """Stage 1/2用の学習データを構築する

        Returns
        -------
        dict with keys:
            samples: list[dict]  各サンプルのメタ情報
            y1: np.ndarray       Stage 1ターゲット (binary)
            y2: np.ndarray       Stage 2ターゲット (first_hit値, 到達のみ)
            sample_weights: np.ndarray  Stage 1の静穏度重み
            stage2_mask: np.ndarray     Stage 2で使うサンプルのmask
            first_hits: np.ndarray      全サンプルのfirst_hit値 (Noneは-1)
        """
        cfg = self.config
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        # TOPIX終値系列
        topix_sorted = topix.sort_values("date").reset_index(drop=True)
        topix_close_col = "close" if "close" in topix_sorted.columns else topix_sorted.columns[-1]
        topix_dates = pd.to_datetime(topix_sorted["date"])
        topix_close = topix_sorted[topix_close_col].astype(float)

        star_codes = {s["code"] for s in star_stocks}
        star_type_map = {}
        for s in star_stocks:
            star_type_map[s["code"]] = s.get("star_type", 0)

        samples = []
        y1_list = []
        first_hit_list = []
        weight_list = []

        # --- スター株サンプル ---
        logger.info("スター株サンプル構築中...")
        for s in star_stocks:
            code = s["code"]
            grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
            if len(grp) < 30:
                continue

            prices = grp[close_col].astype(float)
            dates = pd.to_datetime(grp["date"])

            # TOPIX系列をアラインメント
            topix_aligned = self._align_topix(dates, topix_dates, topix_close)

            # 特徴量計算に十分な履歴を確保するため、先頭をスキップ
            start_idx = max(cfg.feature_history_days, 20)

            for t in range(start_idx, len(grp) - 5):  # 末尾5日分は余裕
                fh = self.compute_first_hit(
                    prices, topix_aligned, t,
                    excess_threshold=cfg.excess_threshold,
                    horizon=cfg.horizon,
                )

                # Stage 1ラベル
                y1 = 1 if (fh is not None and fh <= cfg.T_near) else 0

                # 静穏度重み計算
                weight = self._compute_quietness_weight(grp, close_col, t)

                # 過熱フラグ
                is_overheated = self._is_overheated(grp, close_col, t, topix_aligned)

                meta = meta_map.get(code, {}) if 'meta_map' in dir() else {}
                samples.append({
                    "code": code,
                    "date_idx": t,
                    "date": str(dates.iloc[t].date()) if t < len(dates) else "",
                    "star_type": star_type_map.get(code, 0),
                    "is_star": True,
                    "is_overheated": is_overheated,
                    "sector": s.get("sector", ""),
                    "scale_category": s.get("scale_category", ""),
                })
                y1_list.append(y1)
                first_hit_list.append(fh if fh is not None else -1)
                weight_list.append(weight)

        logger.info(f"スター株サンプル: {len(samples)}件 (positive={sum(y1_list)})")

        # --- 非スター制御群 ---
        logger.info("制御群サンプル構築中...")
        non_star_codes = [
            c for c in all_prices["code"].unique()
            if str(c) not in star_codes
        ]
        # ランダムサンプリング（再現性確保）
        rng = np.random.RandomState(42)
        n_control = min(
            int(len(star_stocks) * cfg.control_sample_ratio),
            len(non_star_codes)
        )
        control_codes = rng.choice(non_star_codes, size=n_control, replace=False)

        for code in control_codes:
            code_str = str(code)
            grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
            if len(grp) < cfg.feature_history_days + 10:
                continue

            prices = grp[close_col].astype(float)
            dates = pd.to_datetime(grp["date"])
            topix_aligned = self._align_topix(dates, topix_dates, topix_close)

            # 制御群は等間隔でサンプリング（全営業日だと多すぎる）
            step = max(1, (len(grp) - cfg.feature_history_days - 5) // 10)
            start_idx = cfg.feature_history_days

            for t in range(start_idx, len(grp) - 5, step):
                fh = self.compute_first_hit(
                    prices, topix_aligned, t,
                    excess_threshold=cfg.excess_threshold,
                    horizon=cfg.horizon,
                )

                y1 = 1 if (fh is not None and fh <= cfg.T_near) else 0
                weight = self._compute_quietness_weight(grp, close_col, t)
                is_overheated = self._is_overheated(grp, close_col, t, topix_aligned)

                samples.append({
                    "code": code_str,
                    "date_idx": t,
                    "date": str(dates.iloc[t].date()) if t < len(dates) else "",
                    "star_type": -1,
                    "is_star": False,
                    "is_overheated": is_overheated,
                    "sector": "",
                    "scale_category": "",
                })
                y1_list.append(y1)
                first_hit_list.append(fh if fh is not None else -1)
                weight_list.append(weight)

        y1 = np.array(y1_list, dtype=np.int32)
        first_hits = np.array(first_hit_list, dtype=np.float64)
        weights = np.array(weight_list, dtype=np.float64)
        # NaN/Inf安全化
        weights = np.where(np.isfinite(weights), weights, 0.5)
        weights = np.clip(weights, 0.01, 1.0)

        # Stage 2: 到達サンプル全体
        stage2_mask = first_hits > 0
        y2 = np.log1p(np.where(stage2_mask, first_hits, 0))

        logger.info(
            f"学習データ完成: {len(samples)}サンプル, "
            f"Stage1 positive={y1.sum()}, Stage2 samples={stage2_mask.sum()}"
        )

        return {
            "samples": samples,
            "y1": y1,
            "y2": y2,
            "sample_weights": weights,
            "stage2_mask": stage2_mask,
            "first_hits": first_hits,
        }

    # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------
    @staticmethod
    def _align_topix(
        stock_dates: pd.Series,
        topix_dates: pd.Series,
        topix_close: pd.Series,
    ) -> pd.Series:
        """銘柄の日付列にTOPIX終値をアラインする"""
        topix_df = pd.DataFrame({"date": topix_dates, "topix_close": topix_close.values})
        topix_df["date"] = pd.to_datetime(topix_df["date"])
        stock_df = pd.DataFrame({"date": pd.to_datetime(stock_dates)})
        merged = stock_df.merge(topix_df, on="date", how="left")
        merged["topix_close"] = merged["topix_close"].ffill().bfill()
        return merged["topix_close"].reset_index(drop=True)

    def _compute_quietness_weight(
        self, grp: pd.DataFrame, close_col: str, t: int
    ) -> float:
        """静穏度重みを計算（静穏ほど高、過熱ほど低）"""
        try:
            close = grp[close_col].astype(float).values

            if t >= len(close) or t < 0:
                return 0.5

            # trailing 20d excess (簡易: 20日リターン)
            if t >= 20 and close[t - 20] > 0 and np.isfinite(close[t]) and np.isfinite(close[t - 20]):
                trailing_20d = close[t] / close[t - 20] - 1
            else:
                trailing_20d = 0.0

            # trailing 5d return
            if t >= 5 and close[t - 5] > 0 and np.isfinite(close[t]) and np.isfinite(close[t - 5]):
                trailing_5d = close[t] / close[t - 5] - 1
            else:
                trailing_5d = 0.0

            # RSI(14) 簡易計算
            rsi = self._calc_rsi(close, t, 14)

            if not np.isfinite(trailing_20d):
                trailing_20d = 0.0
            if not np.isfinite(trailing_5d):
                trailing_5d = 0.0
            if not np.isfinite(rsi):
                rsi = 50.0

            overheat_score = (
                abs(trailing_20d) / 0.10
                + abs(trailing_5d) / 0.05
                + max(0, rsi - 50) / 20
            )
            weight = 1.0 / (1.0 + overheat_score)
            return weight if np.isfinite(weight) else 0.5
        except Exception:
            return 0.5

    def _is_overheated(
        self, grp: pd.DataFrame, close_col: str, t: int,
        topix_aligned: pd.Series,
    ) -> bool:
        """過熱判定"""
        close = grp[close_col].astype(float).values
        cfg = self.config

        # trailing 20d excess
        if t >= 20 and close[t - 20] > 0:
            stock_ret = close[t] / close[t - 20] - 1
            if t < len(topix_aligned) and (t - 20) >= 0:
                t_val = topix_aligned.iloc[t]
                t_val_prev = topix_aligned.iloc[t - 20]
                topix_ret = (t_val / t_val_prev - 1) if t_val_prev > 0 else 0
                trailing_20d_excess = stock_ret - topix_ret
            else:
                trailing_20d_excess = stock_ret
        else:
            trailing_20d_excess = 0.0

        if trailing_20d_excess > cfg.overheat_trailing_20d_excess:
            return True

        # trailing 5d return
        if t >= 5 and close[t - 5] > 0:
            trailing_5d = close[t] / close[t - 5] - 1
            if trailing_5d > cfg.overheat_trailing_5d_return:
                return True

        return False

    @staticmethod
    def _calc_rsi(close: np.ndarray, t: int, period: int = 14) -> float:
        """RSI簡易計算"""
        if t < period + 1:
            return 50.0
        rets = np.diff(close[t - period:t + 1])
        gains = np.where(rets > 0, rets, 0)
        losses = np.where(rets < 0, -rets, 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
