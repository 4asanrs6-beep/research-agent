"""パラメータ化シグナル生成エンジン

テクニカル指標 + 信用取引指標を組み合わせてエントリーシグナルを生成する。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """シグナル生成パラメータ"""

    # --- テクニカル ---
    consecutive_bullish_days: int | None = None
    consecutive_bearish_days: int | None = None
    volume_surge_ratio: float | None = None
    volume_surge_window: int = 20
    price_vs_ma25: str | None = None   # "above" / "below" / None
    price_vs_ma75: str | None = None
    price_vs_ma200: str | None = None
    ma_deviation_pct: float | None = None
    ma_deviation_window: int = 25
    rsi_window: int = 14
    rsi_lower: float | None = None
    rsi_upper: float | None = None
    bb_window: int = 20
    bb_std: float = 2.0
    bb_buy_below_lower: bool = False
    ma_cross_short: int | None = None
    ma_cross_long: int | None = None
    ma_cross_type: str = "golden_cross"   # "golden_cross" / "dead_cross"
    macd_fast: int | None = None
    macd_slow: int | None = None
    macd_signal: int = 9
    atr_window: int = 14
    atr_max: float | None = None

    # --- 一目均衡表 ---
    ichimoku_cloud: str | None = None          # "above" / "below" / None
    ichimoku_tenkan_above_kijun: bool = False

    # --- セクター相対強度 ---
    sector_relative_strength_min: float | None = None  # パーセンタイル下限 (0-100)
    sector_relative_lookback: int = 20

    # --- 信用取引 ---
    margin_type: str = "combined"  # "combined"(合算) / "standard"(制度信用) / "negotiable"(一般信用)
    margin_ratio_min: float | None = None
    margin_ratio_max: float | None = None
    short_selling_ratio_max: float | None = None

    # --- ポジション管理 ---
    holding_period_days: int = 20
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    max_positions: int = 10
    allocation_method: str = "equal_weight"  # "equal_weight" / "inverse_vol"

    # --- ロジック ---
    signal_logic: str = "AND"  # "AND" / "OR"

    def get_active_conditions_summary(self) -> list[str]:
        """有効なシグナル条件のサマリーを返す"""
        parts = []
        if self.consecutive_bullish_days is not None:
            parts.append(f"連続陽線{self.consecutive_bullish_days}日")
        if self.consecutive_bearish_days is not None:
            parts.append(f"連続陰線{self.consecutive_bearish_days}日")
        if self.volume_surge_ratio is not None:
            parts.append(f"出来高{self.volume_surge_ratio}倍")
        if self.price_vs_ma25 is not None:
            parts.append(f"25日線{'上' if self.price_vs_ma25 == 'above' else '下'}")
        if self.price_vs_ma75 is not None:
            parts.append(f"75日線{'上' if self.price_vs_ma75 == 'above' else '下'}")
        if self.price_vs_ma200 is not None:
            parts.append(f"200日線{'上' if self.price_vs_ma200 == 'above' else '下'}")
        if self.ma_deviation_pct is not None:
            parts.append(f"MA乖離{self.ma_deviation_pct}%")
        if self.rsi_lower is not None:
            parts.append(f"RSI<{self.rsi_lower}")
        if self.rsi_upper is not None:
            parts.append(f"RSI>{self.rsi_upper}")
        if self.bb_buy_below_lower:
            parts.append("BB下限タッチ")
        if self.ma_cross_short is not None and self.ma_cross_long is not None:
            label = "GC" if self.ma_cross_type == "golden_cross" else "DC"
            parts.append(f"{label}({self.ma_cross_short}/{self.ma_cross_long})")
        if self.macd_fast is not None and self.macd_slow is not None:
            parts.append("MACDクロス")
        if self.atr_max is not None:
            parts.append(f"ATR<{self.atr_max}")
        if self.ichimoku_cloud is not None:
            parts.append(f"雲の{'上' if self.ichimoku_cloud == 'above' else '下'}")
        if self.ichimoku_tenkan_above_kijun:
            parts.append("転換線>基準線")
        if self.sector_relative_strength_min is not None:
            parts.append(f"セクター相対強度>{self.sector_relative_strength_min}%")
        _MARGIN_TYPE_LABELS = {"combined": "合算", "standard": "制度信用", "negotiable": "一般信用"}
        mt_label = _MARGIN_TYPE_LABELS.get(self.margin_type, self.margin_type)
        if self.margin_ratio_min is not None:
            parts.append(f"貸借倍率({mt_label})>{self.margin_ratio_min}")
        if self.margin_ratio_max is not None:
            parts.append(f"貸借倍率({mt_label})<{self.margin_ratio_max}")
        if self.short_selling_ratio_max is not None:
            parts.append(f"空売り比率<{self.short_selling_ratio_max}")
        return parts

    def has_any_signal(self) -> bool:
        """少なくとも1つのシグナル条件が有効か"""
        return len(self.get_active_conditions_summary()) > 0

    def needs_margin_data(self) -> bool:
        """信用取引データが必要か"""
        return (
            self.margin_ratio_min is not None
            or self.margin_ratio_max is not None
            or self.short_selling_ratio_max is not None
        )


# ======================================================================
# ヘルパー: 連続カウント（ベクトル化）
# ======================================================================
def _vectorized_consecutive(flag: pd.Series, code: pd.Series) -> pd.Series:
    """flag=1 が連続する回数をベクトル化で算出する。code境界でリセット。"""
    code_change = code != code.shift(1)
    reset = (flag == 0) | code_change
    groups = reset.cumsum()
    return flag.groupby(groups).cumsum()


class SignalGenerator:
    """テクニカル指標を算出し、パラメータに基づいてエントリーシグナルを生成する"""

    # ------------------------------------------------------------------
    # テクニカル指標算出
    # ------------------------------------------------------------------
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """株価DataFrameにテクニカル指標列を追加して返す

        入力: columns=[date, code, open, high, low, close, volume, ...]
        出力: 同DataFrame + 各指標列
        """
        df = df.sort_values(["code", "date"]).reset_index(drop=True).copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        high_col = "adj_high" if "adj_high" in df.columns else "high"
        low_col = "adj_low" if "adj_low" in df.columns else "low"
        open_col = "open"
        g = df.groupby("code", sort=False)

        # --- 移動平均 ---
        for w in (25, 75, 200):
            df[f"ma{w}"] = g[close_col].transform(
                lambda s: s.rolling(w, min_periods=w).mean()
            )

        # --- RSI (Wilder's smoothing) ---
        delta = g[close_col].transform(lambda s: s.diff())
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.groupby(df["code"]).transform(
            lambda s: s.ewm(span=14, min_periods=14).mean()
        )
        avg_loss = loss.groupby(df["code"]).transform(
            lambda s: s.ewm(span=14, min_periods=14).mean()
        )
        # avg_loss=0 → RSI=100 (全て上昇)
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - 100 / (1 + rs)
        df.loc[avg_loss == 0, "rsi"] = 100.0
        df.loc[avg_gain == 0, "rsi"] = 0.0

        # --- ボリンジャーバンド ---
        df["bb_mid"] = g[close_col].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        bb_std_val = g[close_col].transform(
            lambda s: s.rolling(20, min_periods=20).std()
        )
        df["bb_upper"] = df["bb_mid"] + 2.0 * bb_std_val
        df["bb_lower"] = df["bb_mid"] - 2.0 * bb_std_val

        # --- MACD ---
        ema12 = g[close_col].transform(
            lambda s: s.ewm(span=12, min_periods=12).mean()
        )
        ema26 = g[close_col].transform(
            lambda s: s.ewm(span=26, min_periods=26).mean()
        )
        df["macd_line"] = ema12 - ema26
        df["macd_signal_line"] = df.groupby("code")["macd_line"].transform(
            lambda s: s.ewm(span=9, min_periods=9).mean()
        )

        # --- ATR ---
        prev_close = g[close_col].shift(1)
        tr1 = (df[high_col] - df[low_col]).abs()
        tr2 = (df[high_col] - prev_close).abs()
        tr3 = (df[low_col] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.groupby(df["code"]).transform(
            lambda s: s.rolling(14, min_periods=14).mean()
        )

        # --- 連続陽線 / 陰線（ベクトル化） ---
        if open_col in df.columns:
            bullish = (df[close_col] > df[open_col]).astype(int)
            bearish = (df[close_col] < df[open_col]).astype(int)
        else:
            bullish = pd.Series(0, index=df.index, dtype=int)
            bearish = pd.Series(0, index=df.index, dtype=int)

        df["consec_bullish"] = _vectorized_consecutive(bullish, df["code"])
        df["consec_bearish"] = _vectorized_consecutive(bearish, df["code"])

        # --- 出来高倍率 ---
        vol_col = "adj_volume" if "adj_volume" in df.columns else "volume"
        if vol_col in df.columns:
            vol_ma = g[vol_col].transform(
                lambda s: s.rolling(20, min_periods=5).mean()
            )
            df["volume_ratio"] = df[vol_col] / vol_ma.replace(0, np.nan)
        else:
            df["volume_ratio"] = np.nan

        # --- 一目均衡表 (transform ベース) ---
        rh9 = g[high_col].transform(lambda s: s.rolling(9, min_periods=9).max())
        rl9 = g[low_col].transform(lambda s: s.rolling(9, min_periods=9).min())
        df["ichi_tenkan"] = (rh9 + rl9) / 2

        rh26 = g[high_col].transform(lambda s: s.rolling(26, min_periods=26).max())
        rl26 = g[low_col].transform(lambda s: s.rolling(26, min_periods=26).min())
        df["ichi_kijun"] = (rh26 + rl26) / 2

        senkou_a = (df["ichi_tenkan"] + df["ichi_kijun"]) / 2
        rh52 = g[high_col].transform(lambda s: s.rolling(52, min_periods=52).max())
        rl52 = g[low_col].transform(lambda s: s.rolling(52, min_periods=52).min())
        senkou_b = (rh52 + rl52) / 2

        df["ichi_senkou_a"] = senkou_a
        df["ichi_senkou_b"] = senkou_b
        df["ichi_cloud_top"] = pd.concat(
            [senkou_a, senkou_b], axis=1
        ).max(axis=1)
        df["ichi_cloud_bottom"] = pd.concat(
            [senkou_a, senkou_b], axis=1
        ).min(axis=1)

        logger.info(
            "指標算出完了: %d行, %d銘柄, "
            "MA25非NaN=%d, consec_bullish>0=%d, volume_ratio非NaN=%d",
            len(df),
            df["code"].nunique(),
            df["ma25"].notna().sum(),
            (df["consec_bullish"] > 0).sum(),
            df["volume_ratio"].notna().sum(),
        )

        return df

    # ------------------------------------------------------------------
    # 信用取引指標
    # ------------------------------------------------------------------
    def compute_margin_indicators(
        self, prices_df: pd.DataFrame, margin_df: pd.DataFrame,
        margin_type: str = "combined",
    ) -> pd.DataFrame:
        """信用取引データを株価DFに結合する

        margin_df は週次データ想定。code, date を使って ffill で日次に展開。

        Args:
            margin_type: "combined"(合算), "standard"(制度信用), "negotiable"(一般信用)
        """
        if margin_df is None or margin_df.empty:
            logger.warning("信用取引データが空です")
            prices_df["margin_ratio"] = np.nan
            return prices_df

        mdf = margin_df.copy()
        logger.info("信用取引データ: %d行, カラム=%s, タイプ=%s", len(mdf), mdf.columns.tolist(), margin_type)

        # Date/Code の正規化
        if "Date" in mdf.columns:
            mdf = mdf.rename(columns={"Date": "date"})
        if "Code" in mdf.columns:
            mdf = mdf.rename(columns={"Code": "code"})

        if "date" in mdf.columns:
            mdf["date"] = pd.to_datetime(mdf["date"])
        if "code" not in mdf.columns:
            logger.warning("信用取引データに code カラムがありません: %s", mdf.columns.tolist())
            prices_df["margin_ratio"] = np.nan
            return prices_df

        # J-Quants API V2 カラム名に基づいて、タイプ別に買残/売残を選択
        # 合算: LongVol(買残), ShrtVol(売残)
        # 制度信用: LongStdVol(買残), ShrtStdVol(売残)
        # 一般信用: LongNegVol(買残), ShrtNegVol(売残)
        _MARGIN_COLS = {
            "combined":   {"buy": ["LongVol", "LongMarginTradeVolume", "MarginBuyBalance"],
                           "sell": ["ShrtVol", "ShortMarginTradeVolume", "MarginSellBalance"]},
            "standard":   {"buy": ["LongStdVol"], "sell": ["ShrtStdVol"]},
            "negotiable":  {"buy": ["LongNegVol"], "sell": ["ShrtNegVol"]},
        }
        col_candidates = _MARGIN_COLS.get(margin_type, _MARGIN_COLS["combined"])

        buy_col = None
        for c in col_candidates["buy"]:
            if c in mdf.columns:
                buy_col = c
                break
        sell_col = None
        for c in col_candidates["sell"]:
            if c in mdf.columns:
                sell_col = c
                break

        # 貸借倍率 = 信用買残 / 信用売残
        if buy_col and sell_col:
            mdf["margin_ratio"] = mdf[buy_col] / mdf[sell_col].replace(0, np.nan)
            logger.info(
                "貸借倍率算出 (%s): buy=%s, sell=%s, 有効行=%d, 平均=%.2f",
                margin_type, buy_col, sell_col,
                mdf["margin_ratio"].notna().sum(),
                mdf["margin_ratio"].mean() if mdf["margin_ratio"].notna().any() else 0,
            )
        else:
            logger.warning(
                "信用買残/売残カラムが見つかりません (%s)。"
                "期待: %s / 実際: %s",
                margin_type,
                col_candidates,
                [c for c in mdf.columns if c not in ("date", "code")],
            )
            mdf["margin_ratio"] = np.nan

        mdf = mdf[["date", "code", "margin_ratio"]].dropna(subset=["date", "code"])

        # 日次に展開 (ffill)
        prices_df = prices_df.sort_values(["code", "date"])
        prices_df = prices_df.merge(mdf, on=["date", "code"], how="left")
        prices_df["margin_ratio"] = prices_df.groupby("code")["margin_ratio"].ffill()

        n_filled = prices_df["margin_ratio"].notna().sum()
        logger.info("貸借倍率 ffill後: 有効=%d / %d行", n_filled, len(prices_df))

        return prices_df

    # ------------------------------------------------------------------
    # シグナル生成
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        prices_df: pd.DataFrame,
        config: SignalConfig,
        margin_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """エントリーシグナルを生成する

        Returns:
            DataFrame[date, code, weight]  — weight=1.0 でエントリー
        """
        close_col = "adj_close" if "adj_close" in prices_df.columns else "close"

        # 指標算出
        df = self.compute_indicators(prices_df)

        # 信用取引データ結合
        if config.needs_margin_data() and margin_df is not None:
            df = self.compute_margin_indicators(df, margin_df, margin_type=config.margin_type)

        # --- 個別条件マスク ---
        masks: list[tuple[str, pd.Series]] = []  # (名前, マスク) で診断用

        # 連続陽線
        if config.consecutive_bullish_days is not None:
            m = df["consec_bullish"] >= config.consecutive_bullish_days
            masks.append((f"連続陽線>={config.consecutive_bullish_days}", m))

        # 連続陰線
        if config.consecutive_bearish_days is not None:
            m = df["consec_bearish"] >= config.consecutive_bearish_days
            masks.append((f"連続陰線>={config.consecutive_bearish_days}", m))

        # 出来高倍率
        if config.volume_surge_ratio is not None:
            if config.volume_surge_window != 20:
                vol_col = "adj_volume" if "adj_volume" in df.columns else "volume"
                if vol_col in df.columns:
                    vol_ma = df.groupby("code")[vol_col].transform(
                        lambda s: s.rolling(config.volume_surge_window, min_periods=5).mean()
                    )
                    vol_ratio = df[vol_col] / vol_ma.replace(0, np.nan)
                else:
                    vol_ratio = df["volume_ratio"]
            else:
                vol_ratio = df["volume_ratio"]
            m = vol_ratio >= config.volume_surge_ratio
            masks.append((f"出来高>={config.volume_surge_ratio}x", m))

        # 終値 vs MA
        for window, attr in [(25, "price_vs_ma25"), (75, "price_vs_ma75"), (200, "price_vs_ma200")]:
            val = getattr(config, attr)
            if val is not None:
                col = f"ma{window}"
                if val == "above":
                    m = df[close_col] > df[col]
                else:
                    m = df[close_col] < df[col]
                masks.append((f"vs MA{window} {val}", m))

        # 移動平均乖離率
        if config.ma_deviation_pct is not None:
            ma_col = f"ma{config.ma_deviation_window}"
            if ma_col not in df.columns:
                df[ma_col] = df.groupby("code")[close_col].transform(
                    lambda s: s.rolling(config.ma_deviation_window, min_periods=config.ma_deviation_window).mean()
                )
            deviation = (df[close_col] - df[ma_col]) / df[ma_col] * 100
            m = deviation.abs() >= abs(config.ma_deviation_pct)
            masks.append((f"MA乖離>={config.ma_deviation_pct}%", m))

        # RSI
        if config.rsi_lower is not None or config.rsi_upper is not None:
            if config.rsi_window != 14:
                delta_r = df.groupby("code")[close_col].transform(lambda s: s.diff())
                g_r = delta_r.clip(lower=0)
                l_r = (-delta_r).clip(lower=0)
                avg_g = g_r.groupby(df["code"]).transform(
                    lambda s: s.ewm(span=config.rsi_window, min_periods=config.rsi_window).mean()
                )
                avg_l = l_r.groupby(df["code"]).transform(
                    lambda s: s.ewm(span=config.rsi_window, min_periods=config.rsi_window).mean()
                )
                rsi_custom = 100 - 100 / (1 + avg_g / avg_l)
                rsi_custom = rsi_custom.where(avg_l > 0, 100.0)
                rsi_custom = rsi_custom.where(avg_g > 0, other=rsi_custom)
            else:
                rsi_custom = df["rsi"]

            if config.rsi_lower is not None:
                m = rsi_custom < config.rsi_lower
                masks.append((f"RSI<{config.rsi_lower}", m))
            if config.rsi_upper is not None:
                m = rsi_custom > config.rsi_upper
                masks.append((f"RSI>{config.rsi_upper}", m))

        # ボリンジャーバンド
        if config.bb_buy_below_lower:
            if config.bb_window != 20 or config.bb_std != 2.0:
                bb_m = df.groupby("code")[close_col].transform(
                    lambda s: s.rolling(config.bb_window, min_periods=config.bb_window).mean()
                )
                bb_s = df.groupby("code")[close_col].transform(
                    lambda s: s.rolling(config.bb_window, min_periods=config.bb_window).std()
                )
                bb_low = bb_m - config.bb_std * bb_s
            else:
                bb_low = df["bb_lower"]
            m = df[close_col] <= bb_low
            masks.append(("BB下限タッチ", m))

        # ゴールデンクロス / デッドクロス
        if config.ma_cross_short is not None and config.ma_cross_long is not None:
            short_ma = df.groupby("code")[close_col].transform(
                lambda s: s.rolling(config.ma_cross_short, min_periods=config.ma_cross_short).mean()
            )
            long_ma = df.groupby("code")[close_col].transform(
                lambda s: s.rolling(config.ma_cross_long, min_periods=config.ma_cross_long).mean()
            )
            prev_short = short_ma.groupby(df["code"]).shift(1)
            prev_long = long_ma.groupby(df["code"]).shift(1)
            if config.ma_cross_type == "golden_cross":
                m = (prev_short <= prev_long) & (short_ma > long_ma)
                masks.append(("ゴールデンクロス", m))
            else:
                m = (prev_short >= prev_long) & (short_ma < long_ma)
                masks.append(("デッドクロス", m))

        # MACD
        if config.macd_fast is not None and config.macd_slow is not None:
            ema_f = df.groupby("code")[close_col].transform(
                lambda s: s.ewm(span=config.macd_fast, min_periods=config.macd_fast).mean()
            )
            ema_s = df.groupby("code")[close_col].transform(
                lambda s: s.ewm(span=config.macd_slow, min_periods=config.macd_slow).mean()
            )
            macd_l = ema_f - ema_s
            macd_sig = macd_l.groupby(df["code"]).transform(
                lambda s: s.ewm(span=config.macd_signal, min_periods=config.macd_signal).mean()
            )
            prev_macd = macd_l.groupby(df["code"]).shift(1)
            prev_sig = macd_sig.groupby(df["code"]).shift(1)
            m = (prev_macd <= prev_sig) & (macd_l > macd_sig)
            masks.append(("MACDクロス", m))

        # ATR フィルター
        if config.atr_max is not None:
            if config.atr_window != 14:
                h_col = "adj_high" if "adj_high" in df.columns else "high"
                l_col = "adj_low" if "adj_low" in df.columns else "low"
                prev_c = df.groupby("code")[close_col].shift(1)
                tr = pd.concat([
                    (df[h_col] - df[l_col]).abs(),
                    (df[h_col] - prev_c).abs(),
                    (df[l_col] - prev_c).abs(),
                ], axis=1).max(axis=1)
                atr_custom = tr.groupby(df["code"]).transform(
                    lambda s: s.rolling(config.atr_window, min_periods=config.atr_window).mean()
                )
            else:
                atr_custom = df["atr"]
            atr_pct = atr_custom / df[close_col] * 100
            m = atr_pct <= config.atr_max
            masks.append((f"ATR<={config.atr_max}%", m))

        # 一目均衡表: 雲の上/下
        if config.ichimoku_cloud is not None:
            if config.ichimoku_cloud == "above":
                m = df[close_col] > df["ichi_cloud_top"]
                masks.append(("雲の上", m))
            else:
                m = df[close_col] < df["ichi_cloud_bottom"]
                masks.append(("雲の下", m))

        # 一目均衡表: 転換線>基準線
        if config.ichimoku_tenkan_above_kijun:
            m = df["ichi_tenkan"] > df["ichi_kijun"]
            masks.append(("転換線>基準線", m))

        # セクター相対強度
        if config.sector_relative_strength_min is not None and "sector_17_name" in df.columns:
            lookback = config.sector_relative_lookback
            df["_ret_lb"] = df.groupby("code")[close_col].pct_change(lookback)
            df["_sector_rank_pct"] = df.groupby(["date", "sector_17_name"])["_ret_lb"].rank(pct=True) * 100
            m = df["_sector_rank_pct"] >= config.sector_relative_strength_min
            masks.append((f"セクター相対>={config.sector_relative_strength_min}%", m))

        # 信用取引: 貸借倍率
        if config.margin_ratio_min is not None and "margin_ratio" in df.columns:
            m = df["margin_ratio"] >= config.margin_ratio_min
            masks.append((f"貸借倍率>={config.margin_ratio_min}", m))
        if config.margin_ratio_max is not None and "margin_ratio" in df.columns:
            m = df["margin_ratio"] <= config.margin_ratio_max
            masks.append((f"貸借倍率<={config.margin_ratio_max}", m))

        # 空売り比率
        if config.short_selling_ratio_max is not None and "short_selling_ratio" in df.columns:
            m = df["short_selling_ratio"] <= config.short_selling_ratio_max
            masks.append((f"空売り比率<={config.short_selling_ratio_max}", m))

        # --- 条件結合 ---
        if not masks:
            logger.warning("有効なシグナル条件がありません")
            return pd.DataFrame(columns=["date", "code", "weight"])

        # 各条件の診断ログ
        for name, m in masks:
            # NaN を False として扱う
            true_count = m.fillna(False).sum()
            nan_count = m.isna().sum() if hasattr(m, 'isna') else 0
            logger.info("  条件 [%s]: True=%d, NaN=%d / %d", name, true_count, nan_count, len(m))

        # NaN は False として扱う
        bool_masks = [m.fillna(False) for _, m in masks]

        if config.signal_logic == "AND":
            combined = bool_masks[0]
            for bm in bool_masks[1:]:
                combined = combined & bm
        else:  # OR
            combined = bool_masks[0]
            for bm in bool_masks[1:]:
                combined = combined | bm

        n_true = combined.sum()
        logger.info("結合後シグナル (%s): %d / %d", config.signal_logic, n_true, len(combined))

        signals = df.loc[combined, ["date", "code"]].copy()
        signals["weight"] = 1.0

        return signals.reset_index(drop=True)
