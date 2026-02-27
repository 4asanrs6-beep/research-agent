"""スター株リバースエンジニアリング分析エンジン

直近1年で異常に上昇した日本株を逆引き分析し、共通特徴を定量的に抽出する。
変化点検出 / ファクター分解 / MLクラスタリング / VPIN / Lead-Lag / クロスセクショナル回帰を搭載。
"""

import json
import logging
import time as _time
from dataclasses import dataclass, field, asdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)

# 一般株式の市場区分（ETF・REIT等はこれ以外の市場区分名を持つ）
_STOCK_MARKET_SEGMENTS = {"プライム", "スタンダード", "グロース"}

# マルチシグナルOnset検出で使用する10個のシグナル名
_MULTI_ONSET_SIGNAL_NAMES = [
    "volume_surge", "quiet_accumulation", "consecutive_accumulation",
    "obv_breakout", "bb_squeeze", "volatility_compression",
    "higher_lows", "range_breakout", "ma_crossover", "up_volume_dominance",
]

# 26個のワイド特徴量キー（反復的特徴量発見で使用）
WIDE_FEATURE_KEYS = [
    # A: 出来高ダイナミクス (7)
    "vol_ratio_5d_20d", "vol_ratio_5d_60d", "vol_surge_count_10d",
    "up_volume_ratio_10d", "quiet_accum_rate_20d", "vol_acceleration", "vpin_5d",
    # B: 価格/リターン (6)
    "ret_5d", "ret_20d", "up_days_ratio_10d", "max_gap_up_5d",
    "higher_lows_slope_10d", "range_position_20d",
    # C: ボラティリティ・レジーム (4)
    "atr_ratio_5d_20d", "bb_width_pctile_60d", "intraday_range_ratio_5d",
    "realized_vol_5d_vs_20d",
    # D: トレンド/OBV (5)
    "obv_slope_10d", "obv_divergence", "ma5_ma20_gap", "price_vs_ma20_pct",
    "consecutive_up_days",
    # E: クロスセクショナル (4)
    "sector_rel_ret_10d", "topix_beta_20d", "residual_vol_ratio",
    "vol_vs_market_vol",
]

_WIDE_FEATURE_LABELS_JP = {
    "vol_ratio_5d_20d": "出来高比(5/20日)",
    "vol_ratio_5d_60d": "出来高比(5/60日)",
    "vol_surge_count_10d": "出来高急増回数(10日)",
    "up_volume_ratio_10d": "上昇日出来高比(10日)",
    "quiet_accum_rate_20d": "静的蓄積率(20日)",
    "vol_acceleration": "出来高加速度",
    "vpin_5d": "VPIN(5日)",
    "ret_5d": "リターン(5日)",
    "ret_20d": "リターン(20日)",
    "up_days_ratio_10d": "上昇日比率(10日)",
    "max_gap_up_5d": "最大ギャップアップ(5日)",
    "higher_lows_slope_10d": "安値切上り傾き(10日)",
    "range_position_20d": "レンジ位置(20日)",
    "atr_ratio_5d_20d": "ATR比(5/20日)",
    "bb_width_pctile_60d": "BB幅パーセンタイル(60日)",
    "intraday_range_ratio_5d": "日中値幅比(5/20日)",
    "realized_vol_5d_vs_20d": "実現Vol比(5/20日)",
    "obv_slope_10d": "OBV傾き(10日)",
    "obv_divergence": "OBVダイバージェンス",
    "ma5_ma20_gap": "MA(5-20)乖離率",
    "price_vs_ma20_pct": "対MA20乖離率",
    "consecutive_up_days": "連続上昇日数",
    "sector_rel_ret_10d": "セクター相対リターン(10日)",
    "topix_beta_20d": "TOPIXβ(20日)",
    "residual_vol_ratio": "残差Vol比(10/60日)",
    "vol_vs_market_vol": "対市場出来高比",
}

# シグナル名の短縮表示用マッピング
_ONSET_SIGNAL_SHORT = {
    "volume_surge": "Vol↑",
    "quiet_accumulation": "静蓄",
    "consecutive_accumulation": "連蓄",
    "obv_breakout": "OBV",
    "bb_squeeze": "BB↓",
    "volatility_compression": "Vol圧",
    "higher_lows": "HiLo",
    "range_breakout": "Range↑",
    "ma_crossover": "MA×",
    "up_volume_dominance": "Up出来",
}

# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------
@dataclass
class StarStockConfig:
    """スター株分析の設定"""
    # 検出閾値
    min_total_return: float = 0.50
    min_excess_return: float = 0.30
    min_volume_increase_ratio: float = 1.5
    auto_detect_enabled: bool = True
    max_auto_detect: int = 50
    user_codes: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    # 仕手株フィルター
    min_market_cap_billion: float = 50.0
    max_drawdown_from_peak: float = 0.40
    max_single_day_return: float = 0.20
    min_up_days_ratio: float = 0.45
    require_positive_end: bool = True
    # 高度分析
    rolling_beta_window: int = 60
    volume_surge_threshold: float = 2.0
    accumulation_price_threshold: float = 0.005
    accumulation_volume_threshold: float = 1.5
    obv_trend_window: int = 60
    sector_correlation_window: int = 60
    factor_window: int = 60
    vpin_bucket_size: int = 20
    lead_lag_max_lag: int = 10
    n_clusters: int = 4
    # 反復的特徴量発見パラメータ
    discovery_max_iterations: int = 5
    discovery_target_precision: float = 0.20
    discovery_min_recall: float = 0.30
    discovery_neg_sample_size: int = 200
    onset_min_forward_return: float = 0.10
    onset_max_candidates: int = 5


@dataclass
class StarStockResult:
    """スター株分析の結果"""
    config: dict
    star_stocks: list[dict]
    topix_return: float
    n_auto_detected: int
    n_user_specified: int
    # AI生成
    common_features_summary: str = ""
    pattern_typology: list[dict] = field(default_factory=list)
    foreign_flow_assessment: str = ""
    detection_rules: str = ""
    # 高度分析結果
    factor_analysis: dict = field(default_factory=dict)
    cluster_analysis: dict = field(default_factory=dict)
    cross_sectional: dict = field(default_factory=dict)
    lead_lag_analysis: dict = field(default_factory=dict)
    # シグナル検証（Precision/Recall）
    signal_validation: dict = field(default_factory=dict)
    # 買いタイミング近接度
    timing_candidates: list[dict] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# メインエンジン
# ---------------------------------------------------------------------------
class StarStockAnalyzer:
    """スター株分析エンジン"""

    def __init__(self, data_provider, ai_client=None):
        self.provider = data_provider
        self.ai_client = ai_client

    def run_analysis(
        self,
        config: StarStockConfig,
        on_progress=None,
    ) -> StarStockResult:
        """分析パイプライン全体を実行する"""

        def _prog(msg: str, pct: float):
            if on_progress:
                on_progress(msg, pct)

        try:
            # Step 1: データ取得
            _prog("銘柄メタデータを取得中...", 0.02)
            listed_stocks = self._api_call_with_retry(
                lambda: self.provider.get_listed_stocks()
            )

            _prog("TOPIX指数を取得中...", 0.05)
            topix = self._api_call_with_retry(
                lambda: self.provider.get_index_prices(
                    "0000", config.start_date, config.end_date,
                )
            )
            topix_return = self._calc_index_return(topix)

            _prog("全銘柄株価を取得中（時間がかかります）...", 0.08)
            all_prices = self._fetch_all_prices_chunked(
                config.start_date, config.end_date, _prog,
            )

            # Step 2: スター株検出 + 仕手株フィルタ
            _prog("スター株を検出中...", 0.20)
            star_stocks, n_auto, n_user = self._detect_star_stocks(
                all_prices, topix, listed_stocks, config,
            )

            if not star_stocks:
                return StarStockResult(
                    config=asdict(config),
                    star_stocks=[],
                    topix_return=topix_return,
                    n_auto_detected=0,
                    n_user_specified=0,
                    error="条件に合致するスター株が見つかりませんでした。閾値を下げてください。",
                )

            # Step 3: 信用残データ取得（スター株のみ）
            _prog(f"信用残データを取得中（{len(star_stocks)}銘柄）...", 0.30)
            margin_data = self._fetch_margin_data(star_stocks, config)

            # Step 4: 基本特徴量計算
            _prog("基本特徴量を計算中...", 0.35)
            star_stocks = self._compute_basic_features(
                star_stocks, all_prices, topix, listed_stocks, margin_data, config,
            )

            # Step 5: 変化点検出（CUSUM）
            _prog("変化点を検出中（CUSUM）...", 0.42)
            star_stocks = self._apply_change_point_detection(star_stocks, all_prices, config)

            # Step 6: ファクター分解
            _prog("ファクター分解中（4ファクターモデル）...", 0.50)
            factor_analysis = self._run_factor_analysis(
                star_stocks, all_prices, listed_stocks, topix, config,
            )

            # Step 7: VPIN
            _prog("VPIN（情報トレーダー参加率）を計算中...", 0.58)
            star_stocks = self._apply_vpin(star_stocks, all_prices, config)

            # Step 8: 海外フロープロキシ + 合成スコア
            _prog("海外フロースコアを算出中...", 0.63)
            star_stocks = self._compute_flow_scores(
                star_stocks, all_prices, topix, listed_stocks, margin_data, config,
            )

            # Step 9: Lead-Lag分析
            _prog("Lead-Lag分析中（Granger因果）...", 0.70)
            lead_lag = self._lead_lag_analysis(star_stocks, all_prices, config)

            # Step 10: MLクラスタリング
            _prog("MLクラスタリング中（K-means + PCA）...", 0.78)
            cluster = self._cluster_analysis(star_stocks, config)

            # Step 11: クロスセクショナル回帰
            _prog("クロスセクショナル回帰中（Fama-MacBeth）...", 0.83)
            cross_sect = self._cross_sectional_regression(
                all_prices, listed_stocks, topix, margin_data, config,
            )

            # Step 12: 反復的特徴量発見（Iterative Discriminative Feature Discovery）
            _prog("反復的特徴量探索中（26特徴量×コンボ探索）...", 0.85)
            signal_validation = self._discover_discriminative_features(
                star_stocks, all_prices, listed_stocks, topix, config,
            )

            # Step 13: AI要約生成
            _prog("AI要約を生成中...", 0.90)
            ai_results = self._generate_ai_summary(
                star_stocks, topix_return, factor_analysis, cluster,
                lead_lag, cross_sect, config, signal_validation,
            )

            # Step 14: 買いタイミング近接度スキャン
            _prog("買いタイミング候補を探索中...", 0.95)
            timing_candidates = self._scan_timing_candidates(
                star_stocks, all_prices, listed_stocks, topix, config,
                signal_validation,
            )

            _prog("完了", 1.0)

            return StarStockResult(
                config=asdict(config),
                star_stocks=star_stocks,
                topix_return=topix_return,
                n_auto_detected=n_auto,
                n_user_specified=n_user,
                common_features_summary=ai_results.get("common_features_summary", ""),
                pattern_typology=ai_results.get("pattern_typology", []),
                foreign_flow_assessment=ai_results.get("foreign_flow_assessment", ""),
                detection_rules=ai_results.get("detection_rules", ""),
                factor_analysis=factor_analysis,
                cluster_analysis=cluster,
                cross_sectional=cross_sect,
                lead_lag_analysis=lead_lag,
                signal_validation=signal_validation,
                timing_candidates=timing_candidates,
            )

        except Exception as e:
            logger.exception("スター株分析エラー")
            return StarStockResult(
                config=asdict(config),
                star_stocks=[],
                topix_return=0.0,
                n_auto_detected=0,
                n_user_specified=0,
                error=str(e),
            )

    # ===================================================================
    # Step 1: ユーティリティ + レート制限対策
    # ===================================================================
    @staticmethod
    def _calc_index_return(topix: pd.DataFrame) -> float:
        if topix.empty:
            return 0.0
        topix = topix.sort_values("date")
        first = topix["close"].iloc[0]
        last = topix["close"].iloc[-1]
        return (last / first - 1) if first > 0 else 0.0

    @staticmethod
    def _api_call_with_retry(fn, max_retries: int = 5):
        """429エラー時にウェイト+リトライするラッパー"""
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
        """全銘柄株価を月単位チャンクで取得（429対策）

        get_price_daily(code=None) は内部で日付ごとに並列リクエストを飛ばし
        429を大量発生させるため、月単位で分割して逐次取得する。
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # 月単位のチャンクに分割
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
                pct = 0.08 + (0.12 * (i / total))
                _prog(f"株価取得中... ({i+1}/{total}チャンク)", pct)

            df = self._api_call_with_retry(
                lambda s=c_start, e=c_end: self.provider.get_price_daily(
                    code=None, start_date=s, end_date=e,
                )
            )
            if df is not None and not df.empty:
                frames.append(df)

            # チャンク間のウェイト（429予防）
            if i < total - 1:
                _time.sleep(2.0)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ===================================================================
    # Step 2: スター株検出 + 仕手株フィルタ
    # ===================================================================
    def _detect_star_stocks(
        self,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        config: StarStockConfig,
    ) -> tuple[list[dict], int, int]:
        """全銘柄からスター株を検出する"""

        topix_ret = self._calc_index_return(topix)

        # 銘柄ごとの集計
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"
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

            # 日次リターン
            daily_rets = grp[close_col].pct_change(fill_method=None).dropna()

            # 最大下落率（高値→終値）
            peak = grp[close_col].cummax()
            drawdown_from_peak = ((peak - grp[close_col]) / peak).max()

            # 最大1日変動
            max_daily = daily_rets.abs().max() if len(daily_rets) > 0 else 0

            # 上昇日比率
            up_days_ratio = (daily_rets > 0).mean() if len(daily_rets) > 0 else 0

            stock_stats.append({
                "code": str(code),
                "total_return": float(total_return),
                "excess_return": float(excess_return),
                "volume_change_ratio": float(vol_ratio),
                "max_drawdown": float(drawdown_from_peak),
                "max_single_day_return": float(max_daily),
                "up_days_ratio": float(up_days_ratio),
                "first_price": float(first_price),
                "last_price": float(last_price),
                "n_days": len(grp),
            })

        # メタデータ結合
        meta_cols = ["code", "name", "sector_17_name", "market_name", "scale_category"]
        available_meta = [c for c in meta_cols if c in listed_stocks.columns]
        meta_map = {}
        if available_meta:
            for _, row in listed_stocks[available_meta].iterrows():
                meta_map[str(row["code"])] = {c: row.get(c, "") for c in available_meta}

        # 時価総額情報取得（scale_categoryで代替）
        scale_cap_map = {
            "TOPIX Core30": 5000.0,
            "TOPIX Large70": 2000.0,
            "TOPIX Mid400": 500.0,
            "TOPIX Small 1": 100.0,
            "TOPIX Small 2": 50.0,
        }

        user_set = set(config.user_codes)
        auto_detected = []
        user_specified = []

        for s in stock_stats:
            code = s["code"]
            meta = meta_map.get(code, {})
            s.update({
                "name": meta.get("name", ""),
                "sector": meta.get("sector_17_name", ""),
                "market": meta.get("market_name", ""),
                "scale_category": meta.get("scale_category", ""),
            })

            # 推定時価総額（scale_categoryから大まかに推定）
            est_cap = scale_cap_map.get(s["scale_category"], 30.0)
            s["est_market_cap_billion"] = est_cap

            is_user = code in user_set

            # 自動検出チェック
            if config.auto_detect_enabled and not is_user:
                if (s["total_return"] >= config.min_total_return
                        and s["excess_return"] >= config.min_excess_return
                        and s["volume_change_ratio"] >= config.min_volume_increase_ratio):
                    # 仕手株フィルター
                    pump_flags = self._check_pump_dump(s, config)
                    if not pump_flags:
                        s["source"] = "auto"
                        s["pump_dump_flags"] = []
                        auto_detected.append(s)
                    # else: filtered out

            if is_user:
                pump_flags = self._check_pump_dump(s, config)
                s["source"] = "user"
                s["pump_dump_flags"] = pump_flags
                user_specified.append(s)

        # 自動検出を超過リターン降順でソートし上限適用
        auto_detected.sort(key=lambda x: x["excess_return"], reverse=True)
        auto_detected = auto_detected[:config.max_auto_detect]

        # マージ（重複排除）
        result_codes = set()
        result = []
        for s in user_specified:
            result_codes.add(s["code"])
            result.append(s)
        for s in auto_detected:
            if s["code"] not in result_codes:
                result_codes.add(s["code"])
                result.append(s)

        return result, len(auto_detected), len(user_specified)

    @staticmethod
    def _check_pump_dump(s: dict, config: StarStockConfig) -> list[str]:
        """仕手株フラグをチェック"""
        flags = []
        if s["est_market_cap_billion"] < config.min_market_cap_billion:
            flags.append(f"時価総額小({s['est_market_cap_billion']:.0f}億円)")
        if s["max_drawdown"] > config.max_drawdown_from_peak:
            flags.append(f"高値からの下落率({s['max_drawdown']:.1%})")
        if s["max_single_day_return"] > config.max_single_day_return:
            flags.append(f"1日最大変動({s['max_single_day_return']:.1%})")
        if s["up_days_ratio"] < config.min_up_days_ratio:
            flags.append(f"上昇日比率低({s['up_days_ratio']:.1%})")
        if config.require_positive_end and s["last_price"] < s["first_price"]:
            flags.append("期末が開始値以下")
        return flags

    # ===================================================================
    # Step 3: 信用残データ取得
    # ===================================================================
    def _fetch_margin_data(
        self,
        star_stocks: list[dict],
        config: StarStockConfig,
    ) -> dict[str, pd.DataFrame]:
        """各スター株の信用残データを取得（429リトライ付き）"""
        margin_data = {}
        for i, s in enumerate(star_stocks):
            try:
                df = self._api_call_with_retry(
                    lambda code=s["code"]: self.provider.get_margin_trading(
                        code=code,
                        start_date=config.start_date,
                        end_date=config.end_date,
                    ),
                    max_retries=3,
                )
                if df is not None and not df.empty:
                    margin_data[s["code"]] = df
            except Exception as e:
                logger.warning("信用残データ取得失敗 %s: %s", s["code"], e)
            # 連続リクエスト間の429予防ウェイト
            if i % 5 == 4:
                _time.sleep(1.0)
        return margin_data

    # ===================================================================
    # Step 4: 基本特徴量計算
    # ===================================================================
    def _compute_basic_features(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        margin_data: dict,
        config: StarStockConfig,
    ) -> list[dict]:
        """各スター株に基本特徴量を追加"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        for s in star_stocks:
            code = s["code"]
            grp = all_prices[all_prices["code"] == code].sort_values("date").copy()
            if len(grp) < 20:
                continue

            prices = grp[close_col].values
            volume = grp["volume"].values
            returns = pd.Series(prices).pct_change(fill_method=None).dropna().values

            # 加速度（後半リターン / 前半リターン）
            n_half = len(returns) // 2
            first_half_ret = np.sum(returns[:n_half])
            second_half_ret = np.sum(returns[n_half:])
            s["acceleration"] = float(second_half_ret / first_half_ret) if first_half_ret != 0 else 0.0

            # 出来高急増日数
            vol_series = pd.Series(volume)
            vol_ma = vol_series.rolling(20, min_periods=1).mean()
            surge_count = int((vol_series > vol_ma * config.volume_surge_threshold).sum())
            s["volume_surge_count"] = surge_count

            # 実現ボラティリティ変化
            n_q = len(returns) // 4
            if n_q > 5:
                vol_early = np.std(returns[:n_q])
                vol_late = np.std(returns[-n_q:])
                s["realized_vol_change"] = float(vol_late / vol_early - 1) if vol_early > 0 else 0.0
            else:
                s["realized_vol_change"] = 0.0

            # シャープ比
            if len(returns) > 1 and np.std(returns) > 0:
                s["sharpe"] = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            else:
                s["sharpe"] = 0.0

        return star_stocks

    # ===================================================================
    # Step 4b: ワイド特徴量計算 (26特徴量)
    # ===================================================================
    @staticmethod
    def _compute_wide_features(
        df: pd.DataFrame,
        close_col: str,
        sector_ret_10d: float | None = None,
        market_vol_ratio: float | None = None,
        topix_ret_series: pd.Series | None = None,
    ) -> dict | None:
        """1銘柄の直近データから26個のワイド特徴量を計算する。

        Parameters
        ----------
        df : pd.DataFrame
            1銘柄分の株価データ（dateソート済み）。最低60行推奨。
        close_col : str
            終値カラム名
        sector_ret_10d : float | None
            セクター平均の10日リターン（クロスセクショナル用）
        market_vol_ratio : float | None
            市場全体の vol5/vol60 比率（クロスセクショナル用）
        topix_ret_series : pd.Series | None
            TOPIX日次リターン（β計算用）

        Returns
        -------
        dict | None  26特徴量のdict。データ不足時はNone。
        """
        if len(df) < 20:
            return None

        close = df[close_col].astype(float).values
        volume = df["volume"].astype(float).values
        high = df["high"].astype(float).values if "high" in df.columns else close
        low = df["low"].astype(float).values if "low" in df.columns else close
        open_ = df["open"].astype(float).values if "open" in df.columns else close

        n = len(close)
        ret = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)
        ret = np.where(np.isfinite(ret), ret, 0.0)

        # 安全なrolling mean (numpy)
        def _ma(arr, w):
            if len(arr) < w:
                return np.full(len(arr), np.nanmean(arr))
            cs = np.cumsum(arr)
            cs = np.insert(cs, 0, 0.0)
            out = np.full(len(arr), np.nan)
            out[w - 1:] = (cs[w:] - cs[:-w]) / w
            # fill前半
            for i in range(w - 1):
                out[i] = np.mean(arr[: i + 1])
            return out

        vol_ma5 = _ma(volume, 5)
        vol_ma20 = _ma(volume, 20)
        vol_ma60 = _ma(volume, 60) if n >= 60 else _ma(volume, max(n, 1))
        close_ma5 = _ma(close, 5)
        close_ma20 = _ma(close, 20)

        feat = {}

        # --- A: 出来高ダイナミクス ---
        # 1. vol_ratio_5d_20d
        feat["vol_ratio_5d_20d"] = float(vol_ma5[-1] / vol_ma20[-1]) if vol_ma20[-1] > 0 else 1.0
        # 2. vol_ratio_5d_60d
        feat["vol_ratio_5d_60d"] = float(vol_ma5[-1] / vol_ma60[-1]) if vol_ma60[-1] > 0 else 1.0
        # 3. vol_surge_count_10d
        window_10 = min(10, n)
        feat["vol_surge_count_10d"] = int(np.sum(volume[-window_10:] > vol_ma20[-window_10:] * 2.0))
        # 4. up_volume_ratio_10d
        if len(ret) >= 10:
            up_mask = ret[-10:] > 0
            up_vol = volume[-10:][:-1][up_mask[:-1]].sum() if len(up_mask) > 1 else 0
            # use last 10 volumes aligned with returns
            vol_10 = volume[-10:]
            ret_10 = ret[-(min(10, len(ret))):]
            r_len = min(len(vol_10) - 1, len(ret_10))
            if r_len > 0:
                up_m = ret_10[-r_len:] > 0
                up_v = vol_10[-r_len:][up_m].sum()
                total_v = vol_10[-r_len:].sum()
                feat["up_volume_ratio_10d"] = float(up_v / total_v) if total_v > 0 else 0.5
            else:
                feat["up_volume_ratio_10d"] = 0.5
        else:
            feat["up_volume_ratio_10d"] = 0.5
        # 5. quiet_accum_rate_20d
        w20 = min(20, len(ret))
        if w20 > 0:
            r20 = ret[-w20:]
            v20 = volume[-w20:]
            vm20 = vol_ma20[-w20:]
            quiet_mask = (np.abs(r20) < 0.003) & (v20[: len(r20)] > vm20[: len(r20)] * 1.3)
            feat["quiet_accum_rate_20d"] = float(quiet_mask.sum() / w20)
        else:
            feat["quiet_accum_rate_20d"] = 0.0
        # 6. vol_acceleration
        if n >= 10:
            first_half = vol_ma5[-10:-5].mean() if len(vol_ma5) >= 10 else vol_ma5.mean()
            second_half = vol_ma5[-5:].mean()
            feat["vol_acceleration"] = float(second_half / first_half) if first_half > 0 else 1.0
        else:
            feat["vol_acceleration"] = 1.0
        # 7. vpin_5d (簡易計算)
        if len(ret) >= 20:
            ret_s = pd.Series(ret)
            rolling_std = ret_s.rolling(20, min_periods=5).std().values
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(rolling_std > 0, ret / rolling_std, 0)
            buy_pct = norm.cdf(z)
            bv = volume[1: len(ret) + 1] * buy_pct
            sv_ = volume[1: len(ret) + 1] * (1 - buy_pct)
            tv = volume[1: len(ret) + 1]
            w5 = min(5, len(bv))
            bv_sum = bv[-w5:].sum()
            sv_sum = sv_[-w5:].sum()
            tv_sum = tv[-w5:].sum()
            feat["vpin_5d"] = float(abs(bv_sum - sv_sum) / tv_sum) if tv_sum > 0 else 0.0
        else:
            feat["vpin_5d"] = 0.0

        # --- B: 価格/リターン ---
        # 8. ret_5d
        if n >= 6:
            feat["ret_5d"] = float(close[-1] / close[-6] - 1) if close[-6] > 0 else 0.0
        else:
            feat["ret_5d"] = 0.0
        # 9. ret_20d
        if n >= 21:
            feat["ret_20d"] = float(close[-1] / close[-21] - 1) if close[-21] > 0 else 0.0
        else:
            feat["ret_20d"] = float(close[-1] / close[0] - 1) if close[0] > 0 else 0.0
        # 10. up_days_ratio_10d
        w10r = min(10, len(ret))
        feat["up_days_ratio_10d"] = float((ret[-w10r:] > 0).mean()) if w10r > 0 else 0.5
        # 11. max_gap_up_5d
        if n >= 2:
            w5g = min(5, n - 1)
            gaps = open_[-w5g:] / close[-w5g - 1: -1] - 1
            gaps = np.where(np.isfinite(gaps), gaps, 0.0)
            feat["max_gap_up_5d"] = float(np.max(gaps)) if len(gaps) > 0 else 0.0
        else:
            feat["max_gap_up_5d"] = 0.0
        # 12. higher_lows_slope_10d
        w10l = min(10, n)
        if w10l >= 5:
            lows_w = low[-w10l:]
            x = np.arange(len(lows_w), dtype=float)
            try:
                slope = stats.linregress(x, lows_w).slope
                feat["higher_lows_slope_10d"] = float(slope / np.mean(close[-w10l:])) if np.mean(close[-w10l:]) > 0 else 0.0
            except Exception:
                feat["higher_lows_slope_10d"] = 0.0
        else:
            feat["higher_lows_slope_10d"] = 0.0
        # 13. range_position_20d
        w20p = min(20, n)
        low_min = np.min(low[-w20p:])
        high_max = np.max(high[-w20p:])
        rng = high_max - low_min
        feat["range_position_20d"] = float((close[-1] - low_min) / rng) if rng > 0 else 0.5

        # --- C: ボラティリティ・レジーム ---
        # 14. atr_ratio_5d_20d
        tr_vals = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        if len(tr_vals) >= 20:
            atr5 = np.mean(tr_vals[-5:])
            atr20 = np.mean(tr_vals[-20:])
            feat["atr_ratio_5d_20d"] = float(atr5 / atr20) if atr20 > 0 else 1.0
        else:
            feat["atr_ratio_5d_20d"] = 1.0
        # 15. bb_width_pctile_60d
        if n >= 20:
            bb_std = pd.Series(close).rolling(20, min_periods=10).std().values
            bb_ma = _ma(close, 20)
            with np.errstate(divide="ignore", invalid="ignore"):
                bb_width = np.where(bb_ma > 0, 2 * bb_std / bb_ma, 0.0)
            bb_width = np.where(np.isfinite(bb_width), bb_width, 0.0)
            valid_bw = bb_width[~np.isnan(bb_width)]
            if len(valid_bw) >= 10:
                current_bw = valid_bw[-1]
                feat["bb_width_pctile_60d"] = float(np.searchsorted(np.sort(valid_bw[-60:]), current_bw) / len(valid_bw[-60:]))
            else:
                feat["bb_width_pctile_60d"] = 0.5
        else:
            feat["bb_width_pctile_60d"] = 0.5
        # 16. intraday_range_ratio_5d
        if n >= 20:
            intra = (high - low) / np.where(close > 0, close, 1.0)
            intra = np.where(np.isfinite(intra), intra, 0.0)
            feat["intraday_range_ratio_5d"] = float(np.mean(intra[-5:]) / np.mean(intra[-20:])) if np.mean(intra[-20:]) > 0 else 1.0
        else:
            feat["intraday_range_ratio_5d"] = 1.0
        # 17. realized_vol_5d_vs_20d
        if len(ret) >= 20:
            rv5 = np.std(ret[-5:])
            rv20 = np.std(ret[-20:])
            feat["realized_vol_5d_vs_20d"] = float(rv5 / rv20) if rv20 > 0 else 1.0
        else:
            feat["realized_vol_5d_vs_20d"] = 1.0

        # --- D: トレンド/OBV ---
        # 18. obv_slope_10d
        signed_vol = np.sign(ret) * volume[1: len(ret) + 1]
        obv = np.cumsum(signed_vol)
        w10o = min(10, len(obv))
        if w10o >= 5:
            x = np.arange(w10o, dtype=float)
            try:
                slope = stats.linregress(x, obv[-w10o:]).slope
                avg_vol = np.mean(volume[-w10o:])
                feat["obv_slope_10d"] = float(slope / avg_vol) if avg_vol > 0 else 0.0
            except Exception:
                feat["obv_slope_10d"] = 0.0
        else:
            feat["obv_slope_10d"] = 0.0
        # 19. obv_divergence
        if len(obv) >= 20:
            try:
                corr, _ = stats.spearmanr(close[-20:], obv[-20:])
                feat["obv_divergence"] = float(corr) if np.isfinite(corr) else 0.0
            except Exception:
                feat["obv_divergence"] = 0.0
        else:
            feat["obv_divergence"] = 0.0
        # 20. ma5_ma20_gap
        feat["ma5_ma20_gap"] = float((close_ma5[-1] - close_ma20[-1]) / close_ma20[-1]) if close_ma20[-1] > 0 else 0.0
        # 21. price_vs_ma20_pct
        feat["price_vs_ma20_pct"] = float(close[-1] / close_ma20[-1] - 1) if close_ma20[-1] > 0 else 0.0
        # 22. consecutive_up_days
        consec = 0
        for i in range(len(ret) - 1, -1, -1):
            if ret[i] > 0:
                consec += 1
            else:
                break
        feat["consecutive_up_days"] = min(consec, 20)

        # --- E: クロスセクショナル ---
        # 23. sector_rel_ret_10d — データなし時はret_10dと同値（中立）
        stock_ret_10d = float(close[-1] / close[-min(11, n)] - 1) if close[-min(11, n)] > 0 else 0.0
        if sector_ret_10d is not None:
            feat["sector_rel_ret_10d"] = stock_ret_10d - sector_ret_10d
        else:
            feat["sector_rel_ret_10d"] = stock_ret_10d  # セクター情報なし→自身のret
        # 24. topix_beta_20d — 日付型を明示的にTimestampに統一
        if topix_ret_series is not None and len(ret) >= 20:
            try:
                if "date" in df.columns:
                    dates_ts = pd.to_datetime(df["date"].values)
                    # retはdiff(close)なのでlen=n-1、dates[1:]と対応
                    stock_s = pd.Series(ret, index=dates_ts[1:])
                    # topix_ret_seriesのインデックスもTimestampに統一
                    topix_idx = pd.to_datetime(topix_ret_series.index)
                    topix_aligned = pd.Series(topix_ret_series.values, index=topix_idx)
                    common = stock_s.index.intersection(topix_aligned.index)
                    if len(common) >= 15:
                        sr = stock_s.loc[common].values[-20:]
                        tr = topix_aligned.loc[common].values[-20:]
                        if len(sr) >= 10 and len(tr) >= 10:
                            cov = np.cov(sr, tr)
                            feat["topix_beta_20d"] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
                        else:
                            feat["topix_beta_20d"] = 1.0
                    else:
                        feat["topix_beta_20d"] = 1.0
                else:
                    feat["topix_beta_20d"] = 1.0
            except Exception:
                feat["topix_beta_20d"] = 1.0
        else:
            feat["topix_beta_20d"] = 1.0
        # 25. residual_vol_ratio — TOPIX残差ベースのvol比率
        if topix_ret_series is not None and len(ret) >= 60 and "date" in df.columns:
            try:
                dates_ts = pd.to_datetime(df["date"].values)
                stock_s = pd.Series(ret, index=dates_ts[1:])
                topix_idx = pd.to_datetime(topix_ret_series.index)
                topix_aligned = pd.Series(topix_ret_series.values, index=topix_idx)
                common = stock_s.index.intersection(topix_aligned.index)
                if len(common) >= 60:
                    resid = stock_s.loc[common].values - topix_aligned.loc[common].values * feat.get("topix_beta_20d", 1.0)
                    rv10 = np.std(resid[-10:])
                    rv60 = np.std(resid[-60:])
                    feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
                else:
                    # フォールバック: 生のリターンで計算
                    rv10 = np.std(ret[-10:])
                    rv60 = np.std(ret[-60:])
                    feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
            except Exception:
                rv10 = np.std(ret[-10:])
                rv60 = np.std(ret[-60:])
                feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
        elif len(ret) >= 60:
            rv10 = np.std(ret[-10:])
            rv60 = np.std(ret[-60:])
            feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
        else:
            feat["residual_vol_ratio"] = 1.0
        # 26. vol_vs_market_vol
        stock_vol_ratio = feat["vol_ratio_5d_60d"]
        if market_vol_ratio is not None and market_vol_ratio > 0:
            feat["vol_vs_market_vol"] = float(stock_vol_ratio / market_vol_ratio)
        else:
            feat["vol_vs_market_vol"] = stock_vol_ratio

        # NaN/Inf安全化
        for k in feat:
            v = feat[k]
            if not np.isfinite(v):
                feat[k] = 0.0

        return feat

    # ===================================================================
    # Step 5: マルチシグナルOnset検出 + 変化点検出 (CUSUM)
    # ===================================================================

    @staticmethod
    def _compute_daily_onset_signals(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        """1銘柄の株価DataFrameから日次の10個のOnsetシグナル(bool)を計算する。

        Parameters
        ----------
        df : pd.DataFrame
            1銘柄分の株価。date/close(adj_close)/high/low/volume列を想定。
            dateソート済みであること。
        close_col : str
            終値カラム名。

        Returns
        -------
        pd.DataFrame
            index=date, columns=_MULTI_ONSET_SIGNAL_NAMES, values=bool
        """
        if len(df) < 60:
            return pd.DataFrame()

        out = pd.DataFrame(index=df.index)
        close = df[close_col].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close
        volume = df["volume"].astype(float)
        ret = close.pct_change()

        vol_ma20 = volume.rolling(20, min_periods=10).mean()

        # 1. volume_surge: volume > 20日MA × 2.5
        out["volume_surge"] = volume > (vol_ma20 * 2.5)

        # 2. quiet_accumulation: |日次リターン| < 0.5% AND volume > 20日MA × 1.3
        out["quiet_accumulation"] = (ret.abs() < 0.005) & (volume > vol_ma20 * 1.3)

        # 3. consecutive_accumulation: 直近5日で4日以上がquiet_accumulation
        qa_roll = out["quiet_accumulation"].astype(int).rolling(5, min_periods=5).sum()
        out["consecutive_accumulation"] = qa_roll >= 4

        # 4. obv_breakout: OBV(20日) > OBV(60日) AND OBVの10日変化率 > 2σ
        signed_vol = volume * ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = signed_vol.cumsum()
        obv_ma20 = obv.rolling(20, min_periods=10).mean()
        obv_ma60 = obv.rolling(60, min_periods=30).mean()
        obv_change10 = obv.pct_change(10)
        obv_change_std = obv_change10.rolling(60, min_periods=20).std()
        obv_change_mean = obv_change10.rolling(60, min_periods=20).mean()
        obv_z = (obv_change10 - obv_change_mean) / obv_change_std.replace(0, np.nan)
        out["obv_breakout"] = (obv_ma20 > obv_ma60) & (obv_z > 2.0)

        # 5. bb_squeeze: BB幅(20日) < 20日移動平均(BB幅)の60%
        ma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        bb_width = (2 * std20) / ma20.replace(0, np.nan)
        bb_width_ma = bb_width.rolling(20, min_periods=10).mean()
        out["bb_squeeze"] = bb_width < (bb_width_ma * 0.6)

        # 6. volatility_compression: 10日ATR < 60日ATR × 0.6
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr10 = tr.rolling(10, min_periods=5).mean()
        atr60 = tr.rolling(60, min_periods=30).mean()
        out["volatility_compression"] = atr10 < (atr60 * 0.6)

        # 7. higher_lows: 直近5安値の回帰傾き > 0 AND R² > 0.6
        def _higher_lows_check(s):
            """ローリング5日安値に対する線形回帰"""
            vals = s.values
            if np.isnan(vals).any():
                return False
            x = np.arange(len(vals), dtype=float)
            slope, intercept, r, p, se = stats.linregress(x, vals)
            return (slope > 0) and (r ** 2 > 0.6)

        hl_series = low.rolling(5, min_periods=5).apply(
            _higher_lows_check, raw=False,
        )
        out["higher_lows"] = hl_series.fillna(0).astype(bool)

        # 8. range_breakout: 終値 > 過去40日高値
        high40 = high.rolling(40, min_periods=20).max().shift(1)
        out["range_breakout"] = close > high40

        # 9. ma_crossover: 5日MA > 20日MA AND 前日は5日MA ≤ 20日MA
        ma5 = close.rolling(5, min_periods=3).mean()
        cross_today = ma5 > ma20
        cross_yesterday = ma5.shift(1) <= ma20.shift(1)
        out["ma_crossover"] = cross_today & cross_yesterday

        # 10. up_volume_dominance: 上昇日出来高(10日sum)/下落日出来高(10日sum) > 2.0
        up_vol = volume.where(ret > 0, 0.0)
        down_vol = volume.where(ret < 0, 0.0)
        up_vol_sum10 = up_vol.rolling(10, min_periods=5).sum()
        down_vol_sum10 = down_vol.rolling(10, min_periods=5).sum()
        out["up_volume_dominance"] = up_vol_sum10 > (down_vol_sum10.replace(0, np.nan) * 2.0)

        # NaNをFalseで埋める
        out = out.fillna(False).astype(bool)
        # dateをindexに(元のindexを維持)
        if "date" in df.columns:
            out.index = df["date"].values

        return out

    def _detect_onset_multi_signal(
        self,
        star_stock: dict,
        grp: pd.DataFrame,
        close_col: str,
    ) -> dict:
        """マルチシグナルでスター株のonset日を検出する。

        Parameters
        ----------
        star_stock : dict
            スター株情報
        grp : pd.DataFrame
            1銘柄分の株価データ（dateソート済み）
        close_col : str
            終値カラム名

        Returns
        -------
        dict : {"onset_date": str, "signals": list[str], "score": int, "method": str}
        """
        empty = {"onset_date": "", "signals": [], "score": 0, "method": ""}

        if len(grp) < 60:
            return empty

        # 1. 日次シグナルマトリクスを計算
        signals_df = self._compute_daily_onset_signals(grp, close_col)
        if signals_df.empty:
            return empty

        # 2. 日次スコア = 各日のシグナル発火数
        daily_score = signals_df.sum(axis=1).astype(int)

        # dateシリーズを準備
        dates = pd.to_datetime(grp["date"].values)
        close_vals = grp[close_col].values

        # date→インデックスのマッピング
        date_to_idx = {d: i for i, d in enumerate(dates)}

        # 3. 閾値 4 → 3 の順で候補日を探す
        for threshold in (4, 3):
            candidate_mask = daily_score >= threshold
            candidate_dates = signals_df.index[candidate_mask]

            for cand_date in candidate_dates:
                cand_ts = pd.Timestamp(cand_date)
                if cand_ts not in date_to_idx:
                    continue
                cand_idx = date_to_idx[cand_ts]

                # 60営業日先リターン >= 15% の検証
                future_idx = cand_idx + 60
                if future_idx >= len(close_vals):
                    # データ末尾に近い場合は残りのデータで検証
                    future_idx = len(close_vals) - 1
                if future_idx <= cand_idx:
                    continue

                price_at_onset = close_vals[cand_idx]
                price_at_future = close_vals[future_idx]
                if price_at_onset <= 0:
                    continue
                fwd_return = price_at_future / price_at_onset - 1

                if fwd_return >= 0.15:
                    # 有効な候補日
                    row = signals_df.loc[cand_date]
                    fired = [name for name in _MULTI_ONSET_SIGNAL_NAMES if row[name]]
                    return {
                        "onset_date": str(cand_date)[:10],
                        "signals": fired,
                        "score": int(row.sum()),
                        "method": "multi_signal",
                    }

        # 全候補でフォワードリターン条件を満たさない場合
        return empty

    @staticmethod
    def _compute_current_onset_signals(df: pd.DataFrame, close_col: str) -> tuple[dict, int]:
        """候補銘柄の直近株価でシグナルマトリクスの最新日スコアを返す。

        _scan_timing_candidates 用。

        Returns
        -------
        (fired_signals: dict[str, bool], score: int)
        """
        signals_df = StarStockAnalyzer._compute_daily_onset_signals(df, close_col)
        if signals_df.empty:
            return {name: False for name in _MULTI_ONSET_SIGNAL_NAMES}, 0
        latest = signals_df.iloc[-1]
        fired = {name: bool(latest[name]) for name in _MULTI_ONSET_SIGNAL_NAMES}
        score = int(latest.sum())
        return fired, score

    def _apply_change_point_detection(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        config: StarStockConfig,
    ) -> list[dict]:
        """マルチシグナルOnset検出（プライマリ）+ CUSUM変化点（フォールバック/参考）

        各スター株に対して:
        1. まずマルチシグナルで初動日を検出
        2. 成功 → onset情報を設定し、CUSUMも参考として実行
        3. 失敗 → CUSUMフォールバック（従来ロジック）
        """
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        for s in star_stocks:
            grp = all_prices[all_prices["code"] == s["code"]].sort_values("date").copy()
            if len(grp) < 30:
                s["change_points"] = []
                s["star_onset_date"] = ""
                s["n_regimes"] = 1
                s["onset_return"] = None
                s["onset_detection_method"] = ""
                s["onset_signals"] = []
                s["onset_signal_score"] = 0
                s["onset_signal_matrix"] = None
                s["onset_candidates"] = []
                continue

            prices = grp[close_col].values
            dates = grp["date"].values

            # --- Onset候補生成 ---
            s["onset_candidates"] = self._generate_onset_candidates(
                grp, close_col,
                min_forward_return=config.onset_min_forward_return,
                max_candidates=config.onset_max_candidates,
            )

            # --- マルチシグナルOnset検出（プライマリ） ---
            onset_result = self._detect_onset_multi_signal(s, grp, close_col)

            if onset_result["onset_date"]:
                s["star_onset_date"] = onset_result["onset_date"]
                s["onset_signals"] = onset_result["signals"]
                s["onset_signal_score"] = onset_result["score"]
                s["onset_detection_method"] = "multi_signal"

                # onset_return計算
                onset = onset_result["onset_date"]
                try:
                    onset_ts = pd.Timestamp(onset)
                    grp_dates = pd.to_datetime(grp["date"])
                    onset_mask = grp_dates >= onset_ts
                    if onset_mask.any():
                        onset_prices = grp.loc[onset_mask, close_col].values
                        if len(onset_prices) >= 2 and onset_prices[0] > 0:
                            s["onset_return"] = float(onset_prices[-1] / onset_prices[0] - 1)
                        else:
                            s["onset_return"] = None
                    else:
                        s["onset_return"] = None
                except Exception:
                    s["onset_return"] = None

                # CUSUMも参考情報として実行
                cps = self._detect_change_points(prices, dates)
                s["change_points"] = cps
                s["n_regimes"] = len(cps) + 1

                # シグナルマトリクス保存（UI用ヒートマップ）
                try:
                    sig_df = self._compute_daily_onset_signals(grp, close_col)
                    s["onset_signal_matrix"] = sig_df
                except Exception:
                    s["onset_signal_matrix"] = None

                continue

            # --- CUSUMフォールバック ---
            cps = self._detect_change_points(prices, dates)
            s["change_points"] = cps
            s["n_regimes"] = len(cps) + 1
            s["onset_detection_method"] = "CUSUM"
            s["onset_signals"] = []
            s["onset_signal_score"] = 0

            # 最初の「上昇」変化点をスター化開始日とする
            onset = ""
            for cp in cps:
                if cp.get("type") in ("上昇開始", "加速"):
                    onset = cp["date"]
                    break
            s["star_onset_date"] = onset

            # onset_return計算
            if onset:
                try:
                    onset_ts = pd.Timestamp(onset)
                    grp_dates = pd.to_datetime(grp["date"])
                    onset_mask = grp_dates >= onset_ts
                    if onset_mask.any():
                        onset_prices = grp.loc[onset_mask, close_col].values
                        if len(onset_prices) >= 2 and onset_prices[0] > 0:
                            s["onset_return"] = float(onset_prices[-1] / onset_prices[0] - 1)
                        else:
                            s["onset_return"] = None
                    else:
                        s["onset_return"] = None
                except Exception:
                    s["onset_return"] = None
            else:
                s["onset_return"] = s.get("total_return")

            # シグナルマトリクス保存（CUSUMフォールバックでも参考用に計算）
            try:
                sig_df = self._compute_daily_onset_signals(grp, close_col)
                s["onset_signal_matrix"] = sig_df
            except Exception:
                s["onset_signal_matrix"] = None

        return star_stocks

    def _detect_change_points(
        self,
        prices: np.ndarray,
        dates: np.ndarray,
        max_cps: int = 5,
        min_segment: int = 15,
        force_at_least_one: bool = True,
    ) -> list[dict]:
        """CUSUM法 + Binary Segmentationで変化点を検出

        force_at_least_one=True の場合、有意な変化点が見つからなくても
        最大CUSUM点を信頼度付きで強制採用する（スター株には必ず変化点があるはず）。
        """
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < 2 * min_segment:
            if not force_at_least_one:
                return []
            # データが短くても最低限の変化点を探す
            min_segment = max(5, len(returns) // 4)
            if len(returns) < 2 * min_segment:
                return []

        change_points = []
        self._binary_segmentation(
            returns, dates[1:], 0, len(returns),
            change_points, max_cps, min_segment,
        )

        # 有意な変化点が1つも見つからなかった場合、最大CUSUM点を強制採用
        if not change_points and force_at_least_one and len(returns) >= 2 * min_segment:
            cp = self._force_best_change_point(returns, dates[1:], min_segment)
            if cp:
                change_points.append(cp)

        change_points.sort(key=lambda x: x["date"])
        return change_points

    def _force_best_change_point(
        self,
        returns: np.ndarray,
        dates: np.ndarray,
        min_segment: int,
    ) -> dict | None:
        """信頼度不足でも最大CUSUM点を強制的に返す"""
        mean_r = returns.mean()
        cusum = np.cumsum(returns - mean_r)
        abs_cusum = np.abs(cusum)

        search_start = min(min_segment, len(returns) // 4)
        search_end = len(returns) - search_start
        if search_start >= search_end:
            return None

        candidate_idx = search_start + np.argmax(abs_cusum[search_start:search_end])

        # ブートストラップで実際の信頼度を計算
        observed_stat = abs_cusum[candidate_idx]
        n_bootstrap = 200
        bootstrap_stats = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            shuffled = np.random.permutation(returns)
            cs = np.cumsum(shuffled - shuffled.mean())
            bootstrap_stats[b] = np.max(np.abs(cs[search_start:search_end]))
        p_value = np.mean(bootstrap_stats >= observed_stat)
        confidence = 1.0 - p_value

        before_mean = returns[:candidate_idx].mean()
        after_mean = returns[candidate_idx:].mean()
        before_vol = float(returns[:candidate_idx].std()) if candidate_idx > 1 else 0
        after_vol = float(returns[candidate_idx:].std()) if candidate_idx < len(returns) - 1 else 0

        if after_mean > before_mean and after_mean > 0:
            cp_type = "上昇開始" if before_mean <= 0 else "加速"
        elif after_mean < before_mean:
            cp_type = "減速"
        else:
            cp_type = "変化"

        date_val = dates[candidate_idx]
        date_str = str(pd.Timestamp(date_val).date()) if not isinstance(date_val, str) else date_val

        return {
            "date": date_str,
            "type": cp_type,
            "confidence": round(float(confidence), 3),
            "before_vol": round(before_vol, 6),
            "after_vol": round(after_vol, 6),
            "forced": True,
        }

    def _binary_segmentation(
        self,
        returns: np.ndarray,
        dates: np.ndarray,
        start: int,
        end: int,
        results: list,
        max_cps: int,
        min_segment: int,
    ):
        """再帰的Binary Segmentation"""
        if len(results) >= max_cps:
            return
        seg = returns[start:end]
        if len(seg) < 2 * min_segment:
            return

        # CUSUM統計量
        mean_seg = seg.mean()
        cusum = np.cumsum(seg - mean_seg)
        abs_cusum = np.abs(cusum)

        # 端を除外
        search_start = min_segment
        search_end = len(seg) - min_segment
        if search_start >= search_end:
            return

        candidate_idx = search_start + np.argmax(abs_cusum[search_start:search_end])
        observed_stat = abs_cusum[candidate_idx]

        # ブートストラップ検定
        n_bootstrap = 200
        bootstrap_stats = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            shuffled = np.random.permutation(seg)
            cs = np.cumsum(shuffled - shuffled.mean())
            bootstrap_stats[b] = np.max(np.abs(cs[search_start:search_end]))

        p_value = np.mean(bootstrap_stats >= observed_stat)
        confidence = 1.0 - p_value

        if confidence < 0.80:
            return

        # 変化点タイプ判定
        before_mean = seg[:candidate_idx].mean()
        after_mean = seg[candidate_idx:].mean()
        before_vol = float(seg[:candidate_idx].std()) if candidate_idx > 1 else 0
        after_vol = float(seg[candidate_idx:].std()) if candidate_idx < len(seg) - 1 else 0

        if after_mean > before_mean and after_mean > 0:
            cp_type = "上昇開始" if before_mean <= 0 else "加速"
        elif after_mean < before_mean:
            cp_type = "減速"
        else:
            cp_type = "変化"

        abs_idx = start + candidate_idx
        cp_date = str(pd.Timestamp(dates[abs_idx]).date()) if abs_idx < len(dates) else ""

        results.append({
            "date": cp_date,
            "type": cp_type,
            "confidence": round(float(confidence), 3),
            "before_vol": round(before_vol, 5),
            "after_vol": round(after_vol, 5),
        })

        # 再帰: 左右のセグメントにも適用
        self._binary_segmentation(returns, dates, start, start + candidate_idx, results, max_cps, min_segment)
        self._binary_segmentation(returns, dates, start + candidate_idx, end, results, max_cps, min_segment)

    # ===================================================================
    # Step 5b: Onset候補生成
    # ===================================================================
    @staticmethod
    def _generate_onset_candidates(
        grp: pd.DataFrame,
        close_col: str,
        min_forward_return: float = 0.10,
        max_candidates: int = 5,
    ) -> list[dict]:
        """1スター株に対し複数のonset候補日を生成する。

        条件: 60日先ピークリターン >= min_forward_return AND 直前20日リターン < 5%
        → 「ここから上がり始めた」ポイントを網羅的に拾う
        → 10日ウィンドウでクラスタリング、ピーク前方リターン上位に絞る
        """
        if len(grp) < 60:
            return []

        close = grp[close_col].astype(float).values
        dates = grp["date"].values
        n = len(close)
        candidates = []

        for i in range(20, n - 30):
            # 直前20日リターン < 5% （まだ大きく上がっていない）
            if close[i - 20] <= 0:
                continue
            pre_ret = close[i] / close[i - 20] - 1
            if pre_ret >= 0.05:
                continue

            # 60日先ピークリターン >= min_forward_return
            future_end = min(i + 60, n)
            future_prices = close[i:future_end]
            if close[i] <= 0 or len(future_prices) < 10:
                continue
            peak_ret = np.max(future_prices) / close[i] - 1
            if peak_ret < min_forward_return:
                continue

            # 実際の60日先リターン
            actual_end = min(i + 60, n - 1)
            fwd_ret = close[actual_end] / close[i] - 1

            candidates.append({
                "idx": i,
                "date": str(pd.Timestamp(dates[i]).date()) if not isinstance(dates[i], str) else str(dates[i])[:10],
                "pre_20d_return": round(float(pre_ret), 4),
                "peak_60d_return": round(float(peak_ret), 4),
                "fwd_60d_return": round(float(fwd_ret), 4),
            })

        if not candidates:
            return []

        # 10日ウィンドウでクラスタリング（近い日付をまとめる）
        clustered = []
        candidates.sort(key=lambda x: x["idx"])
        current_cluster = [candidates[0]]
        for c in candidates[1:]:
            if c["idx"] - current_cluster[-1]["idx"] <= 10:
                current_cluster.append(c)
            else:
                # クラスター内でピーク前方リターン最大を選択
                best = max(current_cluster, key=lambda x: x["peak_60d_return"])
                clustered.append(best)
                current_cluster = [c]
        best = max(current_cluster, key=lambda x: x["peak_60d_return"])
        clustered.append(best)

        # ピーク前方リターン上位に絞る
        clustered.sort(key=lambda x: x["peak_60d_return"], reverse=True)
        return clustered[:max_candidates]

    # ===================================================================
    # Step 6: ファクター分解
    # ===================================================================
    def _run_factor_analysis(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        topix: pd.DataFrame,
        config: StarStockConfig,
    ) -> dict:
        """日本株4ファクターモデルでリターン分解"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        # ファクターリターンを構築
        factor_returns = self._build_factor_returns(all_prices, listed_stocks, topix, close_col)
        if factor_returns.empty:
            return {"error": "ファクターリターン構築失敗"}

        # 各スター株のファクター分解
        alphas = []
        r_squareds = []
        alpha_significant = 0

        for s in star_stocks:
            code = s["code"]
            grp = all_prices[all_prices["code"] == code].sort_values("date").copy()
            if len(grp) < 30:
                s["factor_alpha"] = 0.0
                s["factor_alpha_tstat"] = 0.0
                s["factor_alpha_pvalue"] = 1.0
                s["factor_betas"] = {}
                s["factor_r_squared"] = 0.0
                s["unexplained_return_pct"] = 1.0
                continue

            stock_ret = grp.set_index("date")[close_col].pct_change(fill_method=None).dropna()
            stock_ret.name = "stock_ret"

            decomp = self._decompose_returns(stock_ret, factor_returns)
            s.update(decomp)

            if decomp["factor_alpha_pvalue"] < 0.05:
                alpha_significant += 1
            alphas.append(decomp["factor_alpha"])
            r_squareds.append(decomp["factor_r_squared"])

        n_valid = max(len(alphas), 1)
        return {
            "avg_alpha": float(np.mean(alphas)) if alphas else 0.0,
            "avg_r_squared": float(np.mean(r_squareds)) if r_squareds else 0.0,
            "alpha_significant_pct": alpha_significant / n_valid,
            "n_stocks_analyzed": n_valid,
        }

    def _build_factor_returns(
        self,
        all_prices: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        topix: pd.DataFrame,
        close_col: str,
    ) -> pd.DataFrame:
        """日次ファクターリターンを構築（MKT, SMB, HML, WML）"""
        try:
            # MKTファクター（TOPIX日次リターン）
            topix_sorted = topix.sort_values("date").copy()
            topix_sorted["MKT"] = topix_sorted["close"].pct_change(fill_method=None)
            mkt = topix_sorted.set_index("date")["MKT"].dropna()

            # 各銘柄の日次リターンを横展開
            pivot = all_prices.pivot_table(
                index="date", columns="code", values=close_col,
            )
            daily_returns = pivot.pct_change(fill_method=None).dropna(how="all")

            if daily_returns.empty or len(daily_returns) < 30:
                return pd.DataFrame()

            # SMBファクター（規模）
            scale_map = {}
            if "scale_category" in listed_stocks.columns:
                for _, row in listed_stocks.iterrows():
                    code = str(row["code"])
                    cat = row.get("scale_category", "")
                    if "Core" in str(cat) or "Large" in str(cat):
                        scale_map[code] = "big"
                    else:
                        scale_map[code] = "small"

            big_codes = [c for c in daily_returns.columns if scale_map.get(str(c)) == "big"]
            small_codes = [c for c in daily_returns.columns if scale_map.get(str(c)) == "small"]

            if big_codes and small_codes:
                smb = daily_returns[small_codes].mean(axis=1) - daily_returns[big_codes].mean(axis=1)
            else:
                # 半分で分割
                codes = list(daily_returns.columns)
                mid = len(codes) // 2
                smb = daily_returns[codes[:mid]].mean(axis=1) - daily_returns[codes[mid:]].mean(axis=1)

            # WMLファクター（モメンタム）— 簡易版
            # 月次リバランス代わりに、直近60日リターンで上位/下位を分類
            wml_series = []
            for dt in daily_returns.index:
                idx = daily_returns.index.get_loc(dt)
                if idx < 60:
                    wml_series.append(0.0)
                    continue
                past_ret = daily_returns.iloc[idx-60:idx].sum()
                valid = past_ret.dropna()
                if len(valid) < 10:
                    wml_series.append(0.0)
                    continue
                q33 = valid.quantile(0.33)
                q66 = valid.quantile(0.66)
                winners = valid[valid >= q66].index.tolist()
                losers = valid[valid <= q33].index.tolist()
                if winners and losers:
                    w = daily_returns.loc[dt, winners].mean()
                    l = daily_returns.loc[dt, losers].mean()
                    wml_series.append(float(w - l))
                else:
                    wml_series.append(0.0)

            # HMLファクター — scale_categoryをプロキシとして使用（PBRデータが不十分な場合）
            # 簡易版: ランダム分類の代わりに、セクター内高値/安値で分類
            hml_series = pd.Series(0.0, index=daily_returns.index)

            # DataFrame結合
            factors = pd.DataFrame({
                "MKT": mkt,
                "SMB": smb,
                "HML": hml_series,
                "WML": pd.Series(wml_series, index=daily_returns.index),
            }).dropna()

            return factors

        except Exception as e:
            logger.warning("ファクターリターン構築失敗: %s", e)
            return pd.DataFrame()

    @staticmethod
    def _decompose_returns(stock_ret: pd.Series, factor_returns: pd.DataFrame) -> dict:
        """OLS回帰でファクター分解"""
        # 共通日付で結合
        merged = pd.concat([stock_ret, factor_returns], axis=1, join="inner").dropna()
        if len(merged) < 20:
            return {
                "factor_alpha": 0.0,
                "factor_alpha_tstat": 0.0,
                "factor_alpha_pvalue": 1.0,
                "factor_betas": {},
                "factor_r_squared": 0.0,
                "unexplained_return_pct": 1.0,
            }

        y = merged.iloc[:, 0].values
        X = merged.iloc[:, 1:].values
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS: β = (X'X)^-1 X'y
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            return {
                "factor_alpha": 0.0,
                "factor_alpha_tstat": 0.0,
                "factor_alpha_pvalue": 1.0,
                "factor_betas": {},
                "factor_r_squared": 0.0,
                "unexplained_return_pct": 1.0,
            }

        y_hat = X_with_const @ beta
        resid = y - y_hat
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # t統計量
        n, k = X_with_const.shape
        if n > k:
            mse = ss_res / (n - k)
            try:
                cov = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                se = np.sqrt(np.diag(cov))
                t_stats = beta / se
                alpha_tstat = float(t_stats[0])
                alpha_pvalue = float(2 * (1 - stats.t.cdf(abs(alpha_tstat), df=n - k)))
            except np.linalg.LinAlgError:
                alpha_tstat = 0.0
                alpha_pvalue = 1.0
        else:
            alpha_tstat = 0.0
            alpha_pvalue = 1.0

        factor_names = list(factor_returns.columns)
        betas = {factor_names[i]: float(beta[i + 1]) for i in range(len(factor_names))}

        return {
            "factor_alpha": float(beta[0]),
            "factor_alpha_tstat": round(alpha_tstat, 3),
            "factor_alpha_pvalue": round(alpha_pvalue, 4),
            "factor_betas": betas,
            "factor_r_squared": round(float(max(r_squared, 0)), 4),
            "unexplained_return_pct": round(float(1 - max(r_squared, 0)), 4),
        }

    # ===================================================================
    # Step 7: VPIN
    # ===================================================================
    def _apply_vpin(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        config: StarStockConfig,
    ) -> list[dict]:
        """各スター株にVPIN指標を追加"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        for s in star_stocks:
            grp = all_prices[all_prices["code"] == s["code"]].sort_values("date").copy()
            if len(grp) < config.vpin_bucket_size * 2:
                s["vpin_mean"] = 0.0
                s["vpin_recent"] = 0.0
                s["vpin_earlier"] = 0.0
                s["vpin_increase"] = 0.0
                s["vpin_series"] = []
                continue

            vpin_result = self._compute_vpin(grp, close_col, config.vpin_bucket_size)
            s.update(vpin_result)

        return star_stocks

    @staticmethod
    def _compute_vpin(
        prices_df: pd.DataFrame,
        close_col: str,
        bucket_size: int = 20,
    ) -> dict:
        """VPIN (Volume-Synchronized Probability of Informed Trading) を計算"""
        returns = prices_df[close_col].pct_change(fill_method=None).dropna()
        volume = prices_df["volume"].iloc[1:].values  # returnsと長さを合わせる

        if len(returns) < bucket_size * 2:
            return {
                "vpin_mean": 0.0,
                "vpin_recent": 0.0,
                "vpin_earlier": 0.0,
                "vpin_increase": 0.0,
                "vpin_series": [],
            }

        returns_vals = returns.values
        # ローリング標準偏差
        ret_series = pd.Series(returns_vals)
        rolling_std = ret_series.rolling(20, min_periods=5).std().values

        # Bulk Volume Classification
        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = np.where(rolling_std > 0, returns_vals / rolling_std, 0)
        buy_pct = norm.cdf(z_scores)
        buy_volume = volume * buy_pct
        sell_volume = volume * (1 - buy_pct)

        # ローリングVPIN
        buy_rolling = pd.Series(buy_volume).rolling(bucket_size, min_periods=bucket_size).sum()
        sell_rolling = pd.Series(sell_volume).rolling(bucket_size, min_periods=bucket_size).sum()
        total_rolling = pd.Series(volume).rolling(bucket_size, min_periods=bucket_size).sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            vpin = np.abs(buy_rolling - sell_rolling) / total_rolling
        vpin = pd.Series(vpin).fillna(0).values

        # VPIN変化
        n_q = min(60, len(vpin) // 4)
        if n_q > 5:
            vpin_recent = float(np.mean(vpin[-n_q:]))
            vpin_earlier = float(np.mean(vpin[:n_q]))
        else:
            vpin_recent = float(np.mean(vpin))
            vpin_earlier = vpin_recent

        return {
            "vpin_mean": round(float(np.nanmean(vpin)), 4),
            "vpin_recent": round(vpin_recent, 4),
            "vpin_earlier": round(vpin_earlier, 4),
            "vpin_increase": round(vpin_recent - vpin_earlier, 4),
            "vpin_series": [round(float(v), 4) for v in vpin[::max(1, len(vpin)//100)]],  # ダウンサンプル
        }

    # ===================================================================
    # Step 8: 海外フロープロキシ + 合成スコア
    # ===================================================================
    def _compute_flow_scores(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        margin_data: dict,
        config: StarStockConfig,
    ) -> list[dict]:
        """9指標の合成フロースコアを計算"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        # TOPIX日次リターン
        topix_sorted = topix.sort_values("date").copy()
        topix_ret = topix_sorted.set_index("date")["close"].pct_change(fill_method=None).dropna()

        # セクター平均リターン計算用
        sector_map = {}
        if "sector_17_name" in listed_stocks.columns:
            for _, row in listed_stocks.iterrows():
                sector_map[str(row["code"])] = row["sector_17_name"]

        for s in star_stocks:
            code = s["code"]
            grp = all_prices[all_prices["code"] == code].sort_values("date").copy()
            if len(grp) < 30:
                s["flow_score"] = 0.0
                s["flow_indicators"] = {}
                continue

            stock_ret_raw = grp.set_index("date")[close_col].pct_change(fill_method=None)
            stock_ret = stock_ret_raw.dropna()
            volume = grp.set_index("date")["volume"]

            # インデックスを明示的に揃える（Unalignable防止）
            common_idx = stock_ret.index.intersection(volume.index)
            stock_ret_a = stock_ret.loc[common_idx]
            volume_a = volume.loc[common_idx]

            indicators = {}
            weights = {}

            # 1. アップボリューム比率
            if len(common_idx) > 0:
                up_mask = stock_ret_a.values > 0
                up_vol = volume_a.values[up_mask].sum()
                total_vol = volume_a.values.sum()
                indicators["up_volume_ratio"] = float(up_vol / total_vol) if total_vol > 0 else 0.5
            else:
                indicators["up_volume_ratio"] = 0.5
            weights["up_volume_ratio"] = 2.0

            # 2. アキュミュレーション日比率
            vol_avg = volume_a.rolling(20, min_periods=1).mean()
            vol_cond = volume_a.values > (vol_avg.values * config.accumulation_volume_threshold)
            ret_cond = np.abs(stock_ret_a.values) < config.accumulation_price_threshold
            acc_mask = vol_cond & ret_cond
            indicators["accumulation_day_ratio"] = float(acc_mask.mean()) if len(acc_mask) > 0 else 0.0
            weights["accumulation_day_ratio"] = 2.0

            # 3. 信用残乖離
            mg = margin_data.get(code)
            if mg is not None and not mg.empty and "MarginBuyingNew" in mg.columns:
                try:
                    mg_sorted = mg.copy()
                    if "Date" in mg_sorted.columns:
                        mg_sorted["date"] = pd.to_datetime(mg_sorted["Date"])
                    elif "date" not in mg_sorted.columns:
                        mg_sorted["date"] = mg_sorted.index
                    mg_sorted = mg_sorted.sort_values("date")
                    buy_bal = mg_sorted["MarginBuyingNew"].pct_change(fill_method=None).dropna()
                    # 週次→日次にリサンプル
                    buy_bal.index = mg_sorted["date"].iloc[1:].values
                    # 相関計算（週次リターンと信用買い変化）
                    weekly_ret = stock_ret.resample("W").sum()
                    common = weekly_ret.index.intersection(buy_bal.index)
                    if len(common) > 5:
                        corr_val = np.corrcoef(
                            weekly_ret.loc[common].values,
                            buy_bal.loc[common].values,
                        )[0, 1]
                        indicators["margin_divergence"] = float(-corr_val)  # 負の相関=海外フロー
                    else:
                        indicators["margin_divergence"] = 0.0
                except Exception:
                    indicators["margin_divergence"] = 0.0
            else:
                indicators["margin_divergence"] = 0.0
            weights["margin_divergence"] = 3.0

            # 4. セクター連動性
            sector = sector_map.get(code, "")
            same_sector_codes = [
                c for c, sec in sector_map.items()
                if sec == sector and c != code
            ]
            if same_sector_codes and sector:
                sector_prices = all_prices[all_prices["code"].isin(same_sector_codes)]
                sector_pivot = sector_prices.pivot_table(index="date", columns="code", values=close_col)
                sector_mean_ret = sector_pivot.pct_change(fill_method=None).mean(axis=1).dropna()
                common = stock_ret.index.intersection(sector_mean_ret.index)
                if len(common) > config.sector_correlation_window:
                    rolling_corr = stock_ret.loc[common].rolling(
                        config.sector_correlation_window
                    ).corr(sector_mean_ret.loc[common])
                    avg_corr = rolling_corr.mean()
                    # 低い連動性 = 独自要因 → 海外フローの可能性
                    indicators["sector_decorrelation"] = float(1.0 - avg_corr) if not np.isnan(avg_corr) else 0.5
                else:
                    indicators["sector_decorrelation"] = 0.5
            else:
                indicators["sector_decorrelation"] = 0.5
            weights["sector_decorrelation"] = 1.5

            # 5. 時価総額区分スコア
            scale_score_map = {
                "TOPIX Core30": 1.0,
                "TOPIX Large70": 0.8,
                "TOPIX Mid400": 0.5,
                "TOPIX Small 1": 0.3,
                "TOPIX Small 2": 0.2,
            }
            indicators["scale_score"] = scale_score_map.get(s.get("scale_category", ""), 0.2)
            weights["scale_score"] = 1.0

            # 6. TOPIXベータシフト
            common_dates = stock_ret.index.intersection(topix_ret.index)
            if len(common_dates) > config.rolling_beta_window * 2:
                sr = stock_ret.loc[common_dates]
                tr = topix_ret.loc[common_dates]
                n_half = len(common_dates) // 2
                # 前半ベータ
                cov_first = np.cov(sr.iloc[:n_half], tr.iloc[:n_half])
                beta_first = cov_first[0, 1] / cov_first[1, 1] if cov_first[1, 1] > 0 else 1.0
                # 後半ベータ
                cov_second = np.cov(sr.iloc[n_half:], tr.iloc[n_half:])
                beta_second = cov_second[0, 1] / cov_second[1, 1] if cov_second[1, 1] > 0 else 1.0
                indicators["beta_shift"] = float(beta_second - beta_first)
            else:
                indicators["beta_shift"] = 0.0
            weights["beta_shift"] = 1.5

            # 7. OBVトレンド強度
            obv = pd.Series(
                np.sign(stock_ret_a.values) * volume_a.values,
                index=common_idx,
            ).cumsum()
            if len(obv) > config.obv_trend_window:
                x = np.arange(len(obv))
                slope, intercept, r_val, p_val, std_err = stats.linregress(x, obv.values)
                avg_vol = volume.mean()
                indicators["obv_trend_strength"] = float(
                    slope * r_val**2 / avg_vol
                ) if avg_vol > 0 else 0.0
            else:
                indicators["obv_trend_strength"] = 0.0
            weights["obv_trend_strength"] = 1.5

            # 8. VPIN変化
            indicators["vpin_increase"] = s.get("vpin_increase", 0.0)
            weights["vpin_increase"] = 2.5

            # 9. ファクターα有意性
            alpha_tstat = s.get("factor_alpha_tstat", 0.0)
            indicators["factor_alpha_significant"] = 1.0 if abs(alpha_tstat) > 2.0 else 0.0
            weights["factor_alpha_significant"] = 1.5

            # 合成スコア（0-1に正規化）
            score_components = []
            weight_sum = 0.0
            for key, w in weights.items():
                val = indicators.get(key, 0.0)
                # 各指標を[0,1]にクリップ
                val_clipped = max(0.0, min(1.0, val))
                score_components.append(val_clipped * w)
                weight_sum += w

            flow_score = sum(score_components) / weight_sum if weight_sum > 0 else 0.0
            s["flow_score"] = round(float(flow_score), 4)
            s["flow_indicators"] = {k: round(float(v), 4) for k, v in indicators.items()}

            # 基本特徴量にも追加
            s["up_volume_ratio"] = indicators["up_volume_ratio"]
            s["accumulation_day_ratio"] = indicators["accumulation_day_ratio"]
            s["obv_trend_strength"] = indicators["obv_trend_strength"]
            s["beta_shift"] = indicators["beta_shift"]

        return star_stocks

    # ===================================================================
    # Step 9: Lead-Lag分析
    # ===================================================================
    def _lead_lag_analysis(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        config: StarStockConfig,
    ) -> dict:
        """Granger因果検定でLead-Lag構造を分析"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        if len(star_stocks) < 2:
            return {"pairs": [], "leader_counts": {}, "follower_counts": {}}

        # 各スター株のリターン系列を事前計算
        star_codes = [s["code"] for s in star_stocks[:20]]  # 上位20銘柄に制限
        returns_map = {}
        for code in star_codes:
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 30:
                continue
            ret = grp.set_index("date")[close_col].pct_change(fill_method=None).dropna()
            returns_map[code] = ret

        pairs = []
        leader_counts = {c: 0 for c in star_codes}
        follower_counts = {c: 0 for c in star_codes}

        for code_a, code_b in combinations(returns_map.keys(), 2):
            ret_a = returns_map[code_a]
            ret_b = returns_map[code_b]

            # 共通日付
            common = ret_a.index.intersection(ret_b.index)
            if len(common) < config.lead_lag_max_lag + 20:
                continue

            a_vals = ret_a.loc[common].values
            b_vals = ret_b.loc[common].values

            # A → B のGranger因果
            for lag in range(1, min(config.lead_lag_max_lag + 1, 6)):
                f_stat, p_val = self._granger_test(b_vals, a_vals, lag)
                if p_val < 0.05:
                    pairs.append({
                        "leader": code_a,
                        "follower": code_b,
                        "lag_days": lag,
                        "f_stat": round(float(f_stat), 3),
                        "p_value": round(float(p_val), 4),
                    })
                    leader_counts[code_a] = leader_counts.get(code_a, 0) + 1
                    follower_counts[code_b] = follower_counts.get(code_b, 0) + 1
                    break  # 最初の有意なラグのみ記録

            # B → A のGranger因果
            for lag in range(1, min(config.lead_lag_max_lag + 1, 6)):
                f_stat, p_val = self._granger_test(a_vals, b_vals, lag)
                if p_val < 0.05:
                    pairs.append({
                        "leader": code_b,
                        "follower": code_a,
                        "lag_days": lag,
                        "f_stat": round(float(f_stat), 3),
                        "p_value": round(float(p_val), 4),
                    })
                    leader_counts[code_b] = leader_counts.get(code_b, 0) + 1
                    follower_counts[code_a] = follower_counts.get(code_a, 0) + 1
                    break

        # 各スター株にLead-Lag役割を付与
        for s in star_stocks:
            code = s["code"]
            n_lead = leader_counts.get(code, 0)
            n_follow = follower_counts.get(code, 0)
            if n_lead > n_follow and n_lead > 0:
                s["lead_lag_role"] = "leader"
            elif n_follow > n_lead and n_follow > 0:
                s["lead_lag_role"] = "follower"
            else:
                s["lead_lag_role"] = "independent"
            s["leads"] = [p for p in pairs if p["leader"] == code][:5]
            s["follows"] = [p for p in pairs if p["follower"] == code][:5]

        return {
            "pairs": pairs[:50],  # 上位50ペアに制限
            "leader_counts": {k: v for k, v in leader_counts.items() if v > 0},
            "follower_counts": {k: v for k, v in follower_counts.items() if v > 0},
        }

    @staticmethod
    def _granger_test(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float]:
        """Granger因果検定: xがyをGranger-causeするかF検定"""
        n = len(y)
        if n < 2 * lag + 5:
            return 0.0, 1.0

        # ラグ行列を作成
        Y = y[lag:]
        n_obs = len(Y)

        # 制限モデル: y[t] ~ y[t-1], ..., y[t-lag]
        X_r = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
        X_r = np.column_stack([np.ones(n_obs), X_r])

        # 無制限モデル: y[t] ~ y[t-1:t-lag] + x[t-1:t-lag]
        X_u = np.column_stack([
            X_r,
            *[x[lag - i - 1:n - i - 1] for i in range(lag)],
        ])

        try:
            # 制限モデルのRSS
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            rss_r = np.sum((Y - X_r @ beta_r) ** 2)

            # 無制限モデルのRSS
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
            rss_u = np.sum((Y - X_u @ beta_u) ** 2)

            # F統計量
            df1 = lag
            df2 = n_obs - 2 * lag - 1
            if df2 <= 0 or rss_u <= 0:
                return 0.0, 1.0

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)

            return float(f_stat), float(p_value)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0

    # ===================================================================
    # Step 10: MLクラスタリング
    # ===================================================================
    def _cluster_analysis(
        self,
        star_stocks: list[dict],
        config: StarStockConfig,
    ) -> dict:
        """K-means + PCA可視化"""
        if len(star_stocks) < config.n_clusters:
            return {"error": f"銘柄数({len(star_stocks)})がクラスター数({config.n_clusters})より少ない"}

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError:
            return {"error": "scikit-learn がインストールされていません: pip install scikit-learn"}

        feature_keys = [
            "total_return", "excess_return", "max_drawdown",
            "acceleration", "volume_change_ratio", "volume_surge_count",
            "realized_vol_change", "flow_score", "vpin_increase",
            "factor_alpha", "up_volume_ratio", "accumulation_day_ratio",
            "obv_trend_strength", "beta_shift",
        ]

        # 特徴量マトリクス構築
        X = np.array([
            [float(s.get(k, 0) or 0) for k in feature_keys]
            for s in star_stocks
        ])

        # NaN/Inf置換
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means
        n_clusters = min(config.n_clusters, len(star_stocks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # PCA 2D
        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # 各スター株にクラスタラベルを付与
        for i, s in enumerate(star_stocks):
            s["cluster"] = int(labels[i])

        # クラスタープロファイル
        cluster_profiles = []
        for c in range(n_clusters):
            mask = labels == c
            members = [star_stocks[i]["code"] for i in range(len(star_stocks)) if mask[i]]
            member_names = [star_stocks[i].get("name", "") for i in range(len(star_stocks)) if mask[i]]
            centroid = kmeans.cluster_centers_[c]
            profile = {feature_keys[i]: round(float(centroid[i]), 4) for i in range(len(feature_keys))}
            cluster_profiles.append({
                "cluster_id": c,
                "n_members": int(mask.sum()),
                "member_codes": members,
                "member_names": member_names,
                "centroid_profile": profile,
            })

        return {
            "labels": labels.tolist(),
            "pca_coords": X_pca.tolist(),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "cluster_profiles": cluster_profiles,
            "feature_keys": feature_keys,
            "inertia": float(kmeans.inertia_),
            "n_clusters": n_clusters,
        }

    # ===================================================================
    # Step 11: クロスセクショナル回帰 (Fama-MacBeth)
    # ===================================================================
    def _cross_sectional_regression(
        self,
        all_prices: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        topix: pd.DataFrame,
        margin_data: dict,
        config: StarStockConfig,
    ) -> dict:
        """Fama-MacBeth型横断面回帰"""
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        try:
            # 月次データ構築
            all_prices_copy = all_prices.copy()
            all_prices_copy["date"] = pd.to_datetime(all_prices_copy["date"])
            all_prices_copy["yearmonth"] = all_prices_copy["date"].dt.to_period("M")

            months = sorted(all_prices_copy["yearmonth"].unique())
            if len(months) < 4:
                return {"error": "月数不足（最低4ヶ月必要）"}

            feature_names = [
                "volume_surge_3m", "beta_change", "realized_vol_ratio",
                "momentum_3m", "up_volume_ratio",
            ]

            monthly_gammas = []

            for i in range(2, len(months) - 1):
                current_month = months[i]
                next_month = months[i + 1]

                # 当月と前月のデータ
                current_data = all_prices_copy[all_prices_copy["yearmonth"] == current_month]
                next_data = all_prices_copy[all_prices_copy["yearmonth"] == next_month]

                # 3ヶ月のルックバック
                lookback_months = [months[j] for j in range(max(0, i - 2), i + 1)]
                lookback_data = all_prices_copy[all_prices_copy["yearmonth"].isin(lookback_months)]

                # 翌月リターン
                next_returns = {}
                for code, grp in next_data.groupby("code"):
                    grp = grp.sort_values("date")
                    if len(grp) >= 2:
                        ret = grp[close_col].iloc[-1] / grp[close_col].iloc[0] - 1
                        next_returns[code] = ret

                if len(next_returns) < 20:
                    continue

                # 特徴量計算
                features = {}
                for code, grp in lookback_data.groupby("code"):
                    if code not in next_returns:
                        continue
                    grp = grp.sort_values("date")
                    if len(grp) < 15:
                        continue

                    prices_arr = grp[close_col].values
                    vol_arr = grp["volume"].values
                    rets = pd.Series(prices_arr).pct_change(fill_method=None).dropna().values

                    # 出来高急増回数（3ヶ月）
                    vol_s = pd.Series(vol_arr)
                    vol_ma = vol_s.rolling(20, min_periods=1).mean()
                    surge_count = int((vol_s > vol_ma * 2.0).sum())

                    # β変化
                    beta_change = 0.0  # 簡略化

                    # ボラティリティ変化率
                    rets = rets[np.isfinite(rets)]
                    n_q = max(len(rets) // 4, 2)
                    vol_early = float(np.std(rets[:n_q])) if len(rets[:n_q]) > 1 else 0.001
                    vol_late = float(np.std(rets[-n_q:])) if len(rets[-n_q:]) > 1 else 0.001
                    vol_ratio = vol_late / vol_early if vol_early > 0 else 1.0

                    # モメンタム
                    momentum = prices_arr[-1] / prices_arr[0] - 1 if prices_arr[0] > 0 else 0

                    # アップボリューム比率（numpy配列で直接計算）
                    vol_tail = vol_arr[1:len(rets) + 1]
                    up_vol = vol_tail[rets > 0].sum() if len(vol_tail) == len(rets) else 0
                    total_vol = vol_tail.sum()
                    up_vol_ratio = up_vol / total_vol if total_vol > 0 else 0.5

                    features[code] = [
                        surge_count, beta_change, vol_ratio,
                        momentum, up_vol_ratio,
                    ]

                # 横断面回帰
                codes_common = [c for c in features if c in next_returns]
                if len(codes_common) < 20:
                    continue

                X = np.array([features[c] for c in codes_common])
                y = np.array([next_returns[c] for c in codes_common])

                # NaN/Inf除去
                mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
                X = X[mask]
                y = y[mask]

                if len(y) < 20:
                    continue

                # 切片付きOLS
                X_with_const = np.column_stack([np.ones(len(X)), X])
                try:
                    gamma = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    monthly_gammas.append(gamma[1:])  # 切片除く
                except np.linalg.LinAlgError:
                    continue

            if len(monthly_gammas) < 3:
                return {"error": f"有効な月数不足（{len(monthly_gammas)}ヶ月）"}

            gammas = np.array(monthly_gammas)
            gamma_mean = gammas.mean(axis=0)
            gamma_std = gammas.std(axis=0, ddof=1)
            n_months = len(gammas)

            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = gamma_mean / (gamma_std / np.sqrt(n_months))
            t_stats = np.nan_to_num(t_stats, nan=0.0)

            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(n_months - 1, 1)))

            significant = [
                f for f, p in zip(feature_names, p_values) if p < 0.05
            ]

            return {
                "features": feature_names,
                "coefficients": [round(float(g), 6) for g in gamma_mean],
                "t_statistics": [round(float(t), 3) for t in t_stats],
                "p_values": [round(float(p), 4) for p in p_values],
                "significant_predictors": significant,
                "n_months": n_months,
            }

        except Exception as e:
            logger.warning("クロスセクショナル回帰エラー: %s", e)
            return {"error": str(e)}

    # ===================================================================
    # Step 12: 反復的特徴量発見 (Iterative Discriminative Feature Discovery)
    # ===================================================================
    def _discover_discriminative_features(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        topix: pd.DataFrame,
        config: StarStockConfig,
    ) -> dict:
        """反復的に特徴量と閾値を探索し、スター株onset日を判別する最適な
        シグナル組み合わせを発見する。

        Phase 1: 正例/負例サンプル構築
        Phase 2: 26特徴量マトリクス計算
        Phase 3: 最適閾値 (Youden's J)
        Phase 4: 組み合わせ探索
        Phase 5: 全ユニバース検証
        Phase 6: 反復精緻化
        Phase 7: Onset日確定
        """
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"
        star_codes = set(s["code"] for s in star_stocks)
        n_star = len(star_stocks)

        # ETF/REIT除外
        equity_codes = set()
        if "market_name" in listed_stocks.columns:
            equity_mask = listed_stocks["market_name"].isin(_STOCK_MARKET_SEGMENTS)
            equity_codes = set(listed_stocks.loc[equity_mask, "code"].astype(str))
        else:
            equity_codes = set(listed_stocks["code"].astype(str))

        # TOPIX日次リターン（β計算用） — インデックスを必ずTimestampに統一
        topix_ret_series = None
        if not topix.empty:
            topix_sorted = topix.sort_values("date").copy()
            topix_sorted["date"] = pd.to_datetime(topix_sorted["date"])
            topix_ret_series = topix_sorted.set_index("date")["close"].pct_change(fill_method=None).dropna()

        # 市場全体のvol5/vol60比率を計算（vol_vs_market_vol用）
        market_vol_ratio = None
        try:
            all_vol = all_prices.groupby("date")["volume"].sum().sort_index()
            if len(all_vol) >= 60:
                mv5 = all_vol.rolling(5, min_periods=3).mean().iloc[-1]
                mv60 = all_vol.rolling(60, min_periods=30).mean().iloc[-1]
                market_vol_ratio = float(mv5 / mv60) if mv60 > 0 else None
        except Exception:
            pass

        # 全銘柄をグループ化
        all_prices_sorted = all_prices.sort_values("date")
        price_groups = {str(code): grp.sort_values("date")
                        for code, grp in all_prices_sorted.groupby("code")}

        # セクターマップ構築
        sector_map = {}
        if "sector_17_name" in listed_stocks.columns:
            for _, row in listed_stocks.iterrows():
                sector_map[str(row["code"])] = row["sector_17_name"]

        # --- Phase 1: サンプル構築 ---
        # 正例: 各スター株のonset候補日
        positive_samples = []  # [(code, date_str, grp_slice)]
        onset_candidates_by_code = {}

        for s in star_stocks:
            code = s["code"]
            grp = price_groups.get(code)
            if grp is None or len(grp) < 60:
                continue

            candidates = self._generate_onset_candidates(
                grp, close_col,
                min_forward_return=config.onset_min_forward_return,
                max_candidates=config.onset_max_candidates,
            )
            if not candidates:
                # フォールバック: 既存onset_dateを使用
                od = s.get("star_onset_date", "")
                if od:
                    candidates = [{"date": od, "idx": -1, "peak_60d_return": 0, "fwd_60d_return": 0, "pre_20d_return": 0}]

            onset_candidates_by_code[code] = candidates

            grp_dates = pd.to_datetime(grp["date"])
            for cand in candidates:
                cand_ts = pd.Timestamp(cand["date"])
                pre_onset = grp[grp_dates <= cand_ts].tail(60)
                if len(pre_onset) >= 20:
                    positive_samples.append((code, cand["date"], pre_onset))

        if not positive_samples:
            return {"signals": [], "combo_signals": [], "best_combos": [],
                    "n_star": n_star, "n_all": 0, "base_rate": 0,
                    "star_onset_profiles": {},
                    "signal_thresholds": {},
                    "timing_specificity": {},
                    "discovery_iterations": 0, "discovery_converged": False,
                    "wide_feature_keys": WIDE_FEATURE_KEYS}

        # 負例（銘柄対照）: 各onset日の非スター株からランダム抽出
        negative_stock_samples = []
        rng = np.random.default_rng(42)
        sampled_onset_dates = list(set(s[1] for s in positive_samples))
        non_star_codes = [c for c in price_groups if c not in star_codes and c in equity_codes]

        for onset_date_str in sampled_onset_dates[:5]:  # onset日を最大5日に制限
            onset_ts = pd.Timestamp(onset_date_str)
            sample_codes = rng.choice(
                non_star_codes,
                size=min(config.discovery_neg_sample_size, len(non_star_codes)),
                replace=False,
            ) if len(non_star_codes) > 0 else []

            for code_str in sample_codes:
                grp = price_groups.get(code_str)
                if grp is None or len(grp) < 30:
                    continue
                grp_dates = pd.to_datetime(grp["date"])
                pre_onset = grp[grp_dates <= onset_ts].tail(60)
                if len(pre_onset) >= 20:
                    negative_stock_samples.append((code_str, onset_date_str, pre_onset))

        # 負例（時間対照）: 同一スター株の非候補日
        negative_time_samples = []
        for s in star_stocks:
            code = s["code"]
            grp = price_groups.get(code)
            if grp is None or len(grp) < 120:
                continue
            candidates = onset_candidates_by_code.get(code, [])
            cand_dates = set(c["date"] for c in candidates)
            grp_dates = pd.to_datetime(grp["date"])

            # 4つのコントロール時点
            period_start = pd.Timestamp(config.start_date)
            period_end = pd.Timestamp(config.end_date)
            span = (period_end - period_start).days
            control_offsets = [0.1, 0.3, 0.6, 0.85]
            for frac in control_offsets:
                ctrl_ts = period_start + pd.Timedelta(days=int(span * frac))
                # onset候補日から30日以内はスキップ
                too_close = any(abs((ctrl_ts - pd.Timestamp(d)).days) < 30 for d in cand_dates)
                if too_close:
                    continue
                pre_ctrl = grp[grp_dates <= ctrl_ts].tail(60)
                if len(pre_ctrl) >= 20:
                    negative_time_samples.append((code, ctrl_ts.strftime("%Y-%m-%d"), pre_ctrl))

        # --- Phase 2: 特徴量マトリクス構築 ---
        all_samples = (
            [(code, dt, sl, 1) for code, dt, sl in positive_samples]
            + [(code, dt, sl, 0) for code, dt, sl in negative_stock_samples]
            + [(code, dt, sl, 0) for code, dt, sl in negative_time_samples]
        )

        # セクター平均リターンのプリ計算
        sector_ret_cache = {}

        X_list = []
        y_list = []
        sample_info = []
        for code, dt, sl, label in all_samples:
            sector = sector_map.get(code, "")
            sector_key = f"{sector}_{dt}"
            if sector_key not in sector_ret_cache and sector:
                # 同セクターの10日平均リターンを簡易計算
                sector_codes_list = [c for c, s in sector_map.items() if s == sector and c != code]
                if sector_codes_list:
                    rets_10d = []
                    for sc in sector_codes_list[:30]:  # 30銘柄に制限
                        sg = price_groups.get(sc)
                        if sg is None or len(sg) < 15:
                            continue
                        sg_dates = pd.to_datetime(sg["date"])
                        sg_pre = sg[sg_dates <= pd.Timestamp(dt)].tail(11)
                        if len(sg_pre) >= 2:
                            sc_close = sg_pre[close_col].values
                            rets_10d.append(float(sc_close[-1] / sc_close[0] - 1) if sc_close[0] > 0 else 0)
                    sector_ret_cache[sector_key] = np.mean(rets_10d) if rets_10d else 0.0
                else:
                    sector_ret_cache[sector_key] = 0.0

            feat = self._compute_wide_features(
                sl, close_col,
                sector_ret_10d=sector_ret_cache.get(sector_key),
                market_vol_ratio=market_vol_ratio,
                topix_ret_series=topix_ret_series,
            )
            if feat is None:
                continue

            X_list.append([feat.get(k, 0.0) for k in WIDE_FEATURE_KEYS])
            y_list.append(label)
            sample_info.append((code, dt, label))

        if len(X_list) < 10:
            return {"signals": [], "combo_signals": [], "best_combos": [],
                    "n_star": n_star, "n_all": len(X_list), "base_rate": 0,
                    "star_onset_profiles": {},
                    "signal_thresholds": {},
                    "timing_specificity": {},
                    "discovery_iterations": 0, "discovery_converged": False,
                    "wide_feature_keys": WIDE_FEATURE_KEYS}

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        n_total = len(y)
        base_rate = n_pos / n_total if n_total > 0 else 0

        # --- Phase 3: 最適閾値（Youden's J）---
        signal_results = []
        signal_thresholds = {}

        for fi, fkey in enumerate(WIDE_FEATURE_KEYS):
            col = X[:, fi]
            pos_vals = col[y == 1]
            neg_vals = col[y == 0]

            if len(pos_vals) < 3:
                signal_results.append({
                    "key": fkey,
                    "label": _WIDE_FEATURE_LABELS_JP.get(fkey, fkey),
                    "threshold": 0.0, "precision": 0.0, "recall": 0.0,
                    "f1": 0.0, "lift": 0.0, "youden_j": 0.0, "auc_approx": 0.5,
                    "verdict": "データ不足",
                    "star_triggered": 0, "star_total": n_pos,
                    "non_star_triggered": 0, "non_star_total": n_neg,
                    "total_triggered": 0,
                })
                signal_thresholds[fkey] = 0.0
                continue

            # 正例分布の10-90パーセンタイルを候補閾値として走査
            best_j = -1
            best_thresh = float(np.median(pos_vals))
            best_tpr = 0
            best_fpr = 1

            percentiles = np.arange(10, 95, 10)
            for p in percentiles:
                thresh = float(np.percentile(pos_vals, p))
                tp = np.sum(pos_vals >= thresh)
                fn = np.sum(pos_vals < thresh)
                fp = np.sum(neg_vals >= thresh)
                tn = np.sum(neg_vals < thresh)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                j = tpr - fpr
                if j > best_j:
                    best_j = j
                    best_thresh = thresh
                    best_tpr = tpr
                    best_fpr = fpr

            signal_thresholds[fkey] = best_thresh

            # 最適閾値でのメトリクス
            star_triggered = int(np.sum(pos_vals >= best_thresh))
            non_star_triggered = int(np.sum(neg_vals >= best_thresh))
            total_triggered = star_triggered + non_star_triggered
            precision = star_triggered / total_triggered if total_triggered > 0 else 0
            recall = star_triggered / n_pos if n_pos > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            lift = precision / base_rate if base_rate > 0 else 0

            # AUC近似（台形法）
            auc_approx = 0.5
            try:
                threshs = np.sort(np.unique(col))
                if len(threshs) > 2:
                    tprs, fprs = [], []
                    for t in threshs[::max(1, len(threshs) // 20)]:
                        tp_ = np.sum(pos_vals >= t)
                        fn_ = np.sum(pos_vals < t)
                        fp_ = np.sum(neg_vals >= t)
                        tn_ = np.sum(neg_vals < t)
                        tprs.append(tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0)
                        fprs.append(fp_ / (fp_ + tn_) if (fp_ + tn_) > 0 else 0)
                    # sort by fpr
                    pairs = sorted(zip(fprs, tprs))
                    fprs_s = [p[0] for p in pairs]
                    tprs_s = [p[1] for p in pairs]
                    auc_approx = float(np.trapz(tprs_s, fprs_s))
                    auc_approx = max(0, min(1, auc_approx))
            except Exception:
                pass

            if lift >= 3.0 and best_j >= 0.3:
                verdict = "強いシグナル"
            elif lift >= 2.0:
                verdict = "やや有効"
            elif lift >= 1.2:
                verdict = "弱い"
            else:
                verdict = "無意味"

            signal_results.append({
                "key": fkey,
                "label": _WIDE_FEATURE_LABELS_JP.get(fkey, fkey),
                "threshold": round(best_thresh, 4),
                "star_triggered": star_triggered,
                "star_total": n_pos,
                "non_star_triggered": non_star_triggered,
                "non_star_total": n_neg,
                "total_triggered": total_triggered,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "lift": round(lift, 2),
                "youden_j": round(best_j, 4),
                "auc_approx": round(auc_approx, 4),
                "verdict": verdict,
            })

        signal_results.sort(key=lambda x: x["lift"], reverse=True)

        # --- Phase 4: 組み合わせ探索 ---
        # 上位15特徴量からコンボ探索
        top_signals = sorted(signal_results, key=lambda x: x.get("youden_j", 0), reverse=True)[:15]
        top_keys = [s["key"] for s in top_signals]
        top_indices = [WIDE_FEATURE_KEYS.index(k) for k in top_keys]

        combo_results = []

        # 2特徴量コンボ: C(15,2) = 105
        for ci, cj in combinations(range(len(top_keys)), 2):
            ki, kj = top_keys[ci], top_keys[cj]
            ti, tj = signal_thresholds[ki], signal_thresholds[kj]
            fi_idx, fj_idx = top_indices[ci], top_indices[cj]

            mask_fire = (X[:, fi_idx] >= ti) & (X[:, fj_idx] >= tj)
            star_hit = int((mask_fire & (y == 1)).sum())
            non_star_hit = int((mask_fire & (y == 0)).sum())
            total_hit = star_hit + non_star_hit

            if total_hit < 3:
                continue
            precision = star_hit / total_hit if total_hit > 0 else 0
            recall = star_hit / n_pos if n_pos > 0 else 0
            lift = precision / base_rate if base_rate > 0 else 0

            combo_results.append({
                "keys": [ki, kj],
                "labels": " AND ".join([_WIDE_FEATURE_LABELS_JP.get(k, k) for k in [ki, kj]]),
                "thresholds": {ki: round(ti, 4), kj: round(tj, 4)},
                "n_signals": 2,
                "star_hit": star_hit,
                "non_star_hit": non_star_hit,
                "total_hit": total_hit,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "lift": round(lift, 2),
            })

        # 3特徴量コンボ: 上位10からC(10,3) = 120
        top10_keys = top_keys[:10]
        top10_indices = top_indices[:10]
        for ci, cj, ck in combinations(range(len(top10_keys)), 3):
            ki, kj, kk = top10_keys[ci], top10_keys[cj], top10_keys[ck]
            ti, tj, tk = signal_thresholds[ki], signal_thresholds[kj], signal_thresholds[kk]
            fi_idx, fj_idx, fk_idx = top10_indices[ci], top10_indices[cj], top10_indices[ck]

            mask_fire = (X[:, fi_idx] >= ti) & (X[:, fj_idx] >= tj) & (X[:, fk_idx] >= tk)
            star_hit = int((mask_fire & (y == 1)).sum())
            non_star_hit = int((mask_fire & (y == 0)).sum())
            total_hit = star_hit + non_star_hit

            if total_hit < 3:
                continue
            precision = star_hit / total_hit if total_hit > 0 else 0
            recall = star_hit / n_pos if n_pos > 0 else 0
            lift = precision / base_rate if base_rate > 0 else 0

            combo_results.append({
                "keys": [ki, kj, kk],
                "labels": " AND ".join([_WIDE_FEATURE_LABELS_JP.get(k, k) for k in [ki, kj, kk]]),
                "thresholds": {ki: round(ti, 4), kj: round(tj, 4), kk: round(tk, 4)},
                "n_signals": 3,
                "star_hit": star_hit,
                "non_star_hit": non_star_hit,
                "total_hit": total_hit,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "lift": round(lift, 2),
            })

        combo_results.sort(key=lambda x: x["precision"], reverse=True)

        # --- Phase 5: 全ユニバース検証 (上位20コンボ) ---
        # 1回だけ全銘柄の特徴量を計算してキャッシュ
        best_combos_raw = combo_results[:20]

        universe_features = {}  # code -> {key: val}
        for code_str in list(price_groups.keys())[:3000]:
            if code_str in star_codes:
                continue
            if code_str not in equity_codes:
                continue
            grp = price_groups[code_str]
            if len(grp) < 30:
                continue
            recent = grp.tail(60)
            feat = self._compute_wide_features(
                recent, close_col,
                market_vol_ratio=market_vol_ratio,
                topix_ret_series=topix_ret_series,
            )
            if feat is not None:
                universe_features[code_str] = feat

        # キャッシュされた特徴量で全コンボを検証
        for combo in best_combos_raw:
            fp_count = 0
            total_checked = len(universe_features)
            for code_str, feat in universe_features.items():
                all_fired = all(
                    feat.get(k, 0) >= combo["thresholds"].get(k, 0)
                    for k in combo["keys"]
                )
                if all_fired:
                    fp_count += 1

            fp_rate = fp_count / total_checked if total_checked > 0 else 0
            combo["validation_fp_count"] = fp_count
            combo["validation_total_checked"] = total_checked
            combo["validation_fp_rate"] = round(fp_rate, 4)

        # --- Phase 6: 反復精緻化 ---
        max_iterations = config.discovery_max_iterations
        target_precision = config.discovery_target_precision
        min_recall = config.discovery_min_recall
        converged = False
        iteration = 0

        while iteration < max_iterations:
            passing = [
                c for c in best_combos_raw
                if c["precision"] >= target_precision and c["recall"] >= min_recall
            ]
            if passing:
                converged = True
                break

            # 精緻化: 閾値を10パーセンタイル刻みで厳格化
            improved = False
            for combo in best_combos_raw:
                if combo["precision"] >= target_precision:
                    continue
                for key in combo["keys"]:
                    fi_idx = WIDE_FEATURE_KEYS.index(key)
                    pos_vals = X[y == 1, fi_idx]
                    current_thresh = combo["thresholds"][key]
                    # 現在の閾値より10パーセンタイル上を試す
                    current_pctile = float(np.searchsorted(np.sort(pos_vals), current_thresh) / len(pos_vals) * 100) if len(pos_vals) > 0 else 50
                    new_pctile = min(current_pctile + 10, 90)
                    new_thresh = float(np.percentile(pos_vals, new_pctile))
                    if new_thresh > current_thresh:
                        combo["thresholds"][key] = round(new_thresh, 4)
                        improved = True

                # 再計算
                keys = combo["keys"]
                mask_fire = np.ones(len(y), dtype=bool)
                for k in keys:
                    fi_idx = WIDE_FEATURE_KEYS.index(k)
                    mask_fire &= X[:, fi_idx] >= combo["thresholds"][k]

                star_hit = int((mask_fire & (y == 1)).sum())
                non_star_hit = int((mask_fire & (y == 0)).sum())
                total_hit = star_hit + non_star_hit
                combo["star_hit"] = star_hit
                combo["non_star_hit"] = non_star_hit
                combo["total_hit"] = total_hit
                combo["precision"] = round(star_hit / total_hit, 4) if total_hit > 0 else 0
                combo["recall"] = round(star_hit / n_pos, 4) if n_pos > 0 else 0
                combo["lift"] = round(combo["precision"] / base_rate, 2) if base_rate > 0 else 0

            if not improved:
                break
            iteration += 1

        best_combos_raw.sort(key=lambda x: x["precision"], reverse=True)

        # タイミング特異性をコンボに追加
        for combo in best_combos_raw:
            keys = combo["keys"]
            # onset日発火率
            onset_fires = 0
            onset_total = 0
            for ps_code, ps_dt, ps_sl, ps_label in all_samples:
                if ps_label != 1:
                    continue
                onset_total += 1
                feat_vals = {}
                for k in keys:
                    fi_idx = WIDE_FEATURE_KEYS.index(k)
                    idx_in_X = next(
                        (i for i, (c, d, l) in enumerate(sample_info) if c == ps_code and d == ps_dt and l == 1),
                        None
                    )
                    if idx_in_X is not None:
                        feat_vals[k] = X[idx_in_X, fi_idx]
                if all(feat_vals.get(k, 0) >= combo["thresholds"].get(k, 0) for k in keys):
                    onset_fires += 1
            onset_fire_rate = onset_fires / onset_total if onset_total > 0 else 0

            # コントロール日発火率
            ctrl_fires = 0
            ctrl_total = 0
            for ns_code, ns_dt, ns_sl, ns_label in all_samples:
                if ns_label != 0:
                    continue
                if ns_code not in star_codes:
                    continue  # 時間対照のみ
                ctrl_total += 1
                idx_in_X = next(
                    (i for i, (c, d, l) in enumerate(sample_info) if c == ns_code and d == ns_dt and l == 0),
                    None
                )
                if idx_in_X is not None:
                    fire = all(X[idx_in_X, WIDE_FEATURE_KEYS.index(k)] >= combo["thresholds"].get(k, 0) for k in keys)
                    if fire:
                        ctrl_fires += 1
            ctrl_fire_rate = ctrl_fires / ctrl_total if ctrl_total > 0 else 0

            timing_lift = (
                onset_fire_rate / ctrl_fire_rate if ctrl_fire_rate > 0
                else (10.0 if onset_fire_rate > 0 else 1.0)
            )
            c_timing_score = max(0, onset_fire_rate - ctrl_fire_rate)
            c_lift = combo["lift"]
            is_doubly = c_lift >= 1.5 and (c_timing_score >= 0.30 or timing_lift >= 2.0)
            normalized_lift = min(c_lift / 5.0, 1.0)
            disc_score = normalized_lift * min(c_timing_score * 2, 1.0) if c_timing_score > 0 else 0

            combo.update({
                "onset_fire_rate": round(onset_fire_rate, 4),
                "control_fire_rate": round(ctrl_fire_rate, 4),
                "timing_lift": round(timing_lift, 2),
                "timing_score": round(c_timing_score, 4),
                "discriminative_score": round(disc_score, 4),
                "is_doubly_specific": is_doubly,
            })

        # --- Phase 7: Onset日確定 ---
        # 各スター株の候補日のうち、passingコンボが最も多く発火する日をonset日とする
        star_onset_profiles = {}
        for s in star_stocks:
            code = s["code"]
            candidates = onset_candidates_by_code.get(code, [])
            if not candidates:
                continue

            grp = price_groups.get(code)
            if grp is None:
                continue
            grp_dates = pd.to_datetime(grp["date"])

            best_date = candidates[0]["date"]
            best_score = -1

            for cand in candidates:
                cand_ts = pd.Timestamp(cand["date"])
                pre_onset = grp[grp_dates <= cand_ts].tail(60)
                if len(pre_onset) < 20:
                    continue
                feat = self._compute_wide_features(
                    pre_onset, close_col,
                    market_vol_ratio=market_vol_ratio,
                    topix_ret_series=topix_ret_series,
                )
                if feat is None:
                    continue

                # passingコンボ発火数をスコアとする
                combo_score = 0
                for combo in best_combos_raw[:10]:
                    all_fired = all(
                        feat.get(k, 0) >= combo["thresholds"].get(k, 0)
                        for k in combo["keys"]
                    )
                    if all_fired:
                        combo_score += 1

                cand["combo_score"] = combo_score
                if combo_score > best_score:
                    best_score = combo_score
                    best_date = cand["date"]
                    star_onset_profiles[code] = feat

            # onset日を更新
            s["star_onset_date"] = best_date
            s["onset_combo_score"] = best_score

        # タイミング特異性（個別特徴量）
        timing_specificity = {}
        for sig in signal_results:
            key = sig["key"]
            fi_idx = WIDE_FEATURE_KEYS.index(key)
            # 正例 vs 負例の分布比較
            pos_mean = float(X[y == 1, fi_idx].mean()) if n_pos > 0 else 0
            neg_mean = float(X[y == 0, fi_idx].mean()) if n_neg > 0 else 0
            diff = pos_mean - neg_mean
            timing_specificity[key] = {
                "timing_score": round(min(max(0.5 + diff * 5, 0), 1.0), 4),
                "pos_mean": round(pos_mean, 4),
                "neg_mean": round(neg_mean, 4),
            }

            # シグナル結果にも追加
            ts_info = timing_specificity[key]
            sig["timing_score"] = ts_info["timing_score"]
            sig["discriminative_score"] = round(sig["lift"] / 5.0 * ts_info["timing_score"], 4)
            sig["is_doubly_specific"] = sig["lift"] >= 1.5 and ts_info["timing_score"] >= 0.7

        return {
            "signals": signal_results,
            "combo_signals": best_combos_raw[:30],
            "best_combos": [c for c in best_combos_raw if c["precision"] >= target_precision][:10],
            "n_star": n_star,
            "n_star_samples": n_pos,
            "n_non_star_samples": n_neg,
            "n_all": n_total,
            "base_rate": round(base_rate, 6),
            "star_onset_profiles": star_onset_profiles,
            "signal_thresholds": signal_thresholds,
            "timing_specificity": timing_specificity,
            "onset_candidates": onset_candidates_by_code,
            "discovery_iterations": iteration,
            "discovery_converged": converged,
            "wide_feature_keys": WIDE_FEATURE_KEYS,
        }

    # ===================================================================
    # Step 13: AI要約生成
    # ===================================================================
    def _generate_ai_summary(
        self,
        star_stocks: list[dict],
        topix_return: float,
        factor_analysis: dict,
        cluster_analysis: dict,
        lead_lag: dict,
        cross_sectional: dict,
        config: StarStockConfig,
        signal_validation: dict | None = None,
    ) -> dict:
        """AIに分析結果の要約を生成させる"""
        if self.ai_client is None:
            return self._generate_fallback_summary(
                star_stocks, topix_return, factor_analysis, cluster_analysis,
                lead_lag, cross_sectional, signal_validation,
            )

        # スター株サマリー（上位30銘柄）
        top_stocks = sorted(star_stocks, key=lambda x: x.get("excess_return", 0), reverse=True)[:30]
        stock_summary = []
        for s in top_stocks:
            stock_summary.append({
                "code": s["code"],
                "name": s.get("name", ""),
                "sector": s.get("sector", ""),
                "total_return": f"{s.get('total_return', 0):.1%}",
                "excess_return": f"{s.get('excess_return', 0):.1%}",
                "flow_score": f"{s.get('flow_score', 0):.2f}",
                "vpin_increase": f"{s.get('vpin_increase', 0):.4f}",
                "factor_alpha_tstat": f"{s.get('factor_alpha_tstat', 0):.2f}",
                "cluster": s.get("cluster", -1),
                "lead_lag_role": s.get("lead_lag_role", "unknown"),
                "scale": s.get("scale_category", ""),
            })

        # クラスター情報
        cluster_info = []
        if "cluster_profiles" in cluster_analysis:
            for cp in cluster_analysis["cluster_profiles"]:
                cluster_info.append({
                    "id": cp["cluster_id"],
                    "n": cp["n_members"],
                    "members": cp["member_names"][:5],
                    "profile": {k: f"{v:.2f}" for k, v in list(cp["centroid_profile"].items())[:6]},
                })

        # シグナル検証結果サマリー（個別 + 組み合わせ）
        sig_val_text = "データなし"
        if signal_validation and "signals" in signal_validation:
            sig_lines = ["### 個別シグナル（onset日断面 + タイミング特異性）"]
            for sv in signal_validation["signals"][:7]:
                ts = sv.get('timing_score', 0)
                ds = sv.get('is_doubly_specific', False)
                sig_lines.append(
                    f"- {sv['label']}: Precision={sv['precision']:.1%}, Recall={sv['recall']:.1%}, "
                    f"Lift={sv['lift']:.1f}x, TimingScore={ts:.2f}, 二重特異={'Yes' if ds else 'No'}, "
                    f"判定={sv['verdict']} "
                    f"(閾値{sv['threshold']:.3f}で全{sv['total_triggered']}銘柄中スター{sv['star_triggered']}銘柄)"
                )
            # 組み合わせシグナル
            combo = signal_validation.get("combo_signals", [])
            if combo:
                sig_lines.append("\n### シグナル組み合わせ（AND条件 + タイミング特異性）")
                for cs in combo[:10]:
                    ds = cs.get('is_doubly_specific', False)
                    tl = cs.get('timing_lift', 0)
                    sig_lines.append(
                        f"- [{cs['labels']}]: Precision={cs['precision']:.1%}, "
                        f"Recall={cs['recall']:.1%}, Lift={cs['lift']:.1f}x, "
                        f"TimingLift={tl:.1f}x, 二重特異={'Yes' if ds else 'No'} "
                        f"({cs['total_hit']}銘柄中{cs['star_hit']}がスター)"
                    )
            sig_val_text = "\n".join(sig_lines)

        prompt = f"""以下の日本株「スター株」分析結果を解釈し、JSON形式で回答してください。

**最も重要**: 「シグナル組み合わせ」の結果を中心に、特に「二重特異」と判定されたものを分析してください。
個別シグナルのPrecisionは低くて当然。2-3シグナルのAND条件でPrecisionが高くなる組み合わせが
本当の「スター株早期発見ルール」です。
「二重特異」= 銘柄特異性（スター株だけが持つ）× タイミング特異性（onset日だけに現れる）の両方を満たすシグナル。
「銘柄特異のみ」= 多くの株がいつでも持っている特徴なので価値が低い。
Precision 20%以上の組み合わせを「有効」、30%以上を「強い」と判定してください。

**分析手法**: スター株がスター化した「その日」の断面（onset日の60日前データ）で
全銘柄の特徴量を計算し、その日の断面でスター株とそれ以外を区別できるかを検証しています。

**表現**: 投資初心者にもわかるよう平易な日本語で。

## 分析期間
{config.start_date} ～ {config.end_date}（TOPIX: {topix_return:.1%}）

## スター株サマリー（上位30銘柄）
{json.dumps(stock_summary, ensure_ascii=False, indent=1)}

## シグナル検証結果（Onset日断面分析）
base_rate（断面上のスター株比率）: {signal_validation.get('base_rate', 0):.2%}
{sig_val_text}

## ファクター分析
平均R²: {factor_analysis.get('avg_r_squared', 0):.2f}（既知ファクターで説明できる割合）

## クラスター分析
{json.dumps(cluster_info, ensure_ascii=False, indent=1)}

## 回答形式（JSON）
{{
  "common_features_summary": "シグナル組み合わせの結果を中心に、スター株を最も高い精度で見分けられる条件は何かを300字で。Precision値とLift値を根拠に。",
  "pattern_typology": [
    {{"cluster_id": 0, "pattern_name": "直感的な名前", "description": "特徴と代表銘柄名", "representative_stocks": ["コード"]}},
    ...
  ],
  "foreign_flow_assessment": "海外投資家関与の可能性を具体的根拠とともに200字で",
  "detection_rules": "二重特異と判定されたシグナル組み合わせを使った早期発見ルールを箇条書きで。各ルールにPrecision・Recall・Lift値・TimingLift値を付記。「銘柄特異のみ」（多くの株が持つ特徴）は使えない理由も明記。"
}}
"""

        try:
            response = self.ai_client.send_message(prompt)
            result = self._parse_json_response(response)
            if result:
                return result
        except Exception as e:
            logger.warning("AI要約生成失敗: %s", e)

        return self._generate_fallback_summary(
            star_stocks, topix_return, factor_analysis, cluster_analysis,
            lead_lag, cross_sectional, signal_validation,
        )

    @staticmethod
    def _parse_json_response(response: str) -> dict | None:
        """AI応答からJSONを抽出"""
        # コードブロック内のJSONを探す
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            json_str = response[start:end].strip()
        else:
            # 直接JSONを試す
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # { から } までを抽出
            first_brace = response.find("{")
            last_brace = response.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                try:
                    return json.loads(response[first_brace:last_brace + 1])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _generate_fallback_summary(
        star_stocks: list[dict],
        topix_return: float,
        factor_analysis: dict,
        cluster_analysis: dict,
        lead_lag: dict,
        cross_sectional: dict,
        signal_validation: dict | None = None,
    ) -> dict:
        """AI不使用時のフォールバック要約"""
        n = len(star_stocks)
        avg_ret = np.mean([s.get("total_return", 0) for s in star_stocks]) if star_stocks else 0
        avg_flow = np.mean([s.get("flow_score", 0) for s in star_stocks]) if star_stocks else 0

        # セクター分布
        sectors = {}
        for s in star_stocks:
            sec = s.get("sector", "不明")
            sectors[sec] = sectors.get(sec, 0) + 1
        top_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:3]

        # シグナル検証結果から共通特徴を生成
        common = f"分析期間中に{n}銘柄のスター株を検出（市場平均TOPIX: {topix_return:.1%}、スター株平均: {avg_ret:.1%}）。"
        common += f"主要業種: {', '.join(f'{s}({c}銘柄)' for s, c in top_sectors)}。\n\n"

        # 組み合わせシグナルの結果を中心に表示
        combo = signal_validation.get("combo_signals", []) if signal_validation else []
        ds_combos = [c for c in combo if c.get("is_doubly_specific", False)]
        strong_combos = [c for c in combo if c["precision"] >= 0.20]
        if ds_combos:
            common += "**二重特異なシグナル組み合わせ（銘柄×タイミング特異性）**:\n"
            for cs in ds_combos[:5]:
                common += (
                    f"- {cs['labels']}: Precision {cs['precision']:.1%}, Lift {cs['lift']:.1f}x, "
                    f"TimingLift {cs.get('timing_lift', 0):.1f}x"
                    f"（{cs['total_hit']}銘柄中{cs['star_hit']}がスター）\n"
                )
            common += "\n"
        elif strong_combos:
            common += "**有効なシグナル組み合わせ（Precision 20%以上）**:\n"
            for cs in strong_combos[:5]:
                common += f"- {cs['labels']}: Precision {cs['precision']:.1%}, Lift {cs['lift']:.1f}x（{cs['total_hit']}銘柄中{cs['star_hit']}がスター）\n"
            common += "\n"

        if signal_validation and "signals" in signal_validation:
            strong = [sv for sv in signal_validation["signals"] if sv["verdict"] in ("強いシグナル", "やや有効")]
            if strong:
                common += "**個別で有効なシグナル**: "
                common += "; ".join(
                    f"{sv['label']}（Lift {sv['lift']:.1f}x, Precision {sv['precision']:.1%}）"
                    for sv in strong
                )
                common += "。\n"
            common += "\n※個別シグナルのPrecisionは低いため、組み合わせ条件の利用を推奨。"

        foreign = f"海外フロースコアの平均は{avg_flow:.2f}（1.0が最大）。"

        # クラスタープロファイル
        typology = []
        if "cluster_profiles" in cluster_analysis:
            for cp in cluster_analysis["cluster_profiles"]:
                typology.append({
                    "cluster_id": cp["cluster_id"],
                    "pattern_name": f"タイプ{cp['cluster_id'] + 1}",
                    "description": f"{cp['n_members']}銘柄: " + ", ".join(cp["member_names"][:3]),
                    "representative_stocks": cp["member_codes"][:3],
                })

        # 早期発見ルール — 組み合わせシグナル中心
        rules = "### Onset日断面分析に基づく早期発見ルール\n\n"
        rules += "スター株がスター化した「その日」の断面で全銘柄を評価し、スター株だけに特有の条件を抽出。\n\n"
        if signal_validation:
            base_pct = signal_validation.get("base_rate", 0) * 100
            rules += f"断面上のスター株比率（ベースレート）: {base_pct:.2f}%\n\n"

        if combo:
            rules += "#### シグナル組み合わせルール（AND条件 + タイミング特異性）\n\n"
            for i, cs in enumerate(combo[:15]):
                prec_pct = cs["precision"] * 100
                is_ds = cs.get("is_doubly_specific", False)
                if is_ds:
                    tag = "**二重特異**"
                elif prec_pct >= 30:
                    tag = "**強い**"
                elif prec_pct >= 20:
                    tag = "**有効**"
                elif prec_pct >= 10:
                    tag = "やや有効"
                else:
                    tag = "弱い"
                thresholds_str = " & ".join(
                    f"{k}≥{v:.3f}" for k, v in cs["thresholds"].items()
                )
                timing_info = ""
                if cs.get("timing_lift"):
                    timing_info = f", TimingLift {cs['timing_lift']:.1f}x"
                rules += (
                    f"{i+1}. {tag} [{cs['labels']}]\n"
                    f"   条件: {thresholds_str}\n"
                    f"   Precision {cs['precision']:.1%}（{cs['total_hit']}銘柄中{cs['star_hit']}がスター）, "
                    f"Recall {cs['recall']:.1%}, Lift {cs['lift']:.1f}x{timing_info}\n\n"
                )

        if signal_validation and "signals" in signal_validation:
            rules += "#### 個別シグナル参考値\n\n"
            for sv in signal_validation["signals"]:
                verdict_tag = {"強いシグナル": "**有効**", "やや有効": "**やや有効**", "弱い": "弱い", "無意味": "~~無意味~~"}
                tag = verdict_tag.get(sv["verdict"], sv["verdict"])
                rules += (
                    f"- {tag} **{sv['label']}** ≥ {sv['threshold']:.3f}: "
                    f"Precision {sv['precision']:.1%}, Lift {sv['lift']:.1f}x\n"
                )
            rules += "\n※個別シグナルのPrecisionは低いため、2-3シグナルの組み合わせで使用してください。\n"
        elif not combo:
            rules += "シグナル検証データがありません。\n"

        return {
            "common_features_summary": common,
            "pattern_typology": typology,
            "foreign_flow_assessment": foreign,
            "detection_rules": rules,
        }

    # ===================================================================
    # Step 14: 買いタイミング近接度スキャン
    # ===================================================================
    def _scan_timing_candidates(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        topix: pd.DataFrame,
        config: StarStockConfig,
        signal_validation: dict | None = None,
    ) -> list[dict]:
        """26ワイド特徴量テンプレート + discoveredコンボで候補を探索。

        スコア式: similarity*0.3 + combo_ratio*0.3 + precision_bonus*0.4
        コンボ0発火時は similarity * 0.3 にキャップ。
        """
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        # TOPIX日次リターン系列とmarket_vol_ratioの計算
        topix_ret_series = None
        market_vol_ratio = None
        if topix is not None and len(topix) > 0:
            topix_sorted = topix.sort_values("date").copy()
            topix_sorted["date"] = pd.to_datetime(topix_sorted["date"])
            topix_ret_series = topix_sorted.set_index("date")["close"].pct_change(fill_method=None).dropna()

        if "volume" in all_prices.columns:
            agg_vol = all_prices.groupby("date")["volume"].sum().sort_index()
            mv5 = agg_vol.rolling(5, min_periods=1).mean().iloc[-1] if len(agg_vol) >= 5 else None
            mv60 = agg_vol.rolling(60, min_periods=10).mean().iloc[-1] if len(agg_vol) >= 10 else None
            if mv5 is not None and mv60 is not None and mv60 > 0:
                market_vol_ratio = float(mv5 / mv60)

        # ETF/REIT除外
        equity_codes = set()
        if "market_name" in listed_stocks.columns:
            equity_mask = listed_stocks["market_name"].isin(_STOCK_MARKET_SEGMENTS)
            equity_codes = set(listed_stocks.loc[equity_mask, "code"].astype(str))
        else:
            equity_codes = set(listed_stocks["code"].astype(str))

        # 仕手株フィルター
        scale_cap_map = {
            "TOPIX Core30": 5000.0, "TOPIX Large70": 2000.0,
            "TOPIX Mid400": 500.0, "TOPIX Small 1": 100.0, "TOPIX Small 2": 50.0,
        }
        code_scale_map = {}
        if "scale_category" in listed_stocks.columns:
            for _, row in listed_stocks[["code", "scale_category"]].iterrows():
                code_scale_map[str(row["code"])] = row.get("scale_category", "")

        # --- 1. テンプレート構築（26特徴量） ---
        star_onset_profiles = {}
        if signal_validation:
            star_onset_profiles = signal_validation.get("star_onset_profiles", {})

        template_features = []
        feature_keys = WIDE_FEATURE_KEYS
        if star_onset_profiles:
            template_features = list(star_onset_profiles.values())
        else:
            # フォールバック: _compute_wide_features で計算
            for s in star_stocks:
                onset = s.get("star_onset_date", "")
                code = s["code"]
                grp = all_prices[all_prices["code"] == code].sort_values("date")
                if len(grp) < 60:
                    continue
                if onset:
                    onset_ts = pd.Timestamp(onset)
                    pre_onset = grp[pd.to_datetime(grp["date"]) <= onset_ts].tail(60)
                else:
                    n_quarter = max(len(grp) // 4, 30)
                    pre_onset = grp.head(n_quarter).tail(60)
                if len(pre_onset) < 20:
                    continue
                feat = self._compute_wide_features(
                    pre_onset, close_col,
                    market_vol_ratio=market_vol_ratio,
                    topix_ret_series=topix_ret_series,
                )
                if feat is not None:
                    template_features.append(feat)

        if not template_features:
            return []

        avg_template = {}
        for k in feature_keys:
            vals = [f.get(k, 0) for f in template_features if k in f]
            avg_template[k] = float(np.mean(vals)) if vals else 0.0

        template_vec = np.array([avg_template.get(k, 0) for k in feature_keys])
        template_norm = np.linalg.norm(template_vec)
        if template_norm == 0:
            return []

        # discoveredコンボ（data-driven閾値）
        best_combos = []
        if signal_validation:
            combo_sigs = signal_validation.get("combo_signals", [])
            best_combos = [c for c in combo_sigs if c.get("is_doubly_specific", False)][:10]
            if not best_combos:
                best_combos = [c for c in combo_sigs if c.get("precision", 0) >= 0.10][:10]

        # ベストコンボのPrecision平均（ボーナス計算用）
        avg_combo_precision = np.mean([c.get("precision", 0) for c in best_combos]) if best_combos else 0

        # --- 2. 全銘柄の直近60日を評価 ---
        star_codes = set(s["code"] for s in star_stocks)
        meta_map = {}
        meta_cols = ["code", "name", "sector_17_name", "market_name", "scale_category"]
        available_meta = [c for c in meta_cols if c in listed_stocks.columns]
        for _, row in listed_stocks[available_meta].iterrows():
            meta_map[str(row["code"])] = {c: row.get(c, "") for c in available_meta}

        candidates = []

        for code, grp in all_prices.groupby("code"):
            code_str = str(code)
            if code_str in star_codes or code_str not in equity_codes:
                continue

            cand_scale = code_scale_map.get(code_str, "")
            cand_est_cap = scale_cap_map.get(cand_scale, 30.0)
            if cand_est_cap < config.min_market_cap_billion:
                continue

            grp = grp.sort_values("date")
            recent = grp.tail(60)
            if len(recent) < 30:
                continue

            feat = self._compute_wide_features(
                recent, close_col,
                market_vol_ratio=market_vol_ratio,
                topix_ret_series=topix_ret_series,
            )
            if feat is None:
                continue

            # コサイン類似度
            cand_vec = np.array([feat.get(k, 0) for k in feature_keys])
            cand_norm = np.linalg.norm(cand_vec)
            if cand_norm == 0:
                continue
            similarity = float(np.dot(template_vec, cand_vec) / (template_norm * cand_norm))

            # コンボ発火チェック（discovered閾値を直接使用）
            combos_fired = 0
            doubly_specific_fired = 0
            combos_total = len(best_combos)
            fired_combo_names = []
            fired_ds_combo_names = []
            for combo in best_combos:
                keys = combo["keys"]
                thresholds = combo.get("thresholds", {})
                all_fired = all(feat.get(k, 0) >= thresholds.get(k, 0) for k in keys)
                if all_fired:
                    combos_fired += 1
                    fired_combo_names.append(combo["labels"])
                    if combo.get("is_doubly_specific", False):
                        doubly_specific_fired += 1
                        fired_ds_combo_names.append(combo["labels"])

            # マルチシグナルOnsetスコア
            onset_signals_fired, onset_signal_score = self._compute_current_onset_signals(
                recent, close_col,
            )

            meta = meta_map.get(code_str, {})
            candidates.append({
                "code": code_str,
                "name": meta.get("name", ""),
                "sector": meta.get("sector_17_name", ""),
                "market": meta.get("market_name", ""),
                "scale_category": meta.get("scale_category", ""),
                "est_market_cap": cand_est_cap,
                "similarity": round(similarity, 4),
                "combos_fired": combos_fired,
                "doubly_specific_fired": doubly_specific_fired,
                "combos_total": combos_total,
                "fired_combo_names": fired_combo_names,
                "fired_ds_combo_names": fired_ds_combo_names,
                "onset_signal_score": onset_signal_score,
                "onset_signals_fired": onset_signals_fired,
                "features": {k: round(float(feat.get(k, 0)), 4) for k in feature_keys},
                "recent_return_60d": round(float(
                    recent[close_col].iloc[-1] / recent[close_col].iloc[0] - 1
                ), 4) if recent[close_col].iloc[0] > 0 else 0.0,
                "recent_volume_change": round(float(
                    recent["volume"].iloc[-15:].mean() / recent["volume"].iloc[:15].mean()
                ), 2) if recent["volume"].iloc[:15].mean() > 0 else 1.0,
            })

        # スコア: similarity*0.3 + combo_ratio*0.3 + precision_bonus*0.4
        for c in candidates:
            combo_ratio = c["combos_fired"] / max(c["combos_total"], 1)
            precision_bonus = avg_combo_precision * combo_ratio if c["combos_fired"] > 0 else 0
            if c["combos_fired"] == 0:
                # コンボ0発火時はキャップ
                c["composite_score"] = round(c["similarity"] * 0.3, 4)
            else:
                c["composite_score"] = round(
                    c["similarity"] * 0.3 + combo_ratio * 0.3 + precision_bonus * 0.4, 4
                )

        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        return candidates[:50]

    @staticmethod
    def _compute_onset_features_legacy(df: pd.DataFrame, close_col: str) -> dict | None:
        """旧7特徴量ベクトル（レガシー互換用）"""
        if len(df) < 15:
            return None

        prices = df[close_col].values
        volume = df["volume"].values
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 10:
            return None

        vol_series = pd.Series(volume)
        vol_ma = vol_series.rolling(20, min_periods=5).mean()

        # 出来高急増頻度
        surge_ratio = float((vol_series > vol_ma * 2.0).mean())

        # ボラティリティ（標準偏差）
        volatility = float(np.std(returns))

        # モメンタム（期間リターン）
        momentum = float(prices[-1] / prices[0] - 1) if prices[0] > 0 else 0.0

        # 上昇日比率
        up_ratio = float((returns > 0).mean())

        # 出来高トレンド（後半 / 前半）
        n_half = len(volume) // 2
        vol_trend = float(volume[n_half:].mean() / volume[:n_half].mean()) if volume[:n_half].mean() > 0 else 1.0

        # 価格の安定度（変動の小ささ = アキュミュレーション傾向）
        price_stability = float(1.0 / (1.0 + np.std(returns) * 100))

        # OBVトレンド（符号付き出来高の累積勾配）
        obv = np.cumsum(np.sign(returns) * volume[1:len(returns)+1])
        if len(obv) > 5:
            x = np.arange(len(obv))
            slope = np.polyfit(x, obv, 1)[0]
            obv_trend = float(slope / (volume.mean() + 1))
        else:
            obv_trend = 0.0

        return {
            "volume_surge_freq": surge_ratio,
            "volatility": volatility,
            "momentum": momentum,
            "up_ratio": up_ratio,
            "volume_trend": vol_trend,
            "price_stability": price_stability,
            "obv_trend": obv_trend,
        }
