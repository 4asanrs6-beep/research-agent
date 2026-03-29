"""個別株イベントスタディ — US→JP ペア別伝播分析

US個別株が大きく動いた日（イベント）を起点に、
JP個別株の翌日以降の累積異常リターン（CAR）を計算し、
銘柄間の伝播関係を発見する。

ETF Lead-Lag戦略とは独立した、ミクロ・イベント的な分析。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =====================================================================
# US業種 → JP業種の対応候補 (セクターペアのガイド)
# 完全な一対一対応ではなく、探索範囲を絞るためのヒント
# =====================================================================
SECTOR_PAIR_GUIDE: list[dict] = [
    {
        "label": "半導体",
        "us_tickers": ["NVDA", "AVGO", "AMD", "INTC", "TXN", "QCOM", "MU", "LRCX", "KLAC", "AMAT"],
        "jp_sector_33": ["電気機器"],
        "jp_hint_codes": ["80350", "68570", "67230", "69630", "77350"],  # 東京エレクトロン, アドバンテスト, ルネサス, ローム, スクリーン
    },
    {
        "label": "テック/ソフトウェア",
        "us_tickers": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "CRM", "ADBE", "ORCL"],
        "jp_sector_33": ["情報・通信業", "電気機器"],
        "jp_hint_codes": ["67580", "68610", "99840", "46890", "60980", "94330"],  # ソニー, キーエンス, SBG, Zホールディングス, リクルート, KDDI
    },
    {
        "label": "金融/銀行",
        "us_tickers": ["JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW"],
        "jp_sector_33": ["銀行業", "証券、商品先物取引業", "保険業"],
        "jp_hint_codes": ["83060", "83160", "84110", "86040", "87660", "88010"],  # MUFG, 三井住友, みずほ, 野村, 東京海上, 三井不動産
    },
    {
        "label": "自動車/EV",
        "us_tickers": ["TSLA", "GM", "F", "RIVN"],
        "jp_sector_33": ["輸送用機器"],
        "jp_hint_codes": ["72030", "72670", "72690", "72010", "69020"],  # トヨタ, ホンダ, スズキ, 日産, デンソー
    },
    {
        "label": "医薬品",
        "us_tickers": ["LLY", "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN"],
        "jp_sector_33": ["医薬品"],
        "jp_hint_codes": ["45680", "45190", "45020", "45030", "45230"],  # 第一三共, 中外製薬, 武田, アステラス, エーザイ
    },
    {
        "label": "素材/資源",
        "us_tickers": ["XOM", "CVX", "COP", "FCX", "NEM", "BHP"],
        "jp_sector_33": ["鉱業", "鉄鋼", "非鉄金属", "石油・石炭製品"],
        "jp_hint_codes": ["16050", "54010", "57130", "50200", "58020"],  # INPEX, 日本製鉄, 住友金属鉱山, ENEOS, 住友電工
    },
    {
        "label": "商社/総合",
        "us_tickers": ["BRK-B", "GE", "HON", "MMM", "CAT"],
        "jp_sector_33": ["卸売業"],
        "jp_hint_codes": ["80580", "80310", "80010", "80020", "80530"],  # 三菱商事, 三井物産, 伊藤忠, 丸紅, 住友商事
    },
]


@dataclass
class PairResult:
    """1ペアのイベントスタディ結果"""
    us_ticker: str
    jp_code: str
    jp_name: str = ""
    sector_label: str = ""
    n_events: int = 0
    # CAR at each lag: {1: 0.008, 2: 0.012, 3: 0.011, 5: 0.009, 10: 0.005}
    car: dict[int, float] = field(default_factory=dict)
    car_t_stat: dict[int, float] = field(default_factory=dict)
    car_p_value: dict[int, float] = field(default_factory=dict)
    # ピーク情報
    peak_lag: int = 0
    peak_car: float = 0.0
    # 方向一致率 (US上昇日にJPも上昇する率)
    direction_hit_rate: float = 0.0


@dataclass
class EventStudyResult:
    """全ペアのイベントスタディ結果"""
    pairs: list[PairResult] = field(default_factory=list)
    significant_pairs: list[PairResult] = field(default_factory=list)
    lags_tested: list[int] = field(default_factory=list)
    event_threshold: float = 0.0
    isolated_only: bool = False
    n_us_stocks: int = 0
    n_jp_stocks: int = 0


def _score_pair(p: PairResult) -> float:
    """ペアのスコアを計算する。"""
    if not p.car_p_value:
        return 0
    min_p = min(p.car_p_value.values())
    return abs(p.peak_car) * (-np.log10(max(min_p, 1e-20)))


def _apply_fdr(
    results: list[PairResult],
    lags: list[int],
    alpha: float = 0.05,
) -> list[PairResult]:
    """Benjamini-Hochberg FDR補正を適用して有意なペアを返す。"""
    # 全ペア×全ラグのp値を収集
    pvals = []
    pair_lag_map = []  # (pair_index, lag)
    for i, pair in enumerate(results):
        for lag in lags:
            if lag in pair.car_p_value:
                pvals.append(pair.car_p_value[lag])
                pair_lag_map.append((i, lag))

    if not pvals:
        return []

    pvals = np.array(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    # BH補正: p_{(k)} <= k/n * alpha
    thresholds = np.arange(1, n + 1) / n * alpha
    reject = sorted_pvals <= thresholds

    # 最大のkを見つける
    if not reject.any():
        return []
    max_k = np.where(reject)[0][-1]
    rejected_indices = set(sorted_idx[:max_k + 1])

    # 有意なペアを収集
    sig_pair_indices = set()
    for idx in rejected_indices:
        pair_i, lag = pair_lag_map[idx]
        sig_pair_indices.add(pair_i)

    return [results[i] for i in sorted(sig_pair_indices)]


def run_event_study(
    us_prices: pd.DataFrame,
    jp_prices: pd.DataFrame,
    sector_pairs: list[dict] | None = None,
    lags: list[int] | None = None,
    event_threshold_sigma: float = 2.0,
    min_events: int = 10,
    isolated_only: bool = False,
    isolated_window: int = 3,
    p_threshold: float = 0.05,
    fdr_correction: bool = False,
    jp_code_names: dict[str, str] | None = None,
    progress_callback=None,
) -> EventStudyResult:
    """US→JP 個別株ペアのイベントスタディを実行する。

    Args:
        us_prices: US個別株の日次リターン (index=date, columns=ticker)
        jp_prices: JP個別株の日次リターン (index=date, columns=code)
        sector_pairs: セクターペアのリスト。None=全ペアスキャン
            各要素: {"us_tickers": [...], "jp_codes": [...], "label": "..."}
        lags: テストするラグのリスト。デフォルト [1, 2, 3, 5, 10]
        event_threshold_sigma: イベント閾値（σの倍数）
        min_events: 最低イベント数
        isolated_only: 孤立イベントのみ使う
        isolated_window: 孤立判定の前後日数
        p_threshold: 有意水準
        jp_code_names: {code: company_name} マッピング
        progress_callback: (msg, pct) コールバック

    Returns:
        EventStudyResult
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("ペアリストを構築中...", 0.0)

    # 共通日付
    common_dates = us_prices.index.intersection(jp_prices.index)
    us_ret = us_prices.reindex(common_dates)
    jp_ret = jp_prices.reindex(common_dates)

    # ペアリスト構築
    if sector_pairs:
        pair_list = []
        for sp in sector_pairs:
            us_t = [t for t in sp.get("us_tickers", []) if t in us_ret.columns]
            jp_c = [c for c in sp.get("jp_codes", []) if c in jp_ret.columns]
            label = sp.get("label", "")
            for u in us_t:
                for j in jp_c:
                    pair_list.append((u, j, label))
    else:
        # 全ペア（非推奨だが可能）
        pair_list = [
            (u, j, "")
            for u in us_ret.columns
            for j in jp_ret.columns
        ]

    n_pairs = len(pair_list)
    if n_pairs == 0:
        logger.warning("テスト対象ペアがありません")
        return EventStudyResult(lags_tested=lags)

    logger.info("イベントスタディ: %d ペアをテスト", n_pairs)

    # 各USティッカーのイベント日を事前計算
    _progress("イベント日を抽出中...", 0.05)
    us_events: dict[str, np.ndarray] = {}  # ticker -> array of event date indices
    us_event_signs: dict[str, np.ndarray] = {}  # ticker -> array of signs (+1/-1)

    for ticker in us_ret.columns:
        rets = us_ret[ticker].values
        mu = np.nanmean(rets)
        sigma = np.nanstd(rets)
        if sigma < 1e-10:
            continue

        threshold = event_threshold_sigma * sigma
        event_mask = np.abs(rets - mu) > threshold

        if isolated_only:
            # 前後N日に他のイベントがない日だけ
            clean_mask = event_mask.copy()
            for idx in np.where(event_mask)[0]:
                start = max(0, idx - isolated_window)
                end = min(len(rets), idx + isolated_window + 1)
                neighbors = np.sum(event_mask[start:end]) - 1  # 自分を除く
                if neighbors > 0:
                    clean_mask[idx] = False
            event_mask = clean_mask

        event_indices = np.where(event_mask)[0]
        if len(event_indices) >= min_events:
            us_events[ticker] = event_indices
            us_event_signs[ticker] = np.sign(rets[event_indices] - mu)

    # ペアごとにCAR計算
    _progress("CAR計算中...", 0.10)
    results = []
    max_lag = max(lags)

    for pair_idx, (us_ticker, jp_code, label) in enumerate(pair_list):
        if pair_idx % 50 == 0:
            _progress(f"CAR計算中... ({pair_idx}/{n_pairs})", 0.10 + 0.80 * pair_idx / n_pairs)

        if us_ticker not in us_events:
            continue

        event_idx = us_events[us_ticker]
        event_signs = us_event_signs[us_ticker]
        jp_rets = jp_ret[jp_code].values

        pair = PairResult(
            us_ticker=us_ticker,
            jp_code=jp_code,
            jp_name=(jp_code_names or {}).get(jp_code, jp_code),
            sector_label=label,
        )

        # 各イベントのCAR（符号調整済み）
        # USが上がった日 → JPも上がるなら正のCAR
        # USが下がった日 → JPも下がるなら正のCAR（符号反転して足す）
        car_by_lag: dict[int, list[float]] = {lag: [] for lag in lags}
        direction_hits = []

        for i, idx in enumerate(event_idx):
            if idx + max_lag >= len(jp_rets):
                continue

            sign = event_signs[i]

            for lag in lags:
                # t+1 から t+lag までの累積リターン
                cum_ret = np.nansum(jp_rets[idx + 1: idx + lag + 1])
                # 符号調整: USの方向に合わせる
                adjusted = cum_ret * sign
                car_by_lag[lag].append(adjusted)

            # 方向一致: t+1のリターンの符号
            if not np.isnan(jp_rets[idx + 1]):
                hit = 1.0 if np.sign(jp_rets[idx + 1]) == sign else 0.0
                direction_hits.append(hit)

        # 統計量計算（NaN除外で有効サンプルのみ使用）
        valid_lags = 0
        for lag in lags:
            cars = car_by_lag[lag]
            if not cars:
                continue
            arr = np.array(cars)
            finite_mask = np.isfinite(arr)
            arr_clean = arr[finite_mask]
            if len(arr_clean) < min_events:
                continue
            mean_car = float(np.mean(arr_clean))
            if np.std(arr_clean) < 1e-12:
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = stats.ttest_1samp(arr_clean, 0)
            if np.isfinite(p_val):
                pair.car[lag] = mean_car
                pair.car_t_stat[lag] = float(t_stat)
                pair.car_p_value[lag] = float(p_val)
                valid_lags += 1

        if valid_lags == 0:
            continue

        # n_eventsはpeak_lag確定後にそのラグの有効件数で設定（下で上書き）
        pair.n_events = 0
        pair.direction_hit_rate = float(np.mean(direction_hits)) if direction_hits else 0.0

        # ピーク: CARが最大のラグ
        if pair.car:
            peak_lag = max(pair.car, key=lambda k: abs(pair.car[k]))
            pair.peak_lag = peak_lag
            pair.peak_car = pair.car[peak_lag]
            # n_eventsはpeak_lagの有効件数
            peak_arr = np.array(car_by_lag.get(peak_lag, []))
            pair.n_events = int(np.sum(np.isfinite(peak_arr)))

        results.append(pair)

    # 有意なペアを抽出（1箇所にまとめ、FDR有無で分岐）
    _progress("有意ペアを抽出中...", 0.95)
    if fdr_correction and len(results) > 0:
        significant = _apply_fdr(results, lags, p_threshold)
    else:
        significant = []
        for pair in results:
            for lag in lags:
                if lag in pair.car_p_value and pair.car_p_value[lag] < p_threshold:
                    significant.append(pair)
                    break

    significant.sort(key=_score_pair, reverse=True)
    results.sort(key=_score_pair, reverse=True)

    _progress("完了", 1.0)

    return EventStudyResult(
        pairs=results,
        significant_pairs=significant,
        lags_tested=lags,
        event_threshold=event_threshold_sigma,
        isolated_only=isolated_only,
        n_us_stocks=len(us_events),
        n_jp_stocks=len(jp_ret.columns),
    )


def get_sp500_tickers(top_n: int = 500) -> list[str]:
    """S&P 500の主要ティッカーリストを返す。

    yfinanceで動的取得はせず、主要銘柄のハードコードリストを使う。
    時価総額上位から順に並べている。
    """
    # S&P 500 時価総額上位 (2025年基準)
    _SP500_TOP = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
        "TSLA", "UNH", "XOM", "V", "MA", "PG", "COST", "JNJ", "HD", "ABBV",
        "WMT", "NFLX", "BAC", "CRM", "AMD", "CVX", "MRK", "KO", "LIN", "PEP",
        "TMO", "ORCL", "ACN", "ADBE", "MCD", "CSCO", "ABT", "WFC", "PM", "GE",
        "IBM", "TXN", "ISRG", "QCOM", "INTU", "CAT", "GS", "MS", "VZ", "AXP",
        "AMGN", "DHR", "PFE", "BKNG", "T", "NEE", "RTX", "HON", "LOW", "UNP",
        "BLK", "SPGI", "COP", "SYK", "AMAT", "SCHW", "DE", "MDLZ", "LMT", "CB",
        "LRCX", "BA", "MMC", "GILD", "ADP", "BMY", "TJX", "VRTX", "C", "CI",
        "SO", "MO", "DUK", "CME", "INTC", "KLAC", "ICE", "PGR", "REGN", "SHW",
        "MU", "ZTS", "EQIX", "APD", "SNPS", "CL", "ITW", "SLB", "CDNS", "PH",
        "NOC", "USB", "PNC", "CMG", "MCK", "EMR", "GD", "TT", "BDX", "FDX",
        "MMM", "AON", "EOG", "CTAS", "PYPL", "APH", "TDG", "ECL", "FCX", "ABNB",
        "HUM", "GM", "AFL", "SPG", "NEM", "PSA", "FTNT", "TGT", "KMB", "MCO",
        "CARR", "MCHP", "ROP", "AIG", "D", "HCA", "MSI", "WELL", "KDP", "GIS",
        "F", "OXY", "DOW", "AEP", "MPC", "HAL", "PSX", "VLO", "NUE", "RIVN",
        "BHP", "KIE", "KBE", "KCE", "KRE",
    ]
    return _SP500_TOP[:top_n]


def get_jp_top_stocks(
    provider,
    n: int = 500,
    market_segments: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """J-Quantsから時価総額上位のJP銘柄コードを取得する。

    scale_categoryで絞り込み、上位N銘柄を返す。

    Returns:
        (codes, code_to_name_map)
    """
    listed = provider.get_listed_stocks()

    # ETF/REITを除外
    valid_markets = market_segments or ["プライム", "スタンダード"]
    filtered = listed[listed["market_name"].isin(valid_markets)]

    # scale_categoryで優先順位づけ
    scale_order = {
        "TOPIX Core30": 0,
        "TOPIX Large70": 1,
        "TOPIX Mid400": 2,
        "TOPIX Small 1": 3,
        "TOPIX Small 2": 4,
    }
    filtered = filtered.copy()
    filtered["_scale_rank"] = filtered["scale_category"].map(scale_order).fillna(5)
    filtered = filtered.sort_values("_scale_rank")

    codes = filtered["code"].head(n).tolist()
    names = dict(zip(filtered["code"], filtered["name"]))
    return codes, names


def build_summary_table(result: EventStudyResult, top_n: int = 30) -> pd.DataFrame:
    """結果を見やすいテーブルにまとめる。"""
    rows = []
    for pair in result.significant_pairs[:top_n]:
        row = {
            "セクター": pair.sector_label,
            "US銘柄": pair.us_ticker,
            "JP銘柄": pair.jp_name,
            "JPコード": pair.jp_code,
            "イベント数": pair.n_events,
            "方向一致率": pair.direction_hit_rate,
            "ピークラグ": f"t+{pair.peak_lag}",
            "ピークCAR": pair.peak_car,
        }
        # 各ラグのCAR
        for lag in result.lags_tested:
            row[f"CAR(t+{lag})"] = pair.car.get(lag, float("nan"))
            row[f"p(t+{lag})"] = pair.car_p_value.get(lag, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


def fetch_us_stock_returns(
    tickers: list[str],
    start_date: str,
    end_date: str,
    progress_callback=None,
) -> pd.DataFrame:
    """yfinanceでUS個別株のリターンを取得する。

    Returns:
        DataFrame (index=date, columns=ticker, values=daily return)
    """
    import yfinance as yf

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("US株価取得中...", 0.0)
    data = yf.download(
        tickers, start=start_date, end=end_date,
        auto_adjust=True, progress=False, threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    # Close prices → daily returns
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers[:1]

    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    returns = close.pct_change().dropna(how="all")
    _progress("US株価取得完了", 1.0)
    return returns


# =====================================================================
# 期間分割クロスバリデーション (案1)
# =====================================================================

@dataclass
class ValidatedPair:
    """前半/後半の両方で有意なペア"""
    us_ticker: str
    jp_code: str
    jp_name: str = ""
    sector_label: str = ""
    # 前半(発見)の結果
    discovery_car: dict[int, float] = field(default_factory=dict)
    discovery_p: dict[int, float] = field(default_factory=dict)
    discovery_hit_rate: float = 0.0
    discovery_n_events: int = 0
    # 後半(検証)の結果
    validation_car: dict[int, float] = field(default_factory=dict)
    validation_p: dict[int, float] = field(default_factory=dict)
    validation_hit_rate: float = 0.0
    validation_n_events: int = 0
    # 統合指標
    peak_lag: int = 0
    avg_car: float = 0.0  # 前半後半の平均CAR
    car_consistency: bool = False  # 前半後半でCARの符号が一致


@dataclass
class CrossValidatedResult:
    """期間分割クロスバリデーション結果"""
    validated_pairs: list[ValidatedPair] = field(default_factory=list)
    discovery_result: EventStudyResult | None = None
    validation_result: EventStudyResult | None = None
    split_date: str = ""
    n_discovery_significant: int = 0
    n_validated: int = 0
    reproduction_rate: float = 0.0  # 再現率
    lags_tested: list[int] = field(default_factory=list)


def run_cross_validated_scan(
    us_prices: pd.DataFrame,
    jp_prices: pd.DataFrame,
    sector_pairs: list[dict] | None = None,
    lags: list[int] | None = None,
    event_threshold_sigma: float = 2.0,
    min_events: int = 15,
    fdr_correction: bool = True,
    p_threshold: float = 0.05,
    jp_code_names: dict[str, str] | None = None,
    progress_callback=None,
) -> CrossValidatedResult:
    """期間分割クロスバリデーション付きペアスキャン。

    1. 全期間を前半(発見)と後半(検証)に50:50分割
    2. 前半でフルスキャン → FDR補正 → 有意ペア抽出
    3. 後半で同一ペア・同一パラメータで再テスト
    4. 両期間で有意なペアのみ「検証済み」として返す
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    # 共通日付で期間を分割
    common_dates = us_prices.index.intersection(jp_prices.index).sort_values()
    if len(common_dates) < 100:
        raise ValueError(f"共通日付が少なすぎます ({len(common_dates)}日)")

    mid = len(common_dates) // 2
    split_date = common_dates[mid]
    discovery_dates = common_dates[:mid]
    validation_dates = common_dates[mid:]

    _progress(f"期間分割: 発見={discovery_dates[0].strftime('%Y-%m-%d')}〜{discovery_dates[-1].strftime('%Y-%m-%d')}, "
              f"検証={validation_dates[0].strftime('%Y-%m-%d')}〜{validation_dates[-1].strftime('%Y-%m-%d')}", 0.05)

    # --- 前半: 発見 ---
    _progress("前半(発見)スキャン中...", 0.10)
    us_discovery = us_prices.loc[discovery_dates]
    jp_discovery = jp_prices.loc[discovery_dates]

    # min_eventsは期間長比率で調整（最低5を保証）
    discovery_ratio = len(discovery_dates) / len(common_dates)
    min_events_discovery = max(5, round(min_events * discovery_ratio))

    discovery_result = run_event_study(
        us_prices=us_discovery,
        jp_prices=jp_discovery,
        sector_pairs=sector_pairs,
        lags=lags,
        event_threshold_sigma=event_threshold_sigma,
        min_events=min_events_discovery,
        fdr_correction=fdr_correction,
        p_threshold=p_threshold,
        jp_code_names=jp_code_names,
        progress_callback=lambda m, p: _progress(m, 0.10 + p * 0.35),
    )

    n_discovery = len(discovery_result.significant_pairs)
    _progress(f"前半: {n_discovery}件の有意ペア発見", 0.50)

    if n_discovery == 0:
        _progress("前半で有意ペアなし。終了。", 1.0)
        return CrossValidatedResult(
            split_date=split_date.strftime("%Y-%m-%d"),
            discovery_result=discovery_result,
            n_discovery_significant=0,
            lags_tested=lags,
        )

    # --- 後半: 検証 (発見ペアのみ) ---
    _progress("後半(検証)スキャン中...", 0.55)
    us_validation = us_prices.loc[validation_dates]
    jp_validation = jp_prices.loc[validation_dates]

    # 発見ペアだけをsector_pairsとして渡す
    discovered_pairs_list = []
    for p in discovery_result.significant_pairs:
        discovered_pairs_list.append({
            "us_tickers": [p.us_ticker],
            "jp_codes": [p.jp_code],
            "label": p.sector_label,
        })

    validation_ratio = len(validation_dates) / len(common_dates)
    min_events_validation = max(5, round(min_events * validation_ratio))

    # 検証フェーズ: 計算は全ラグ行うが、検証通過判定は discovery の peak_lag のみで行う。
    # 各ペアにつき1ラグの単一仮説検定なのでFDR補正は不要。
    validation_result = run_event_study(
        us_prices=us_validation,
        jp_prices=jp_validation,
        sector_pairs=discovered_pairs_list,
        lags=lags,
        event_threshold_sigma=event_threshold_sigma,
        min_events=min_events_validation,
        fdr_correction=False,
        p_threshold=p_threshold,
        jp_code_names=jp_code_names,
        progress_callback=lambda m, p: _progress(m, 0.55 + p * 0.35),
    )

    _progress("検証済みペアを抽出中...", 0.95)

    # --- 突き合わせ: 両期間で有意なペア ---
    validation_map: dict[tuple[str, str], PairResult] = {}
    for p in validation_result.pairs:
        validation_map[(p.us_ticker, p.jp_code)] = p

    validated_pairs = []
    for disc_pair in discovery_result.significant_pairs:
        key = (disc_pair.us_ticker, disc_pair.jp_code)
        val_pair = validation_map.get(key)
        if val_pair is None:
            continue

        # 発見時のpeak_lagで検証判定（別ラグでの偶然一致を防ぐ）
        peak = disc_pair.peak_lag
        disc_car = disc_pair.car.get(peak, 0)
        val_car = val_pair.car.get(peak, 0)

        # 同一ラグで有意か
        val_p = val_pair.car_p_value.get(peak, 1.0)
        if val_p >= p_threshold:
            continue

        # 同方向か（符号一致を通過条件とする）
        car_consistent = (disc_car > 0 and val_car > 0) or (disc_car < 0 and val_car < 0)
        if not car_consistent:
            continue

        vp = ValidatedPair(
            us_ticker=disc_pair.us_ticker,
            jp_code=disc_pair.jp_code,
            jp_name=disc_pair.jp_name,
            sector_label=disc_pair.sector_label,
            discovery_car=disc_pair.car,
            discovery_p=disc_pair.car_p_value,
            discovery_hit_rate=disc_pair.direction_hit_rate,
            discovery_n_events=disc_pair.n_events,
            validation_car=val_pair.car,
            validation_p=val_pair.car_p_value,
            validation_hit_rate=val_pair.direction_hit_rate,
            validation_n_events=val_pair.n_events,
            peak_lag=peak,
            avg_car=(disc_car + val_car) / 2,
            car_consistency=car_consistent,
        )
        validated_pairs.append(vp)

    # CAR一致 & 平均CARでソート
    validated_pairs.sort(
        key=lambda v: (v.car_consistency, abs(v.avg_car)),
        reverse=True,
    )

    reproduction_rate = len(validated_pairs) / n_discovery if n_discovery > 0 else 0

    _progress(f"完了: {len(validated_pairs)}/{n_discovery}ペアが検証通過 (再現率{reproduction_rate:.0%})", 1.0)

    return CrossValidatedResult(
        validated_pairs=validated_pairs,
        discovery_result=discovery_result,
        validation_result=validation_result,
        split_date=split_date.strftime("%Y-%m-%d"),
        n_discovery_significant=n_discovery,
        n_validated=len(validated_pairs),
        reproduction_rate=reproduction_rate,
        lags_tested=lags,
    )


def build_cv_summary_table(result: CrossValidatedResult, top_n: int = 50) -> pd.DataFrame:
    """クロスバリデーション結果を表形式にまとめる。"""
    rows = []
    for vp in result.validated_pairs[:top_n]:
        row = {
            "US銘柄": vp.us_ticker,
            "JP銘柄": vp.jp_name,
            "ピークラグ": f"t+{vp.peak_lag}",
            "CAR一致": "Yes" if vp.car_consistency else "No",
            "平均CAR": vp.avg_car,
        }
        # 前半/後半の主要指標
        peak = vp.peak_lag
        row["発見CAR"] = vp.discovery_car.get(peak, float("nan"))
        row["発見p"] = vp.discovery_p.get(peak, float("nan"))
        row["発見一致率"] = vp.discovery_hit_rate
        row["発見イベント数"] = vp.discovery_n_events
        row["検証CAR"] = vp.validation_car.get(peak, float("nan"))
        row["検証p"] = vp.validation_p.get(peak, float("nan"))
        row["検証一致率"] = vp.validation_hit_rate
        row["検証イベント数"] = vp.validation_n_events
        rows.append(row)

    return pd.DataFrame(rows)
