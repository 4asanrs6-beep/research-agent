"""Lead-Lag特徴量のQlib統合 (Phase 5)

PCAシグナルをAlpha158に追加する特徴量として生成する。
交互作用特徴量（× sector_beta, × market_cap_bucket）を含む。
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lead-Lag追加特徴量の名前 (Qlib内で使う)
LEADLAG_FEATURE_NAMES = [
    # 基本特徴量
    "ll_sector_signal",        # 所属セクターのPCA_SUBシグナル
    "ll_sector_signal_rank",   # シグナルの全セクター内順位 (0-1)
    "ll_sector_signal_abs",    # シグナル絶対値 (強度)
    "ll_us_sector_ret",        # 対応USセクターETFの前日リターン
    "ll_gap_flag",             # ギャップフィルター該当フラグ (0/1)
    # 交互作用特徴量
    "ll_signal_x_beta",        # sector_signal × sector_beta
    "ll_signal_x_cap_bucket",  # sector_signal × market_cap_bucket
    "ll_signal_x_momentum",    # sector_signal × 20日モメンタム
]


def generate_leadlag_features(
    signals_df: pd.DataFrame,
    stock_sectors: dict[str, str],
    stock_prices: pd.DataFrame,
    us_cc_returns: pd.DataFrame | None = None,
    jp_overnight_gaps: pd.DataFrame | None = None,
    etf_prices: pd.DataFrame | None = None,
    market_caps: pd.DataFrame | None = None,
    beta_window: int = 60,
    progress_callback=None,
) -> pd.DataFrame:
    """Lead-Lag特徴量を個別株レベルで生成する。

    Args:
        signals_df: PCA_SUB.signals (index=date, columns=JP ETF codes)
        stock_sectors: {stock_code: etf_code} マッピング
        stock_prices: DataFrame[date, code, adj_close]
        us_cc_returns: USセクターETF日次リターン (index=date, columns=US tickers)
        jp_overnight_gaps: JPセクターETFのオーバーナイトギャップ
        etf_prices: JPセクターETF価格 (index=date, columns=JP ETF codes)
        market_caps: 時価総額 (index=date, columns=stock codes) [任意]
        beta_window: ベータ推定ウィンドウ

    Returns:
        DataFrame[date, code, ll_sector_signal, ll_sector_signal_rank, ...]
        Qlib学習前にAlpha158特徴量とjoinして使う
    """
    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("Lead-Lag特徴量を生成中...", 0.0)

    # --- 基本: セクターシグナルの配賦 ---
    rows = []
    dates = signals_df.index

    # セクターETFリターンを計算 (ベータ推定用)
    etf_returns = None
    if etf_prices is not None:
        etf_returns = etf_prices.pct_change()

    # 個別株リターンを計算 (ベータ/モメンタム用)
    stock_ret_pivot = None
    if not stock_prices.empty:
        close_col = "adj_close" if "adj_close" in stock_prices.columns else "close"
        sp = stock_prices.pivot_table(index="date", columns="code", values=close_col)
        stock_ret_pivot = sp.pct_change()

    for t_idx, dt in enumerate(dates):
        if t_idx % 100 == 0:
            _progress(f"特徴量生成中... ({t_idx}/{len(dates)})", t_idx / max(1, len(dates)))

        sig_row = signals_df.loc[dt]
        sig_valid = sig_row.dropna()

        if len(sig_valid) == 0:
            continue

        # セクターランク (0-1正規化)
        sig_rank = sig_valid.rank(pct=True)

        for code, etf_code in stock_sectors.items():
            if etf_code not in sig_valid.index:
                continue

            signal_val = sig_valid[etf_code]
            signal_rank = sig_rank[etf_code]

            feat = {
                "date": dt,
                "code": code,
                "ll_sector_signal": signal_val,
                "ll_sector_signal_rank": signal_rank,
                "ll_sector_signal_abs": abs(signal_val),
            }

            # US セクターリターン
            if us_cc_returns is not None and dt in us_cc_returns.index:
                # ETF code → US対応セクターのマッピングは直接使えないが、
                # 全USセクターの平均リターンを代用
                feat["ll_us_sector_ret"] = us_cc_returns.loc[dt].mean()
            else:
                feat["ll_us_sector_ret"] = 0.0

            # ギャップフラグ
            if jp_overnight_gaps is not None and dt in jp_overnight_gaps.index:
                gap_row = jp_overnight_gaps.loc[dt]
                if etf_code in gap_row.index:
                    gap = gap_row[etf_code]
                    feat["ll_gap_flag"] = 1.0 if abs(gap) > abs(signal_val) * 0.1 else 0.0
                else:
                    feat["ll_gap_flag"] = 0.0
            else:
                feat["ll_gap_flag"] = 0.0

            # --- 交互作用特徴量 ---

            # sector_beta: 個別株のセクターETFに対するベータ
            beta = _compute_rolling_beta(
                stock_ret_pivot, etf_returns, code, etf_code, dt, beta_window
            )
            feat["ll_signal_x_beta"] = signal_val * beta if not np.isnan(beta) else 0.0

            # market_cap_bucket: 時価総額バケット (1-5)
            cap_bucket = 3.0  # デフォルト中央値
            if market_caps is not None and code in market_caps.columns and dt in market_caps.index:
                cap = market_caps.loc[dt, code]
                if not np.isnan(cap) and cap > 0:
                    # 全銘柄の中での5分位
                    day_caps = market_caps.loc[dt].dropna()
                    if len(day_caps) > 10:
                        cap_bucket = float(pd.Series([cap]).apply(
                            lambda x: pd.qcut(day_caps, 5, labels=False, duplicates="drop").get(code, 3)
                        ).iloc[0]) + 1
            feat["ll_signal_x_cap_bucket"] = signal_val * cap_bucket

            # momentum: 過去20日リターン
            mom = _compute_momentum(stock_ret_pivot, code, dt, 20)
            feat["ll_signal_x_momentum"] = signal_val * mom if not np.isnan(mom) else 0.0

            rows.append(feat)

    _progress("完了", 1.0)

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["date"] = pd.to_datetime(result["date"])
        logger.info(
            "Lead-Lag特徴量: %d行 (%d日 × 平均%.0f銘柄)",
            len(result), result["date"].nunique(),
            len(result) / max(1, result["date"].nunique()),
        )
    return result


def _compute_rolling_beta(
    stock_returns: pd.DataFrame | None,
    etf_returns: pd.DataFrame | None,
    code: str,
    etf_code: str,
    dt,
    window: int,
) -> float:
    """ローリングベータを計算する。"""
    if stock_returns is None or etf_returns is None:
        return 1.0

    if code not in stock_returns.columns or etf_code not in etf_returns.columns:
        return 1.0

    try:
        dt_loc = stock_returns.index.get_loc(dt)
    except KeyError:
        return 1.0

    if dt_loc < window:
        return 1.0

    sr = stock_returns.iloc[dt_loc - window:dt_loc][code].values
    er = etf_returns.iloc[dt_loc - window:dt_loc][etf_code].values

    mask = ~(np.isnan(sr) | np.isnan(er))
    if mask.sum() < window // 2:
        return 1.0

    sr_clean, er_clean = sr[mask], er[mask]
    cov = np.cov(sr_clean, er_clean)
    if cov.shape == (2, 2) and cov[1, 1] > 1e-12:
        return float(cov[0, 1] / cov[1, 1])
    return 1.0


def _compute_momentum(
    stock_returns: pd.DataFrame | None,
    code: str,
    dt,
    lookback: int,
) -> float:
    """過去N日の累積リターンを計算する。"""
    if stock_returns is None or code not in stock_returns.columns:
        return 0.0

    try:
        dt_loc = stock_returns.index.get_loc(dt)
    except KeyError:
        return 0.0

    if dt_loc < lookback:
        return 0.0

    rets = stock_returns.iloc[dt_loc - lookback:dt_loc][code].values
    rets = np.where(np.isnan(rets), 0, rets)
    return float(np.sum(rets))


def inject_features_to_qlib_data(
    features_df: pd.DataFrame,
    qlib_data_dir: Path,
) -> dict[str, Path]:
    """生成したLead-Lag特徴量をQlibバイナリ形式で書き出す。

    Alpha158とは別に、追加特徴量として学習時にjoinする。
    Qlibの通常のDataHandlerでは外部特徴量の注入が難しいため、
    学習前のDataFrame結合で対応する。

    Args:
        features_df: generate_leadlag_features() の結果
        qlib_data_dir: Qlibデータディレクトリ

    Returns:
        保存先パスの辞書 {"features_parquet": Path}
    """
    output_dir = qlib_data_dir / "leadlag_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "features.parquet"
    features_df.to_parquet(output_path, index=False)

    logger.info("Lead-Lag特徴量を保存: %s (%d行)", output_path, len(features_df))
    return {"features_parquet": output_path}


def run_experiment_with_leadlag(
    leadlag_features: pd.DataFrame,
    instrument_codes: list[str] | None = None,
    train_period: tuple[str, str] | None = None,
    valid_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
    on_progress=None,
) -> dict:
    """Alpha158 + Lead-Lag特徴量でQlib実験を実行する。

    Qlibの標準ワークフローにLead-Lag特徴量をマージして学習する。
    Alpha158の158特徴量 + Lead-Lag特徴量(8個) = 166特徴量。

    Args:
        leadlag_features: generate_leadlag_features() の結果
        instrument_codes: フィルタ済み銘柄コード
        train/valid/test_period: 学習/検証/テスト期間

    Returns:
        Qlib実験結果 dict (workflow_runner.run_experiment と同形式)
    """
    from qlib_integration.workflow_runner import init_qlib_jp, _write_filtered_instruments
    from qlib_integration.config import (
        QLIB_DATA_DIR, TRAIN_PERIOD, VALID_PERIOD, TEST_PERIOD,
        LGB_DEFAULT_PARAMS,
    )

    if train_period is None:
        train_period = TRAIN_PERIOD
    if valid_period is None:
        valid_period = VALID_PERIOD
    if test_period is None:
        test_period = TEST_PERIOD

    def progress(msg, pct):
        logger.info("[%.0f%%] %s", pct * 100, msg)
        if on_progress:
            on_progress(msg, pct)

    progress("Qlib初期化中...", 0.0)
    init_qlib_jp()

    # 銘柄フィルタ
    if instrument_codes:
        instruments_name = _write_filtered_instruments(QLIB_DATA_DIR, instrument_codes)
    else:
        instruments_name = "all"

    # Alpha158 DataHandler構築
    progress("Alpha158 DataHandler構築中...", 0.05)
    from qlib_integration.workflow_runner import _build_alpha158_jp_handler
    handler = _build_alpha158_jp_handler(
        instruments=instruments_name,
        start_time=train_period[0],
        end_time=test_period[1],
        fit_start_time=train_period[0],
        fit_end_time=train_period[1],
    )

    # Dataset構築
    progress("Dataset構築中...", 0.15)
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": train_period,
            "valid": valid_period,
            "test": test_period,
        },
    )

    # Alpha158特徴量を取得してLead-Lag特徴量をマージ
    progress("Lead-Lag特徴量をマージ中...", 0.20)
    try:
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        valid_data = dataset.prepare("valid", col_set=["feature", "label"])
        test_data = dataset.prepare("test", col_set=["feature", "label"])
    except Exception as e:
        logger.error("Dataset準備エラー: %s", e)
        raise

    # Lead-Lag特徴量をMultiIndex形式に変換
    ll_feat = leadlag_features.copy()
    ll_feat["date"] = pd.to_datetime(ll_feat["date"])
    ll_feat = ll_feat.set_index(["date", "code"])

    feature_cols = [c for c in LEADLAG_FEATURE_NAMES if c in ll_feat.columns]
    ll_subset = ll_feat[feature_cols]

    # マージ関数
    def _merge_features(alpha_data, ll_data):
        """Alpha158 DataFrameにLead-Lag特徴量をマージする。"""
        if isinstance(alpha_data, pd.DataFrame) and isinstance(alpha_data.index, pd.MultiIndex):
            # MultiIndex (datetime, instrument)
            common = alpha_data.index.intersection(ll_data.index)
            if len(common) > 0:
                merged = alpha_data.copy()
                for col in ll_data.columns:
                    merged[col] = ll_data.reindex(merged.index)[col].fillna(0)
                return merged
        return alpha_data

    # 各セグメントをマージ
    train_merged = _merge_features(train_data, ll_subset)
    valid_merged = _merge_features(valid_data, ll_subset)
    test_merged = _merge_features(test_data, ll_subset)

    # LightGBMで学習
    progress("LightGBMモデル学習中...", 0.30)
    import lightgbm as lgb

    def _split_xy(df):
        label_col = [c for c in df.columns if "LABEL" in str(c).upper()]
        if not label_col:
            label_col = df.columns[-1:]
        features = df.drop(columns=label_col)
        labels = df[label_col[0]] if len(label_col) == 1 else df[label_col]
        return features, labels

    X_train, y_train = _split_xy(train_merged)
    X_valid, y_valid = _split_xy(valid_merged)
    X_test, y_test = _split_xy(test_merged)

    params = dict(LGB_DEFAULT_PARAMS)
    params.update({
        "verbosity": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    })

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        callbacks=[lgb.log_evaluation(period=100)],
    )

    progress("予測実行中...", 0.70)
    pred = pd.Series(model.predict(X_test), index=X_test.index)

    # IC計算
    progress("評価指標計算中...", 0.85)
    from qlib_integration.workflow_runner import _compute_ic_metrics
    # 簡易IC計算
    if isinstance(X_test.index, pd.MultiIndex):
        dates_test = X_test.index.get_level_values(0).unique()
        ics = []
        for dt in dates_test:
            try:
                p = pred.loc[dt]
                l = y_test.loc[dt]
                if len(p) < 5:
                    continue
                ic = p.corr(l)
                if not np.isnan(ic):
                    ics.append(ic)
            except Exception:
                continue
        ic_mean = float(np.mean(ics)) if ics else float("nan")
        ic_std = float(np.std(ics)) if len(ics) > 1 else float("nan")
        icir = ic_mean / ic_std if ic_std > 1e-10 else float("nan")
    else:
        ic_mean = float(pred.corr(y_test))
        icir = float("nan")

    # 特徴量重要度
    importance = model.feature_importance(importance_type="gain")
    feature_names = X_train.columns.tolist()
    imp_dict = dict(zip(feature_names, importance.tolist()))
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:30]

    # Lead-Lag特徴量の重要度ランク
    ll_importance = {k: v for k, v in imp_dict.items() if k.startswith("ll_")}
    ll_rank = {k: i + 1 for i, (k, _) in enumerate(sorted_imp) if k.startswith("ll_")}

    progress("完了", 1.0)

    return {
        "model_type": "lgb+leadlag",
        "train_period": train_period,
        "valid_period": valid_period,
        "test_period": test_period,
        "n_features_alpha158": len(X_train.columns) - len(feature_cols),
        "n_features_leadlag": len(feature_cols),
        "n_features_total": len(X_train.columns),
        "ic_mean": ic_mean,
        "icir": icir,
        "n_test_samples": len(X_test),
        "feature_importance": dict(sorted_imp),
        "leadlag_importance": ll_importance,
        "leadlag_importance_rank": ll_rank,
        "predictions": pred,
    }
