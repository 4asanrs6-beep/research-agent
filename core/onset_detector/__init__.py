"""Onset State Detection System — 初動状態検出システム

Phase 1: スター株共通点発見 + 追加スター株発見 + 初動特定 + AI解釈
Phase 2 (TBD): 次のスター株予測
"""

from .config import OnsetDetectorConfig
from .scanner import ScanResult

__all__ = [
    "OnsetDetectorConfig",
    "ScanResult",
    "run_full_pipeline",
    "run_phase1_discovery",
]


def run_full_pipeline(
    data_provider,
    config: OnsetDetectorConfig | None = None,
    progress_callback=None,
):
    """学習→評価→スキャンの完全パイプラインを実行する。

    Parameters
    ----------
    data_provider : JQuantsProvider
        市場データプロバイダ
    config : OnsetDetectorConfig | None
        設定。Noneならデフォルト
    progress_callback : callable | None
        (step, total, message) を受け取るコールバック

    Returns
    -------
    dict with keys:
        model: TwoStageOnsetModel (trained)
        eval_result: dict (Track A/B metrics)
        scan_results: list[ScanResult]
        best_event_params: (excess_threshold, horizon)
    """
    if config is None:
        config = OnsetDetectorConfig()

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    # --- Step 1: データ取得 ---
    _progress(1, 8, "市場データ取得中（月単位チャンク）...")
    from .labeler import OnsetLabeler
    labeler = OnsetLabeler(data_provider, config)
    all_prices, topix, listed_stocks = labeler.fetch_market_data(
        progress_callback=lambda msg: _progress(1, 8, msg),
    )

    # --- Step 2: スター株検出 ---
    _progress(2, 8, "スター株検出中...")
    star_stocks = labeler.identify_star_stocks(all_prices, topix, listed_stocks)

    if len(star_stocks) < 5:
        return {
            "error": f"スター株が{len(star_stocks)}件しか見つかりません（最低5件必要）",
            "star_stocks": star_stocks,
        }

    # --- Step 3: 類型発見 ---
    _progress(3, 8, "スター株類型分析中...")
    star_types = labeler.discover_star_types(star_stocks, all_prices, topix)

    # --- Step 4: 学習データ構築 ---
    _progress(4, 8, "学習データ構築中（first-hit計算）...")
    dataset = labeler.build_training_dataset(
        star_stocks, star_types, all_prices, topix, listed_stocks
    )

    # --- Step 5: 特徴量計算 ---
    _progress(5, 8, "特徴量計算中（~200特徴量）...")
    from .features import TemporalFeatureEngine
    engine = TemporalFeatureEngine(config)

    def _batch_progress(processed, total, elapsed):
        if elapsed > 0 and processed > 0:
            remaining = elapsed / processed * (total - processed)
            _progress(5, 8, f"特徴量計算中 ({processed}/{total}銘柄, 残り約{int(remaining)}秒)")
        else:
            _progress(5, 8, f"特徴量計算中 ({processed}/{total}銘柄)")

    X_train, feature_names = engine.compute_features_batch(
        dataset, all_prices, topix, listed_stocks,
        progress_callback=_batch_progress,
    )

    # --- Step 6: モデル学習 ---
    _progress(6, 8, "二段階モデル学習中（ネストCV）...")
    from .model import TwoStageOnsetModel
    model = TwoStageOnsetModel(config)
    eval_result = model.train(X_train, dataset, feature_names)

    # --- Step 7: スキャン ---
    _progress(7, 8, "全銘柄スキャン中...")
    from .scanner import OnsetScanner
    scanner = OnsetScanner(model, engine, config)

    def _scan_progress(processed, total, elapsed, phase=None):
        if phase == "market_cap":
            _progress(7, 8, f"時価総額取得中 ({processed}/{total}銘柄)")
        elif elapsed > 0 and processed > 0:
            remaining = elapsed / processed * (total - processed)
            _progress(7, 8, f"スキャン中 ({processed}/{total}銘柄, 残り約{int(remaining)}秒)")
        else:
            _progress(7, 8, f"スキャン中 ({processed}/{total}銘柄)")

    scan_results = scanner.scan(
        all_prices, topix, listed_stocks,
        progress_callback=_scan_progress,
    )

    # --- Step 8: 完了 ---
    _progress(8, 8, "完了")

    return {
        "model": model,
        "eval_result": eval_result,
        "scan_results": scan_results,
        "best_event_params": (model.best_excess_threshold, model.best_horizon),
        "star_stocks": star_stocks,
        "star_types": star_types,
        "dataset_stats": {
            "n_stage1_samples": len(dataset["y1"]),
            "n_stage1_positive": int(sum(dataset["y1"])),
            "n_stage2_samples": len(dataset["y2"]),
            "n_features": len(feature_names),
        },
    }


def run_phase1_discovery(
    data_provider,
    config: OnsetDetectorConfig | None = None,
    progress_callback=None,
):
    """Phase 1: スター株共通点発見 + 追加発見 + 初動特定

    Parameters
    ----------
    data_provider : JQuantsProvider
        市場データプロバイダ
    config : OnsetDetectorConfig | None
        設定。Noneならデフォルト
    progress_callback : callable | None
        (step, total, message) を受け取るコールバック

    Returns
    -------
    dict with keys:
        common_features: 共通特徴量発見結果
        additional_stars: 追加発見スター株
        onset_dates: 各スター株の初動情報
        all_stars: 全スター株リスト
        ai_interpretation: AI解釈テキスト
        star_stocks: 入力スター株
        warnings: 警告リスト
    """
    import logging
    logger = logging.getLogger(__name__)

    if config is None:
        config = OnsetDetectorConfig()

    total_steps = 8
    warnings = []

    def _progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    # --- Step 1: データ取得 + 時価総額フィルタ ---
    _progress(1, total_steps, "市場データ取得中...")
    from .labeler import OnsetLabeler
    labeler = OnsetLabeler(data_provider, config)
    all_prices, topix, listed_stocks = labeler.fetch_market_data(
        progress_callback=lambda msg: _progress(1, total_steps, msg),
    )

    # ETF・REIT除外: listed_stocks（一般株式のみ）のコードでall_pricesを絞る
    stock_codes = set(str(c) for c in listed_stocks["code"].unique())
    n_before_etf = len(all_prices["code"].unique())
    all_prices = all_prices[all_prices["code"].apply(lambda c: str(c) in stock_codes)]
    n_after_etf = len(all_prices["code"].unique())
    if n_before_etf != n_after_etf:
        logger.info(
            f"ETF・REIT除外: {n_before_etf} → {n_after_etf}銘柄 "
            f"({n_before_etf - n_after_etf}件除外)"
        )

    # 信用取引データ取得・マージ
    if config.use_margin_features:
        _progress(1, total_steps, "信用取引データ取得中...")
        try:
            start_date = config.start_date or str(
                __import__("datetime").date.today()
                - __import__("datetime").timedelta(days=365)
            )
            end_date = config.end_date or str(__import__("datetime").date.today())
            margin_df = labeler._fetch_margin_data_bulk(
                start_date, end_date,
                _prog=lambda msg: _progress(1, total_steps, msg),
            )
            if margin_df is not None and not margin_df.empty:
                all_prices = OnsetLabeler.merge_margin_into_prices(all_prices, margin_df)
                logger.info(f"信用取引データ統合完了: {len(margin_df)}行")
            else:
                warnings.append("信用取引データが空でした（信用特徴量は0.0にフォールバック）")
        except Exception as e:
            logger.warning(f"信用取引データ取得失敗: {e}")
            warnings.append(f"信用取引データ取得失敗: {e}（信用特徴量は0.0にフォールバック）")

    # 時価総額フィルタ（yfinance実値、ユーザー指定スター株は保護）
    if config.scan_min_market_cap > 0:
        _progress(1, total_steps, "時価総額フィルタ適用中...")
        from .scanner import fetch_market_caps
        all_codes = [str(c) for c in all_prices["code"].unique()]

        # ユーザー指定スター株は除外対象外
        user_codes_5 = set()
        for uc in config.user_star_codes:
            user_codes_5.add(uc if len(uc) >= 5 else uc + "0")

        def _mcap_progress(done, total, msg):
            _progress(1, total_steps, f"時価総額取得中 ({done}/{total})")

        market_caps = fetch_market_caps(all_codes, progress_callback=_mcap_progress)
        min_cap_yen = config.scan_min_market_cap * 1e8  # 億円→円
        # cap=0は取得失敗→除外しない。確認済み低時価総額のみ除外（スター株除く）
        exclude_codes = {
            c for c, cap in market_caps.items()
            if 0 < cap < min_cap_yen and c not in user_codes_5
        }
        valid_codes = {c for c in all_codes if c not in exclude_codes}

        n_before = len(all_codes)
        all_prices = all_prices[
            all_prices["code"].apply(lambda c: str(c) in valid_codes)
        ]
        listed_stocks = listed_stocks[
            listed_stocks["code"].apply(lambda c: str(c) in valid_codes)
        ]
        n_after = len(all_prices["code"].unique())
        logger.info(
            f"時価総額フィルタ: {n_before} → {n_after}銘柄 "
            f"(除外: {len(exclude_codes)}件, 下限: {config.scan_min_market_cap:.0f}億円)"
        )

    # --- Step 2: スター株解決 ---
    _progress(2, total_steps, "スター株解決中...")
    user_codes = config.user_star_codes

    if not user_codes:
        return {
            "error": "スター株コードが指定されていません。サイドバーで銘柄コードを入力してください。",
            "warnings": warnings,
        }

    star_stocks = labeler._resolve_user_codes(
        user_codes, all_prices, topix, listed_stocks,
    )

    # 解決できなかった銘柄をチェック
    resolved_codes = {str(s["code"]) for s in star_stocks}
    for uc in user_codes:
        normalized = uc if len(uc) >= 5 else uc + "0"
        if normalized not in resolved_codes:
            warnings.append(f"銘柄 {uc} はデータ不足により除外されました")

    if len(star_stocks) < 3:
        return {
            "error": f"有効なスター株が{len(star_stocks)}件のみです（最低3件必要）",
            "star_stocks": star_stocks,
            "warnings": warnings,
        }

    logger.info(f"スター株: {len(star_stocks)}件解決")

    # --- Step 3-6: OnsetDiscoverer ---
    from .discoverer import OnsetDiscoverer
    discoverer = OnsetDiscoverer(config)

    step_map = {
        "共通特徴量発見中...": 3,
        "追加スター株探索中...": 4,
        "母集団確率計算中...": 5,
        "初動日特定中...": 6,
        "AI解釈生成中...": 7,
    }

    def _discovery_progress(msg):
        for prefix, step in step_map.items():
            if msg.startswith(prefix.rstrip("...")):
                _progress(step, total_steps, msg)
                return
        # デフォルト
        _progress(3, total_steps, msg)

    result = discoverer.run(
        user_star_codes=user_codes,
        star_stocks=star_stocks,
        all_prices=all_prices,
        topix=topix,
        listed_stocks=listed_stocks,
        progress_callback=_discovery_progress,
    )

    # --- Step 8: 完了 ---
    _progress(total_steps, total_steps, "完了")

    result["star_stocks"] = star_stocks
    result["warnings"] = warnings
    return result
