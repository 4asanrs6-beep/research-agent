"""Onset Detector v2 設定"""

from dataclasses import dataclass, field


@dataclass
class OnsetDetectorConfig:
    """二段階TTEモデルの全設定"""

    # --- イベント定義（探索対象） ---
    excess_threshold: float = 0.30       # first-hit超過リターン閾値
    horizon: int = 60                    # first-hit到達判定地平（日数）
    excess_threshold_grid: list[float] = field(
        default_factory=lambda: [0.20, 0.30, 0.40]
    )
    horizon_grid: list[int] = field(
        default_factory=lambda: [40, 60, 80]
    )

    # --- Stage 1: 近傍イベント発生分類 ---
    T_near: int = 30                     # 近傍判定カットオフ（日数）
    p_event_min: float = 0.10            # 推論時スクリーニング閾値

    # --- Stage 2: TTE回帰 ---
    tau_range: tuple[int, int] = (15, 30)  # τ探索範囲

    # --- 過熱フィルタ ---
    overheat_trailing_20d_excess: float = 0.10  # 20日超過リターン上限
    overheat_trailing_5d_return: float = 0.05   # 5日リターン上限
    overheat_rsi_14: float = 70.0               # RSI(14)上限

    # --- 分析期間 ---
    start_date: str = ""                 # YYYY-MM-DD（空なら1年前）
    end_date: str = ""                   # YYYY-MM-DD（空なら今日）

    # --- スター株入力 ---
    star_stocks_input: list[dict] | None = None  # 外部から注入（ページ4結果等）
    user_star_codes: list[str] = field(default_factory=list)  # 手動指定コード

    # --- スター株自動検出 ---
    star_min_total_return: float = 0.50
    star_min_excess_return: float = 0.30
    star_min_volume_ratio: float = 1.5
    star_max_auto_detect: int = 50
    # 仕手株フィルタ
    star_min_market_cap_billion: float = 50.0
    star_max_drawdown: float = 0.40
    star_max_single_day_return: float = 0.20
    star_min_up_days_ratio: float = 0.45

    # --- CV ---
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    embargo_multiplier: float = 1.0      # embargo_days = horizon × multiplier

    # --- LightGBM Stage 1 ---
    stage1_num_leaves: int = 31
    stage1_min_child_samples: int = 20
    stage1_learning_rate: float = 0.05
    stage1_feature_fraction: float = 0.8
    stage1_bagging_fraction: float = 0.8
    stage1_n_estimators: int = 500

    # --- LightGBM Stage 2 ---
    stage2_num_leaves: int = 31
    stage2_min_child_samples: int = 10
    stage2_learning_rate: float = 0.05
    stage2_feature_fraction: float = 0.8
    stage2_n_estimators: int = 300

    # --- クラスタリング ---
    n_star_types: int = 4

    # --- 制御サンプル ---
    n_control_per_type: int = 3          # 非スター制御群の種類数
    control_sample_ratio: float = 3.0    # スターに対する制御群倍率

    # --- レンジ分類 ---
    range_early_min: int = 15
    range_early_max: int = 30
    range_imminent_min: int = 5
    range_imminent_max: int = 14

    # --- 特徴量 ---
    feature_history_days: int = 120      # 特徴量計算に必要な履歴日数
    zscore_window: int = 120             # 自己z-score窓

    # --- Phase 1: 共通特徴量発見 ---
    discovery_min_precision: float = 0.15    # コンボの最低精度
    discovery_min_recall: float = 0.30       # コンボの最低再現率
    discovery_max_additional_stars: int = 30  # 追加スター株上限
    discovery_neg_sample_size: int = 200     # 負例サンプル数（per onset date）
    discovery_max_iterations: int = 5        # 反復改善の最大回数
    onset_min_signal_score: int = 3          # 初動判定の最低シグナル数
    onset_min_forward_return: float = 0.10   # 初動候補の最低前方リターン
    onset_max_candidates: int = 5            # 銘柄あたり初動候補上限
    use_ai_interpretation: bool = True       # Claude CLI解釈を有効化
    use_margin_features: bool = True         # 信用取引特徴量を使用

    # --- スキャン ---
    top_k: int = 50                      # 最終出力上位K
    scan_min_market_cap: float = 300.0   # スキャン対象の最低時価総額（億円）
    scan_max_workers: int = 4            # 並列ワーカー数

    @property
    def embargo_days(self) -> int:
        return int(self.horizon * self.embargo_multiplier)

    def get_event_grid(self) -> list[tuple[float, int]]:
        """excess_threshold × horizon の全組み合わせを返す"""
        return [
            (et, h)
            for et in self.excess_threshold_grid
            for h in self.horizon_grid
        ]
