"""分析計画作成・実行ページ（従来型の固定ロジック分析）"""

import json
from datetime import datetime, timedelta

import streamlit as st

from config import DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY, BACKTEST_DEFAULTS
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.idea_manager import IdeaManager
from core.planner import Planner, ANALYSIS_TEMPLATES
from core.analyzer import Analyzer
from core.backtester import Backtester
from core.evaluator import Evaluator
from core.knowledge_base import KnowledgeBase

st.set_page_config(page_title="分析実行", page_icon="🔬", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


def main():
    st.title("🔬 分析実行（従来型）")
    st.caption("固定ロジックによる統計分析・バックテスト")

    db = get_db()
    idea_mgr = IdeaManager(db)
    planner = Planner(db)

    # アイデア選択
    ideas = idea_mgr.list_all()
    if not ideas:
        st.warning("アイデアがありません。先に「アイデア管理」ページでアイデアを追加してください。")
        return

    idea_options = {f"[{i.category}] {i.title} (ID:{i.id})": i for i in ideas}
    selected_key = st.selectbox("分析対象のアイデア", list(idea_options.keys()))
    idea = idea_options[selected_key]

    st.info(f"**{idea.title}**: {idea.description}")

    st.divider()

    # --- 分析計画作成 ---
    st.subheader("分析計画の作成")

    template = planner.get_template(idea.category)
    st.write(f"**テンプレート**: {template['description']}")

    col1, col2 = st.columns(2)

    with col1:
        plan_name = st.text_input("計画名", value=f"{idea.title}_分析")
        universe = st.selectbox("ユニバース", ["all", "sector", "individual"])
        universe_detail = None
        if universe != "all":
            universe_detail = st.text_input("ユニバース詳細", placeholder="セクターコードまたは銘柄コード")
        start_date = st.date_input("開始日", value=datetime.now() - timedelta(days=365 * 5))
        end_date = st.date_input("終了日", value=datetime.now())

    with col2:
        st.write("**分析パラメータ**")
        params = {}
        for key, default in template["parameters"].items():
            if isinstance(default, int):
                params[key] = st.number_input(key, value=default, key=f"param_{key}")
            elif isinstance(default, float):
                params[key] = st.number_input(key, value=default, format="%.4f", key=f"param_{key}")
            elif isinstance(default, str):
                params[key] = st.text_input(key, value=default, key=f"param_{key}")
            else:
                params[key] = default

        rebalance = st.selectbox("リバランス頻度", ["daily", "weekly", "monthly"], index=2)

    # --- 計画保存・実行 ---
    st.divider()

    existing_plans = planner.list_plans(idea.id)
    if existing_plans:
        st.write(f"既存の計画: {len(existing_plans)}件")
        for p in existing_plans:
            st.write(f"  - {p.name} ({p.status})")

    if st.button("計画を保存して分析を実行", type="primary", use_container_width=True):
        # 計画保存
        plan = planner.create_plan(
            idea_id=idea.id,
            name=plan_name,
            category=idea.category,
            universe=universe,
            universe_detail=universe_detail,
            start_date=str(start_date),
            end_date=str(end_date),
            parameters=params,
            backtest_config={"rebalance_frequency": rebalance, **BACKTEST_DEFAULTS},
        )
        planner.update(plan.id, status="running")
        idea_mgr.update(idea.id, status="active")

        # Run作成
        idea_snap = db.get_idea(idea.id)
        plan_snap = db.get_plan(plan.id)
        run_id = db.create_run(plan.id, idea_snap, plan_snap)

        provider = get_provider()

        with st.status("分析を実行中...", expanded=True) as status:
            try:
                # データ取得
                st.write("📥 データを取得中...")
                if provider.is_available():
                    prices = provider.get_price_daily(
                        code=universe_detail if universe == "individual" else None,
                        start_date=str(start_date),
                        end_date=str(end_date),
                    )
                    index_prices = provider.get_index_prices(
                        start_date=str(start_date),
                        end_date=str(end_date),
                    )
                else:
                    st.warning("J-Quants API未接続のため、ダミーデータは使用できません。")
                    db.update_run(run_id, status="failed",
                                  evaluation={"error": "API未接続"},
                                  finished_at=datetime.now().isoformat())
                    status.update(label="エラー", state="error")
                    return

                # 統計分析
                st.write("📊 統計分析中...")
                analyzer = Analyzer()
                stats_result = analyzer.analyze(prices, plan.analysis_method, params, index_prices)

                # バックテスト
                st.write("📈 バックテスト中...")
                backtester = Backtester(**{
                    k: v for k, v in BACKTEST_DEFAULTS.items()
                    if k in ("initial_capital", "commission_rate", "slippage_rate")
                })
                signals = backtester.generate_signals_from_analysis(prices, plan.analysis_method, params)
                bt_result = backtester.run(prices, signals, index_prices, rebalance)

                # 評価
                st.write("🔍 結果を評価中...")
                evaluator = Evaluator()
                evaluation = evaluator.evaluate(stats_result, bt_result)

                # 保存
                db.update_run(
                    run_id,
                    statistics_result=stats_result,
                    backtest_result=bt_result,
                    evaluation=evaluation,
                    evaluation_label=evaluation["label"],
                    status="completed",
                    finished_at=datetime.now().isoformat(),
                )

                # 知見保存
                st.write("💾 知見を保存中...")
                kb = KnowledgeBase(db)
                kb.save_from_run(
                    run_id=run_id,
                    hypothesis=idea.description,
                    evaluation=evaluation,
                    tags=[idea.category],
                )

                planner.update(plan.id, status="completed")
                idea_mgr.update(idea.id, status="completed")

                status.update(label="分析完了!", state="complete")
                st.success(f"分析が完了しました (Run #{run_id}, 評価: {evaluation['label']})")

            except Exception as e:
                db.update_run(run_id, status="failed",
                              evaluation={"error": str(e)},
                              finished_at=datetime.now().isoformat())
                status.update(label="エラー", state="error")
                st.error(f"分析中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
