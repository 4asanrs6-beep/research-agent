"""AI研究ページ - AIエージェントによる自動研究実行"""

import json
import time

import plotly.graph_objects as go
import streamlit as st

from config import (
    DB_PATH, MARKET_DATA_DIR, ANALYSIS_CATEGORIES,
    JQUANTS_API_KEY, AI_API_KEY,
)
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.ai_client import create_ai_client
from core.ai_researcher import AiResearcher, ResearchProgress

st.set_page_config(page_title="AI研究", page_icon="🤖", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


def plot_equity_curve(backtest: dict) -> go.Figure | None:
    equity = backtest.get("equity_curve", [])
    if not equity:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[e["date"] for e in equity],
        y=[e["value"] for e in equity],
        name="ポートフォリオ",
        line=dict(color="#1f77b4"),
    ))
    bench = backtest.get("benchmark_curve", [])
    if bench:
        fig.add_trace(go.Scatter(
            x=[b["date"] for b in bench],
            y=[b["value"] for b in bench],
            name="ベンチマーク",
            line=dict(color="#ff7f0e", dash="dash"),
        ))
    fig.update_layout(
        title="エクイティカーブ",
        xaxis_title="日付", yaxis_title="資産額 (円)",
        hovermode="x unified", height=400,
    )
    return fig


def main():
    st.title("🤖 AI研究エージェント")
    st.caption("AIが投資アイデアを自動で分析・検証します")

    # API状態チェック
    if not AI_API_KEY:
        st.warning(
            "⚠️ AI_API_KEY が未設定です。.envファイルにAI_API_KEYを設定してください。\n\n"
            "未設定でもデモモード（ダミーAI応答）で動作します。"
        )

    st.divider()

    # --- アイデア入力 ---
    st.subheader("投資アイデアを入力")

    idea_title = st.text_input(
        "タイトル",
        placeholder="例: 月曜日の株式リターンは低い（月曜効果）",
    )
    idea_text = st.text_area(
        "アイデアの詳細説明",
        height=150,
        placeholder=(
            "例: 月曜日は他の曜日と比較して平均リターンが低い傾向がある。"
            "週末効果とも呼ばれるこのアノマリーが日本市場でも有効かを検証したい。"
            "大型株と小型株で効果に違いがあるかも確認する。"
        ),
    )
    category = st.selectbox("カテゴリ", ANALYSIS_CATEGORIES)

    st.divider()

    # --- AI研究実行 ---
    if st.button(
        "🚀 AI研究を実行",
        type="primary",
        use_container_width=True,
        disabled=not idea_text,
    ):
        if not idea_text:
            st.error("アイデアの説明を入力してください。")
            return

        db = get_db()
        provider = get_data_provider()
        ai_client = create_ai_client()

        researcher = AiResearcher(
            db=db,
            ai_client=ai_client,
            data_provider=provider,
        )

        # 進捗表示
        progress_container = st.container()
        progress_bar = st.progress(0)

        phase_progress = {
            "planning": 0.15,
            "coding": 0.35,
            "executing": 0.55,
            "interpreting": 0.75,
            "saving": 0.90,
            "done": 1.0,
            "error": 1.0,
        }

        phase_labels = {
            "planning": "📋 分析計画を生成中...",
            "coding": "💻 分析コードを生成中...",
            "executing": "⚡ コードを実行中...",
            "interpreting": "🧠 結果を解釈中...",
            "saving": "💾 知見を保存中...",
            "done": "✅ 研究完了!",
            "error": "❌ エラーが発生しました",
        }

        status_placeholder = st.empty()

        def on_progress(progress: ResearchProgress):
            pct = phase_progress.get(progress.phase, 0)
            progress_bar.progress(pct)
            label = phase_labels.get(progress.phase, progress.phase)
            status_placeholder.info(f"{label}\n\n{progress.message}")

        # 研究実行
        with st.spinner("AI研究を実行中..."):
            result = researcher.run_research(
                idea_text=idea_text,
                idea_title=idea_title or idea_text[:50],
                category=category,
                on_progress=on_progress,
            )

        progress_bar.progress(1.0)

        # --- 結果表示 ---
        st.divider()

        if result.phase == "error":
            st.error(f"研究中にエラーが発生しました: {result.error}")
            if result.generated_code:
                with st.expander("生成されたコード"):
                    st.code(result.generated_code, language="python")
            if result.execution_result and result.execution_result.get("error"):
                with st.expander("実行エラー詳細"):
                    st.code(result.execution_result["error"])
            return

        st.success(f"🎉 研究が完了しました! (Run #{result.run_id})")

        # タブで結果表示
        tab_summary, tab_plan, tab_code, tab_stats, tab_bt, tab_interp = st.tabs([
            "サマリー", "分析計画", "生成コード", "統計結果", "バックテスト", "AI評価",
        ])

        # --- サマリー ---
        with tab_summary:
            interp = result.interpretation
            label = interp.get("evaluation_label", "needs_review")
            confidence = interp.get("confidence", 0)
            label_color = {"valid": "green", "invalid": "red", "needs_review": "orange"}.get(label, "gray")

            st.markdown(f"### 判定: :{label_color}[{label}] (信頼度: {confidence:.0%})")

            if interp.get("summary"):
                st.write(interp["summary"])

            col1, col2 = st.columns(2)
            with col1:
                exec_result = result.execution_result.get("result", {})
                bt = exec_result.get("backtest", {}) if exec_result else {}
                if bt:
                    st.metric("シャープ比", f"{bt.get('sharpe_ratio', 0):.2f}")
                    st.metric("累計リターン", f"{bt.get('cumulative_return', 0):.2%}")
            with col2:
                stats = exec_result.get("statistics", {}) if exec_result else {}
                if stats:
                    st.metric("p値", f"{stats.get('p_value', 'N/A'):.4f}" if isinstance(stats.get('p_value'), (int, float)) else "N/A")
                    st.metric("効果量", f"{stats.get('cohens_d', 0):.3f}" if isinstance(stats.get('cohens_d'), (int, float)) else "N/A")

        # --- 分析計画 ---
        with tab_plan:
            st.subheader("AI生成の分析計画")
            plan = result.plan
            if plan:
                if plan.get("hypothesis"):
                    st.write(f"**仮説**: {plan['hypothesis']}")
                if plan.get("methodology", {}).get("approach"):
                    st.write(f"**アプローチ**: {plan['methodology']['approach']}")
                steps = plan.get("methodology", {}).get("steps", [])
                if steps:
                    st.write("**ステップ**:")
                    for i, s in enumerate(steps, 1):
                        st.write(f"  {i}. {s}")
                with st.expander("計画の全JSON"):
                    st.json(plan)

        # --- 生成コード ---
        with tab_code:
            st.subheader("AI生成の分析コード")
            if result.generated_code:
                st.code(result.generated_code, language="python", line_numbers=True)

        # --- 統計結果 ---
        with tab_stats:
            st.subheader("統計分析結果")
            exec_result = result.execution_result.get("result", {})
            stats = exec_result.get("statistics", {}) if exec_result else {}
            if stats and "error" not in stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    pv = stats.get("p_value")
                    st.metric("p値", f"{pv:.4f}" if isinstance(pv, (int, float)) else "N/A")
                with col2:
                    cd = stats.get("cohens_d")
                    st.metric("Cohen's d", f"{cd:.3f}" if isinstance(cd, (int, float)) else "N/A")
                with col3:
                    st.metric("条件群勝率", f"{stats.get('win_rate_condition', 0):.1%}")
                with col4:
                    sig = stats.get("is_significant", False)
                    st.metric("有意性", "✅ 有意" if sig else "❌ 非有意")

                with st.expander("統計結果の全JSON"):
                    st.json(stats)
            elif stats:
                st.warning(f"統計分析エラー: {stats.get('error', '不明')}")
            else:
                st.info("統計結果がありません")

        # --- バックテスト ---
        with tab_bt:
            st.subheader("バックテスト結果")
            exec_result = result.execution_result.get("result", {})
            bt = exec_result.get("backtest", {}) if exec_result else {}
            if bt and "error" not in bt:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("累計リターン", f"{bt.get('cumulative_return', 0):.2%}")
                with col2:
                    st.metric("年率リターン", f"{bt.get('annual_return', 0):.2%}")
                with col3:
                    st.metric("シャープ比", f"{bt.get('sharpe_ratio', 0):.2f}")
                with col4:
                    st.metric("最大DD", f"{bt.get('max_drawdown', 0):.2%}")
                with col5:
                    st.metric("取引回数", bt.get("total_trades", 0))

                fig = plot_equity_curve(bt)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            elif bt:
                st.warning(f"バックテストエラー: {bt.get('error', '不明')}")
            else:
                st.info("バックテスト結果がありません")

        # --- AI評価 ---
        with tab_interp:
            st.subheader("AI による評価")
            interp = result.interpretation

            if interp.get("reasons"):
                st.write("**判断理由:**")
                for r in interp["reasons"]:
                    st.write(f"- {r}")

            for section, title in [("strengths", "💪 強み"), ("weaknesses", "⚠️ 弱み"), ("suggestions", "💡 提案")]:
                items = interp.get(section, [])
                if items:
                    st.write(f"**{title}:**")
                    for item in items:
                        st.write(f"- {item}")

            with st.expander("AI評価の全JSON"):
                st.json(interp)


if __name__ == "__main__":
    main()
