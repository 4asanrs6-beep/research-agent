# Qlib統合モジュール

J-Quantsデータ + Qlib MLランキングによるクロスセクショナル銘柄ランキングシステム。

## 概要

既存のルールベースシグナル（イベントスタディ）を補完する **MLベースの銘柄ランキング** を提供する。

- **既存**: 「この条件に合致する銘柄はTOPIXに勝てるか？」（シグナル検証）
- **Qlib**: 「全銘柄の中でどれが良いか？」（銘柄選択）

## セットアップ

```bash
pip install -r requirements.txt
```

pyqlib 0.9.7 が Python 3.11 + Windows 11 で動作確認済み。

## 使い方

### 1. Streamlit UI

```bash
streamlit run app.py
```

サイドバーの「8_Qlib実験」ページを開く。

1. **DATA MANAGEMENT** タブ: J-QuantsデータをQlibバイナリ形式に変換
2. **EXPERIMENT** タブ: Alpha158 + LightGBM/XGBoost で学習・予測
3. **RESULTS** タブ: IC/ICIR/特徴量重要度/Top-K銘柄一覧を確認

### 2. Python API

```python
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from qlib_integration.data_bridge import convert_full
from qlib_integration.workflow_runner import run_experiment

# データ変換
provider = JQuantsProvider(api_key="...", cache=DataCache(...))
convert_full(provider=provider, start_date="2017-01-01", end_date="2026-03-07")

# 実験実行
result = run_experiment(model_type="lgb")
print(f"IC: {result['ic_mean']:.4f}, ICIR: {result['icir']:.4f}")
```

## アーキテクチャ

```
qlib_integration/
  config.py           - 設定・定数
  calendar_jp.py      - TOPIXから営業日カレンダー生成
  instruments_jp.py   - 銘柄リスト生成
  data_bridge.py      - J-Quants → Qlib bin変換（核心）
  workflow_runner.py   - Alpha158 + LGBModel 実験オーケストレータ
  result_adapter.py   - Qlib結果 → 既存形式変換
```

## 評価指標

| 指標 | 意味 | 目安 |
|---|---|---|
| IC | 予測スコアと実リターンのPearson相関 | >0.03: 使える, >0.05: 良い |
| ICIR | IC / IC標準偏差 | >0.5: 安定的に予測力あり |
| Rank IC | Spearman順位相関 | 外れ値に頑健 |

## 既存システムとの関係

既存コードは **一切変更なし**。追記のみ:
- `config.py` 末尾に `QLIB_DATA_DIR` 追加
- `requirements.txt` に `pyqlib>=0.9.5` 追加
- `core/sidebar.py` にQlib実験の実行インジケーター追加
- `pages/8_Qlib実験.py` 新規ページ追加
