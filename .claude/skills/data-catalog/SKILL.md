---
name: data-catalog
description: 利用可能なデータソース・既存手法・分析基盤の一覧を表示する
allowed-tools: Read, Grep, Glob
---

このプロジェクトで利用可能なデータと手法を一覧化してください。

## データソースの確認

### J-Quants API V2（日本株）
!`grep -n "def get_" data/jquants_provider.py 2>/dev/null | head -15`

### yfinance（米国株・指標）
!`grep -n "yf\.\|yfinance" core/lead_lag/data_fetcher.py 2>/dev/null | head -10`

### 既存モジュール
!`ls core/*.py 2>/dev/null | xargs -I{} basename {}`
!`ls core/stock_allocation/*.py 2>/dev/null | xargs -I{} basename {}`
!`ls core/lead_lag/*.py 2>/dev/null | xargs -I{} basename {}`
!`ls qlib_integration/*.py 2>/dev/null | xargs -I{} basename {}`

## 出力形式

以下のカテゴリ別に整理して出力してください。

### 1. 日本株データ（J-Quants）
- 株価日足（OHLCV、調整済み）
- 信用取引（買い残/売り残、標準/制度/一般）
- 財務サマリー
- 空売り比率
- 銘柄情報（セクター17/33、市場、時価総額区分）
- 指数（TOPIX等）

### 2. 米国株データ（yfinance）
- 個別株・ETF日足
- VIX
- 為替（USDJPY）
- セクターETF（11種 + 22業種）

### 3. 既存分析手法
- Lead-Lag PCA（セクターETF間）
- イベントスタディ（個別株ペア）
- クロスバリデーション（期間分割検証）
- ネットワーク分析（ペアの可視化）
- Alpha158 + LightGBM（ML銘柄ランキング）
- 残差分析（市場/業種/スタイル分解）

### 4. 制約・注意点
- J-Quants APIのレートリミット
- yfinanceのリアルタイム性（15分遅延）
- Qlibデータは事前変換が必要
- 日米の取引時間差（JP先行、US後行）
