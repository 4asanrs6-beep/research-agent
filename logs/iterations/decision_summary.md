# Decision Summary

## T2: turnoverショック固有効果の戦略化 ← 現在のテーマ

T1で確立した因果チェーン（米国ショック → turnover高群がショック固有に継続下落 → 保有者交代メカニズム → 5日集中）を実際の戦略として構築・検証する。

**T1からの引き継ぎ知見:** K1-K26（`logs/experiments/knowledge.md`）
**フェーズ:** Q03/Q04選定完了。着手順: Q04→Q03。Phase 3（設計）へ

## T2 知見
27. ブレークポイントはday 5。下落はday 3-4に集中。固定5日保有が最適
28. 動的エグジット（turnover正常化ベース）は不可。rho=-0.007
29. turnover変化倍率はエントリー精度を改善しない（負の証拠）。当日消化度のproxy
30. (探索的) 変化倍率のCARは非線形。Q1(1.2倍)が最悪

詳細: `logs/experiments/knowledge.md`

### ❌ T2-Q01 entry-timing-turnover-observability — エントリータイミング

**結論:** FAIL。仮説と逆方向。turnover変化倍率高群(2.3倍)は+0.17%で回復。変化倍率は当日消化度のproxy
**得られた知見:** K29追加、K30(探索的)
**詳細:** `logs/experiments/T2-Q01_entry-timing-turnover-observability_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済
Phase 4 実験 ......... 済 — FAIL
Phase 5 議論 ......... 済 — Codex裁定: 次の問いに移る

---

### ✅ T2-Q02 optimal-holding-period-decay — 固定5日保有が最適

**結論:** PASS。ブレークポイントday 5。下落はday 3-4に集中。動的エグジットは不可（rho=-0.007）
**得られた知見:** K27-K28追加
**詳細:** `logs/experiments/T2-Q02_optimal-holding-period-decay_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済 — Codex conditional approve
Phase 4 実験 ......... 済 — **PASS**
Phase 5 議論 ......... 済 — Codex裁定: PASS。次の問いへ

---

### ⬜ T2-Q04 event-clustering-capacity -- 連続ショック時の効果維持

**問い:** ショック間隔5日以内の連続ショックでもturnover効果は維持されるか。VIX統制+感度分析(3/7/10日)
**詳細:** `logs/iterations/multi_perspective.md` T2 Round 2

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 未
Phase 4 実験 ......... 未
Phase 5 議論 ......... 未

---

### ⬜ T2-Q03 backtest-transaction-cost-viability -- 取引コスト控除後の戦略成立性

**問い:** expanding windowでday+1ショート→5日クローズ。コスト10/20/30bps。年率リターン・Sharpe・最大DD・95%CI
**詳細:** `logs/iterations/multi_perspective.md` T2 Round 2

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 未
Phase 4 実験 ......... 未
Phase 5 議論 ......... 未

---

## T1: 米国ショック後の日本株の銘柄固有効果 ← 完了

米国株が急落したとき、日本株の中で特に大きく下がる銘柄の特徴を見つけ、トレードに活かす

## 知見
1. ボラが高い銘柄は超過下落する（一貫して有意）
2. 回転率が高い銘柄も超過下落する（ボラとは独立）
3. beta（市場連動度）は関係ない（一貫して非有意）
4. 回転率の効果は大型株に集中。中小型では信用残が効く
5. 信用残proxyでは回転率効果を説明できない
6. volは増幅器（上下対称）。方向を問わず反応を拡大
7. turnoverはパニック固有（下方のみ）。恐怖時の売り圧力。**Q06 Step3で再確認（上方p=0.268で消失）**
8. betaは閾値依存で不安定（未確定）。サンプルサイズで有意性が変わる
9. 大型turnover効果も下落時固有。ETFフローよりパニック売り
10. abnormal_vol_shareに追加説明力なし（棄却）
11. ~~vol起因の超過下落は20日で+1.6%反転~~ → **Q05で否定: 反転の85%は高ボラ株の通常ドリフト。ショック固有成分は+0.087%(p=0.222)**
12. **turnover起因の超過下落は反転せず-0.6%継続下落（p=0.003）**
13. **vol/turnoverのポストショック動態は正反対**
14. **高ボラ株は構造的に正のドリフトを持つ（非ショック日vol-high CAR=+0.584%/20d）**
15. **ショック規模とvol反転幅に相関なし（rho=0.035, p=0.661）**
16. **turnoverはモメンタムの完全な代理ではない（prior_ret_20d統制後もp=0.002）**
17. **turnoverとprior_retは独立した2チャネルで超過下落を説明**
18. **ポストショックturnover効果にサイズ依存性なし（Large p=0.59, Small p=0.35）**
19. **K4のday-0サイズ依存性はポストショック動態に波及しない**

詳細: `logs/experiments/knowledge.md`

## 問いの系譜と進捗

```
T2: turnoverショック固有効果の戦略化
  |
  +-- ❌ T2-Q01 entry-timing-turnover-observability .. FAIL（K29-30）
  +-- ✅ T2-Q02 optimal-holding-period-decay ........ PASS（K27-28）
  +-- ⬜ T2-Q03 backtest-transaction-cost-viability .. 未着手（Q04完了後）
  +-- ⬜ T2-Q04 event-clustering-capacity ............ 未着手（先行実行）
```

```
T1: 米国ショック後の日本株の銘柄固有効果
  |
  +-- ❌ T1-Q01 margin-shock ............. 棄却（知見1-3）
        |
        | 知見1-3から3問を生成 -> 人間が選定
        |
        +-- ✅ T1-Q02 holder-instability ... 完了（知見4-5）
        |
        +-- ✅ T1-Q03 symmetry-test ........ 完了（知見6-9）
        |
        +-- ❌✅ T1-Q04 attention-penalty .. 主仮説棄却+副次知見（知見10-13）
              |
              +-- ❌ T1-Q05 vol-reversal-specificity .. ショック非固有（K11修正, K14-15）
              +-- ❌(条件付き✅) T1-Q06 turnover-momentum-disentangle .. 連続PASS/離散FAIL（K16-17）
              +-- ❌ T1-Q07 size-regime-interaction .. FAIL（K18-19）
              +-- ⏭️ T1-Q08 cross-factor-portfolio .. スキップ（前提崩壊）
              +-- ✅ T1-Q09 turnover-postshock-specificity-did .. **PASS (MAJOR)**（K21-23）
              +-- ✅ T1-Q10 turnover-decline-mechanism-separation .. **PASS**（K24-26）
              +-- ⏭️ T1-Q11 vol-drift-as-alternative-theme .. Q09 PASSにより不要
```

## 各問いの詳細

---

### ❌ T1-Q01 margin-shock — 信用残が多い銘柄は余計に下がるか

**結論:** 信用残は効かなかった（符号反転、p=0.56）
**得られた知見:** ボラと回転率が有意、betaが非有意（知見1-3）
**詳細:** `logs/experiments/T1-Q01_margin-shock.md`

---

### ✅ T1-Q02 holder-instability — 回転率は何を捉えているか

**結論:** 大型株では回転率が効く（p=0.010）。中小型では信用残が効く（p=0.050）。メカニズムがサイズ依存
**得られた知見:** 知見4-5
**詳細:** `logs/experiments/T1-Q02_holder-instability_stage1.md`, `logs/experiments/T1-Q02_holder-instability_stage2.md`

---

### ✅ T1-Q03 symmetry-test — 上方ショックでも同じパターンか <-- NEW

**結論:** volは増幅器（上下で符号反転、両方有意）。turnoverは下方のみ有意で上方で消失。betaは上方で復活の兆し
**得られた知見:** 知見6-9
**詳細:** `logs/experiments/T1-Q03_symmetry-test.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済
Phase 4 実験 ......... 済
Phase 5 議論 ......... 済  -- vol=増幅器、turnover=パニック固有。beta未確定

---

### ❌✅ T1-Q04 attention-penalty — 注目集中度は効かなかったが反転テストで重要知見

**結論:** 主仮説（abnormal_vol_share）は棄却（p=0.189）。副次テストでvol反転+1.6%、turnover継続下落-0.6%を発見
**得られた知見:** 知見10-13
**詳細:** `logs/experiments/T1-Q04_attention-penalty_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済
Phase 4 実験 ......... 済
Phase 5 議論 ......... 済 — Codex裁定: 次の問いに移る

---

### ❌ T1-Q05 vol-reversal-specificity — vol反転はショック固有ではなかった

**結論:** pseudo-DiD=+0.087%(p=0.222)。vol反転の85%は高ボラ銘柄の通常ドリフト。ショック固有の反転は確認されず
**得られた知見:** K11修正（ショック非固有）、K14-K15追加
**詳細:** `logs/experiments/T1-Q05_vol-reversal-specificity_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済（multi-perspective v2で進化）
Phase 3 設計 ......... 済 — Codex-fallback approve
Phase 4 実験 ......... 済 — FAIL
Phase 5 議論 ......... 済 — Codex裁定: 次の問いに移る

---

### ❌(条件付き✅) T1-Q06 turnover-momentum-disentangle — モメンタムの代理ではないが離散テスト不発

**結論:** 離散層別FAIL（1/3ウィンドウのみ）。ただし連続回帰でturnover p=0.002（事前リターン統制後も維持）。上方ショックで効果消失確認(PASS)
**得られた知見:** K16-K17追加
**詳細:** `logs/experiments/T1-Q06_turnover-momentum-disentangle_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済 — Codex-fallback approve
Phase 4 実験 ......... 済 — FAIL(離散)/PASS(連続)
Phase 5 議論 ......... 済 — Codex裁定: 次の問いに移る

---

### ❌ T1-Q07 size-regime-interaction — サイズ別で戦略を分けるべきか

**結論:** FAIL。ポストショックturnover効果にサイズ依存性なし。Large diff=-0.43%(p=0.59), Small diff=-0.38%(p=0.35)
**得られた知見:** K18-19追加、K20(参考)
**詳細:** `logs/experiments/T1-Q07_size-regime-interaction_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済 — Codex approve (3往復)
Phase 4 実験 ......... 済 — FAIL
Phase 5 議論 ......... 済 — Codex裁定: 次の問いに移る

---

### ⏭️ T1-Q08 cross-factor-portfolio — スキップ（人間判断で差し替え）

**理由:** Q05でvol反転がショック非固有と判明し、Q08の前提が崩れた。

---

### ✅ T1-Q09 turnover-postshock-specificity-did — turnover継続下落はショック固有！（T1最大の成果）

**結論:** PASS。pseudo-DiD=-0.703% (p<0.0001)。非ショック日turnover高群は+0.196%。ショック固有成分が確定
**得られた知見:** K21-K23追加、K12補強
**詳細:** `logs/experiments/T1-Q09_turnover-postshock-specificity-did_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済 — Codex conditional approve
Phase 4 実験 ......... 済 — **PASS (MAJOR)**
Phase 5 議論 ......... 済 — Codex裁定: 次の問いに移る（Q10へ）

---

### ✅ T1-Q10 turnover-decline-mechanism-separation — メカニズムは保有者交代

**結論:** PASS。turnover上昇群CAR=-0.662%(p=0.005)。流動性枯渇ではなく保有者交代。ショートが可能
**得られた知見:** K24-K26追加
**詳細:** `logs/experiments/T1-Q10_turnover-decline-mechanism-separation_review.md`

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 済 — Codex conditional approve
Phase 4 実験 ......... 済 — **PASS**
Phase 5 議論 ......... 済 — Codex裁定: 戦略実装フェーズへ移行

---

### ⬜ T1-Q11 vol-drift-as-alternative-theme — 高ボラ株ドリフトの代替テーマ（Q09がFAIL後）

**問い:** K14の構造的ドリフト(+0.584%/20d)を独立テーマとして追求。ショック日を押し目買い機会として再定義
**詳細:** `logs/iterations/multi_perspective.md` Round 3

Phase 1 生成 ......... 済
Phase 2 選定 ......... 済
Phase 3 設計 ......... 未
Phase 4 実験 ......... 未
Phase 5 議論 ......... 未

---

## 直近の動き

### [2026-03-31] T2-Q04 Phase 4 実験実行中
Codex設計approve(2往復)。--t2q04 --skip-symmetryで実行中

### [2026-03-31] T2-Q04 Phase 3 設計完了
/idea-generationを実行中。Q04: 連続ショック時の効果維持（ショック間隔×CAR）

### [2026-03-31] T2 Phase 2 multi-perspective完了 → Phase 3（設計）へ
2問を5ロール×2ラウンドで議論。全ロール合意。着手順: Q04(連続ショック)→Q03(バックテスト)
Q04のキャパ制約をQ03に反映する設計。**次: T2-Q04 Phase 3（設計）**

### [2026-03-31] T2 Phase 2 選定完了
人間が2問を選定: backtest-transaction-cost-viability, event-clustering-capacity

### [2026-03-31] T2 Phase 1（問い生成）完了
Claude 5問 + Claude-fallback 4問 = 9問を生成。人間が選定

### [2026-03-31] T2-Q01 Phase 5完了 → FAIL。turnover変化倍率はエントリー精度を改善しない
仮説と逆方向。変化倍率高群(2.3倍)は+0.17%で回復。K24(保有者交代)と整合的で当日消化度のproxy。
Codex裁定: 次の問いに移る。K29-K30追加

### [2026-03-31] T2-Q02 Phase 5完了 → PASS。固定5日保有が最適
ブレークポイントday 5。下落はday 3-4に集中。動的エグジットは不可。K27-K28追加。
**次: T2-Q01 entry-timing-turnover-observability**

### [2026-03-31] T2 Phase 2完了 → Phase 3（設計）へ
着手順: Q02(保有期間)→Q01(エントリー)。Q02が先（実行コスト低・情報価値高）。
**次: Phase 3（設計）→ optimal-holding-period-decayから**

### [2026-03-31] テーマ移行: T1完了 → T2開始
T1（米国ショック後の銘柄固有効果）の研究フェーズ完了。10問で因果チェーン確立。
T2（turnoverショック固有効果の戦略化）をPhase 1から開始。
T1の知見K1-K26はT2でも引き継ぎ。

### [2026-03-31] T1-Q10 Phase 5完了 → PASS! メカニズムは保有者交代
turnover上昇群CAR=-0.662%(p=0.005)。ショートが可能。T1メカニズム解明完了。
**次: 戦略実装フェーズ（リアルタイムシグナル設計、バックテスト）へ移行。人間の判断待ち**

### [2026-03-31] T1-Q09 Phase 5完了 → PASS! turnover継続下落はショック固有
pseudo-DiD=-0.703%(p<0.0001)。T1テーマの戦略価値が確定。K21-K23追加。
**次: Q10 turnover-decline-mechanism-separation（メカニズム解明）**

### [2026-03-31] Phase 2完了: multi-perspective議論 + 着手順決定
Q09最優先 → PASS: Q10, FAIL: Q11 の分岐構成。
**次: Phase 3（設計）→ turnover-postshock-specificity-didから**

### [2026-03-31] T1-Q07 Phase 5 議論完了 → サイズ依存性なし、次の問いへ
FAIL。Large diff=-0.43%(p=0.59), Small diff=-0.38%(p=0.35)。K4のday-0サイズ依存性はポストショックに波及しない。
Codex裁定: 次の問いに移る。**最優先: K12ショック固有性DiD**

### [2026-03-30] T1-Q07 Phase 4 実験完了 (FAIL)
コードレビュー4往復（Codex3回+Claude-fallback1回）で修正完了。`--q07 --skip-symmetry`で実行中

### [2026-03-30] T1-Q07 Phase 3 設計完了 → Phase 4 実装へ
Codex設計レビュー3往復でapprove。主判定=TOPIX離散分割(Large/Small)でのturnover高低別20d CAR差。
補助=log_trading_value連続交互作用。結論範囲はショック日内部のサイズ依存性の記述に限定

### [2026-03-30] T1-Q06 Phase 5 議論完了 → turnoverはモメンタム代理ではないが離散テスト不発
離散層別FAIL、連続回帰PASS(p=0.002)。上方ショックで消失確認。
Codex裁定: 次の問いに移る。**次: Q07 size-regime-interaction（またはK12ショック固有性DiD）**

### [2026-03-30] T1-Q05 Phase 5 議論完了 → vol反転はショック固有ではない
pseudo-DiD=+0.087%(p=0.222)。K11を修正。vol反転ロング戦略は棄却。
Codex裁定: 次の問いに移る。**次: T1-Q06 turnover-momentum-disentangle**

### [2026-03-30] T1-Q05 vol-reversal-specificity Phase 3 設計完了
案1（3+1条件ベースライン+pseudo-DiD）を採用。Codex-fallback approve。

### [2026-03-30] /multi-perspective完了 → Phase 3（設計）へ
4問を5ロール×2ラウンドで議論。着手順: vol-reversal → turnover-momentum → size-regime → cross-factor

### [2026-03-30] Phase 2: 人間が4問を選定
vol-reversal-specificity, turnover-momentum-disentangle, size-regime-interaction, cross-factor-portfolio。

### [2026-03-30] Phase 1: 次の問い候補10問を生成
知見K1-K13から3方向（前提条件潰し/戦略テスト/メカニズム深掘り）で10問を生成。
Claude 5問 + Claude-fallback 5問。

### [2026-03-30] T1-Q04 attention-penalty: Phase 5 議論完了
主仮説（abnormal_vol_share）棄却。反転テストでvol=反転、turnover=継続下落を発見（知見K10-K13）。
Codex裁定: 次の問いに移る。Q05候補: vol反転のショック固有性検証。
全問い完了 → **Phase 1（問い生成）に進む。人間が選定。**

### [2026-03-30 00:00] T1-Q03 symmetry-test: Phase 5 完了
volは増幅器（上下対称）、turnoverはパニック固有（下方のみ）という明確な結果。

### [2026-03-29 20:00] T1-Q03 symmetry-test: Phase 3 設計完了
Codex設計2往復+コード3往復でapprove。

### [2026-03-29 19:00] T1-Q02 holder-instability 完了 -> T1-Q03へ
大型/中小型でメカニズムが分かれることが判明。
