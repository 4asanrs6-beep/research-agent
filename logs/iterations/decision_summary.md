# Decision Summary

## T1: 米国ショック後の日本株の銘柄固有効果

米国株が急落したとき、日本株の中で特に大きく下がる銘柄の特徴を見つけ、トレードに活かす

## 知見
1. ボラが高い銘柄は超過下落する（一貫して有意）
2. 回転率が高い銘柄も超過下落する（ボラとは独立）
3. beta（市場連動度）は関係ない（一貫して非有意）
4. 回転率の効果は大型株に集中。中小型では信用残が効く
5. 信用残proxyでは回転率効果を説明できない
6. volは増幅器（上下対称）。方向を問わず反応を拡大
7. turnoverはパニック固有（下方のみ）。恐怖時の売り圧力
8. betaは閾値依存で不安定（未確定）。サンプルサイズで有意性が変わる
9. 大型turnover効果も下落時固有。ETFフローよりパニック売り
10. **abnormal_vol_shareに追加説明力なし（棄却）**
11. **vol起因の超過下落は20日で+1.6%反転（p<0.001）**
12. **turnover起因の超過下落は反転せず-0.6%継続下落（p=0.003）**
13. **vol/turnoverのポストショック動態は正反対**

詳細: `logs/experiments/knowledge.md`

## 問いの系譜と進捗

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
        +-- ❌✅ T1-Q04 attention-penalty .. 主仮説棄却+副次知見（知見10-13）<-- DONE
              |
              +-- Q05: vol反転のショック固有性検証（次の問い候補）
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

## 直近の動き

### [2026-03-30] Phase 2: 人間が4問を選定
vol-reversal-specificity, turnover-momentum-disentangle, size-regime-interaction, cross-factor-portfolio。
前提条件潰し(1,2) → サイズ境界(3) → 戦略テスト(4) の依存順。
**次: Phase 3（設計）→ vol-reversal-specificityから着手**

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
