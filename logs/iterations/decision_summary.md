# Decision Summary

## T1: 米国ショック後の日本株の銘柄固有効果

米国株が急落したとき、日本株の中で特に大きく下がる銘柄の特徴を見つけ、トレードに活かす

## 知見
1. ボラが高い銘柄は超過下落する（一貫して有意）
2. 回転率が高い銘柄も超過下落する（ボラとは独立）
3. beta（市場連動度）は関係ない（一貫して非有意）
4. 回転率の効果は大型株に集中。中小型では信用残が効く
5. 信用残proxyでは回転率効果を説明できない
6. **volは増幅器（上下対称）。方向を問わず反応を拡大**
7. **turnoverはパニック固有（下方のみ）。恐怖時の売り圧力**
8. **betaは閾値依存で不安定（未確定）。サンプルサイズで有意性が変わる**
9. **大型turnover効果も下落時固有。ETFフローよりパニック売り**

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
        +-- ✅ T1-Q03 symmetry-test ........ 完了（知見6-9）<-- NEW
        |
        +-- ⬜ T1-Q04 attention-penalty .... 未着手
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

### 🔄 T1-Q04 attention-penalty — 注目集中銘柄の一時的罰則か <-- 次

**状態:** Phase 3（設計）から開始
**詳細:** `logs/iterations/multi_perspective.md` cq4セクション

---

## 直近の動き

### [2026-03-30 00:00] T1-Q03 symmetry-test: Phase 4 実験完了
volは増幅器（上下対称）、turnoverはパニック固有（下方のみ）という明確な結果。
次: Phase 5 議論（/experiment-review）

### [2026-03-29 20:00] T1-Q03 symmetry-test: Phase 3 設計完了
Codex設計2往復+コード3往復でapprove。

### [2026-03-29 19:00] T1-Q02 holder-instability 完了 -> T1-Q03へ
大型/中小型でメカニズムが分かれることが判明。
