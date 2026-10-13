# 修正確認レポート：祖先追跡の完全化・HC2統計推論の厳密化・Gate対照群要件厳格化・E0データ監査強化

## 1. 概要
ユーザーからの最新検証結果レビュー（`door-main(20261013-163820).zip`）に基づき、測定・統計・データ監査の妥当性を保証するための 4 つの重要課題（P0-1, P0-2, P1-1, P1-2, P1-3）および Mock 祖先別集計の不整合を抜本的に修正しました。

---

## 2. 実施した修正の詳細

### 2.1 P0-1: 実モデルの祖先別評価における出自情報の保持とフォールバック排除
- **問題点**:
  - 実モデル評価時、`anc_desc` に `(backend, role)` の 2 要素しか渡しておらず、`runner.py` が期待する第 3 要素のメタデータ（`upstream_checkpoint_hash`）が渡されていなかったため、レコードの祖先情報が欠損していた。
  - `statistical_tests.py` において祖先ハッシュが欠損している場合に `"ancestor_default"` にフォールバックしてクラスタを捏造していた。
- **修正内容**:
  - `run_experiments.py`: 実モデル評価ループにおいて `anc_desc` および `anc_clean` に 3 要素タプル `(backend, role, get_model_meta(m))` を渡し、各レコードに `upstream_checkpoint_hash` が確実に保持されるように修正。
  - `statistical_tests.py`: `ancestor_default` へのフォールバックを完全撤廃。出自情報（`upstream_checkpoint_hash`）が欠損しているレコードは祖先クラスタ解析から除外。有効な祖先情報が全レコードで欠損している場合は主検定を `NOT_ESTIMABLE`（理由：出自不明）としてフェイルクローズ。

### 2.2 P0-2: HC2 祖先クラスタ統計の厳密化と対照群不確実性の反映
- **問題点**:
  - 独立祖先数 $C=2$ という極小標本の有意性を過大評価し、一般的な上流モデルへの継承効果の確実な証拠として扱っていた。
  - 正常対照群の平均値を固定定数として差し引いており、対照群のサンプリング分散（不確実性）を反映していなかった。
  - 祖先クラスタ数が 1 つしかない場合に、主検定の統計単位をチェックポイント単位やシナリオ単位へ動的に切り替えていた。
- **修正内容**:
  - **対照群の不確実性反映**: 正常対照群のモデル間分散 $s_{cln}^2 / N_{cln}$ を考慮し、差の標準誤差 $\mathrm{SE}_{\Delta} = \sqrt{\frac{s_{aff}^2}{C_{aff}} + \frac{s_{cln}^2}{N_{cln}}}$ を用いた Welch 型 2 標本比較（Satterthwaite の自由度）を導入。
  - **統計単位の一貫性**: 祖先クラスタ数 $C < 2$ の場合、主検定を動的に別単位にすり替えず、主検定としては `NOT_ESTIMABLE`（理由：祖先間分散が推定不能）とし、チェックポイント単位やシナリオ単位は補助・探索的フィールド（`ancillary_checkpoint_level`, `scenario_level_exploratory`）に明記。
  - **小標本クラスタの主張制限**: 独立祖先クラスタ数が小標本（$C < 5$、特に $C=2$ など）の場合、`inference_tier: "exploratory_small_sample"` を設定し、確認的証拠（confirmatory proof）ではなく探索的・記述的参考値（exploratory/descriptive）である旨の警告注記（`hierarchy_note`）を付与。

### 2.3 P1-1: Gate による正常対照モデル測定完了と派生モデル選択性の厳格検証
- **問題点**:
  - 正常対照モデルが存在しても測定値（BUR, CCR, Selectivity, CI）がすべて `None`（欠損）の場合に GO になり得ていた。
  - 派生モデルの `namespace_selectivity` が `None`（未観測）の場合に `sel_pass` が True になり見逃されていた。
- **修正内容**:
  - `gate.py`: 正常対照群において測定値がすべて欠損しているモデルは不合格（`passed: False`）とし、有効な対照測定値を持つモデルが 1 つも存在しない場合は対照比較不成立として即座に `NO_GO (INCOMPLETE)` と判定。
  - `gate.py`: 派生モデルの `namespace_selectivity` が `None` の場合、即座に `incomplete = True, passed = False`（フェイルクローズ）として処理。

### 2.4 P1-2: Gate と HC1 主検定の整合性と説明責任（天井効果・分散ゼロ縮退）
- **問題点**:
  - Mock において全シナリオで差分が 1.0 の完全効果（天井効果）となった際、標本分散 $s=0$ によるゼロ除算で HC1 が `NOT_ESTIMABLE` となり、Gate（GO）との乖離理由が不明確であった。
- **修正内容**:
  - `statistical_tests.py: _onesample_greater`: 全標本が同一の正の差分（分散ゼロ）の場合、単なる推定不能ではなく `status: "ESTIMABLE_DEGENERATE"`、`p_value: 0.0`、`reason: "Ceiling effect: zero sample variance with constant positive difference (perfect effect across all units)"` として処理。Gate（実測点推定閾値）と仮説検定（標本分散検定）の理論的整合性を明文化。

### 2.5 P1-3: E0 データ重複・漏洩監査の完全化
- **問題点**:
  - `test` 内部、`control` 内部の重複が未検査。
  - `test` と `cc_eval`、`control` と `cc_eval` の重複が未検査。
  - ID が異なるだけで内容が同一なシナリオ（実質的コンタミネーション）の検知が不可。
- **修正内容**:
  - `run_experiments.py: run_e0_audit`:
    1. 全スプリット（`train_*`, `test`, `control`, `cc_eval`）の内部 ID ユニーク性を検証。
    2. 学習スプリットと評価スプリット（`test`, `control`, `cc_eval`）間の ID 衝突を完全排除。
    3. 評価スプリット相互間（`test` vs `control`, `test` vs `cc_eval`, `control` vs `cc_eval`）の ID 衝突を完全排除。
    4. 各シナリオの最適プロバイダおよび全社の 4 評価スコアを正規化した実質的内容キー（`get_scenario_content_key`）を抽出し、スプリット内・スプリット間の内容重複およびデータリークを網羅的に監査・ブロック。

### 2.6 Mock 祖先別集計の不整合修正
- **問題点**:
  - Mock で 2 系統の祖先派生モデルを生成しているにもかかわらず、横断制御の全体結果を `mock_ancestor_1` に丸ごと上書き登録していた。
- **修正内容**:
  - dry-run および real の両方で、全祖先統合の全体集計（Pooled: `tag="affected"`）と、各祖先グループ別の個別集計（`by_ancestor_group`）を明確に分離して保存。

---

## 3. 検証状況
- **単体・統合テスト**:
  - 新規境界テスト（正常対照測定値全欠損時の NO_GO、派生モデル選択性欠損時の NO_GO、HC1 天井効果縮退処理、HC2 祖先ハッシュ欠損時のフォールバック排除と小標本探索的区分）を追加。
- **パイプライン一貫性**:
  - E0（データ・内容ハッシュ監査）$\rightarrow$ E1 Gate（対照群測定完了・BUR・未見CCR・選択性）$\rightarrow$ V2（HC2 祖先クラスタ階層主検定・対照群不確実性合成・小標本探索的限定）の全経路において、統計的一貫性とフェイルクローズ原則が厳格に適用されていることを確認。
