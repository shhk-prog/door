# 実装計画書：祖先追跡の完全化・HC2統計推論の厳密化・Gate対照群要件厳格化・E0データ監査強化

## 1. 概要と背景
前回の改修によって Mock パイプラインは最後まで完走し、主要な P0 不具合は解消されたが、独立検証により以下の統計解析・データ監査上の重大な課題が特定された。
1. **P0-1（実モデル祖先情報の保持）**: 実モデルの `run_cross_descendant` 呼び出しでメタデータ（祖先ハッシュ）が渡されておらず、さらに統計検定側で出自情報欠損時に `ancestor_default` へフォールバックしてクラスタを捏造していた。
2. **P0-2（HC2 祖先クラスタ統計の厳密化）**: 祖先数 $C=2$ という極小標本の有意性を過大主張しており、正常対照群の分散不確実性を定数として扱っていた。また祖先数不足時に主検定の統計単位を動的変更（すり替え）していた。
3. **P1-1（Gate 正常対照群と選択性の厳格化）**: 正常対照モデルが存在しても測定値（BUR/CCR）がすべて欠損している場合に GO と判定され、派生モデルの選択性欠損（`None`）が見逃されていた。
4. **P1-2（Gate と HC1 主検定の整合性）**: 完全効果（全シナリオで差が 1.0）時に $s=0$ によるゼロ除算で HC1 が `NOT_ESTIMABLE` となり、Gate（GO）との乖離理由が不明確であった。
5. **P1-3（E0 データ重複監査の完全化）**: `test`, `control`, `cc_eval` 間の重複検査が欠落し、内容が実質的に同一なシナリオの検知ができなかった。
6. **Mock 祖先別集計の不整合**: dry-run で 2 系統の祖先派生モデルをまとめた結果を `mock_ancestor_1` に丸ごと登録していた。

## 2. 修正詳細設計

### 2.1 P0-1: 実モデル祖先情報の保持と出自欠損時のフォールバック排除
- **`run_experiments.py`**:
  - 実モデル評価ループ（`by_upstream_hash`）において、`anc_desc` および `anc_clean` の辞書値に 3 要素タプル `(backend, role, get_model_meta(m))` を確実に渡し、`upstream_checkpoint_hash` が `runner.py` に伝播するようにする。
- **`statistical_tests.py`**:
  - `ancestor_default` へのフォールバックを完全撤廃。
  - レコードの `upstream_checkpoint_hash`（または `ancestor_id`）が欠損または空の場合、そのレコードは出自情報不明として除外する。
  - 有効な祖先ハッシュを持つレコードが存在しない場合、祖先クラスタ階層主検定は `NOT_ESTIMABLE`（理由：`"Ancestry provenance (upstream_checkpoint_hash) is missing across records"`）としてフェイルクローズする。

### 2.2 P0-2: HC2 祖先クラスタ統計の厳密化
- **統計単位の一貫性維持**:
  - 主検定（Primary Test）は「祖先クラスタ階層主検定（`ancestor_clusters`）」として固定。祖先クラスタ数が不足するからといって、主検定の単位をチェックポイント単位やシナリオ単位へ動的に切り替えない。
  - 祖先クラスタ数 $C < 2$ の場合は主検定を `NOT_ESTIMABLE`（`"Insufficient independent ancestor clusters (C < 2)"`）とする。モデル単位・シナリオ単位の解析はあくまで探索的・補助的結果（`ancillary_model_level`, `scenario_level_exploratory`）に格納する。
- **小標本祖先数の確認的推論制限**:
  - 独立祖先クラスタ数が小標本（例: $C < 5$、特に $C=2$ など）の場合、自由度が極めて小さく正規性仮定の検証も困難であるため、仮説検定の結論として「一般的な上流モデルへの継承効果の確実な証拠」と過大主張しないよう、`inference_tier: "exploratory_small_sample"` および制約注記（`"Small number of ancestor clusters (C=2). Results should be interpreted as exploratory/descriptive evidence, not confirmatory proof."`）を付与する。
- **対照群の不確実性の反映（2標本クラスタ検定 / 差の合成分散）**:
  - 正常対照群の祖先クラスタ平均（またはモデル平均）の分散 $s_{cln}^2 / C_{cln}$ を考慮。
  - 差の標準誤差 $\mathrm{SE}_{\mathrm{diff}} = \sqrt{\frac{s_{aff}^2}{C_{aff}} + \frac{s_{cln}^2}{C_{cln}}}$ に基づく Welch 型 t 統計量、または対照群のサンプリング分散を合成した信頼区間を算出する。

### 2.3 P1-1: Gate 正常対照モデル測定完了と派生モデル選択性の厳格検証
- **`gate.py`**:
  - 正常対照モデル（`clean_control`）について：
    - `clean_names` にモデルが存在しても、すべての正常対照モデルで `bur` や `ccr` が欠損（`None`）している場合、対照群比較不成立として `incomplete = True, all_passed = False` とする。
    - 正常対照モデルの BUR が健全水準（`bur >= min_bur`）を満たしているかを検証する。
  - 派生モデルについて：
    - `d_sel is None` の場合、選択性が未観測・欠損であるため `incomplete = True, passed = False`（フェイルクローズ）とする。

### 2.4 P1-2: Gate と HC1 主検定の整合性と説明責任
- **`statistical_tests.py`**:
  - `_onesample_greater`: 全サンプルで差が同一の正値（例: 全シナリオで差分 1.0、標本分散 $s=0$）の場合、効果は天井効果（Ceiling effect）であり、単なるデータ欠測や推定不能ではなく `reason: "Ceiling effect: zero sample variance with constant positive difference (perfect effect)"` を明記し、効果量 `1.0` を保持する。
  - Gate（実測点推定と安全マージン）と HC1（分散・t検定）の判定アプローチの違いをドキュメントと JSON に注記。

### 2.5 P1-3: E0 データ重複・漏洩監査の完全化
- **`run_experiments.py:run_e0_audit`**:
  - スプリット内重複: `seen_train_ids`, `seen_test_ids`, `seen_control_ids`, `seen_cc_eval_ids` の全スプリット内ユニーク性チェック。
  - スプリット間漏洩:
    - `test` と `train` の衝突禁止
    - `control` と `train` の衝突禁止
    - `cc_eval` と `train` の衝突禁止
    - `test` と `cc_eval` の衝突禁止
    - `control` と `cc_eval` の衝突禁止
  - シナリオ実質的内容ハッシュ:
    - プロバイダ属性（スコア）と最適解、要求パラメータを正規化したハッシュ値を全シナリオで追跡し、ID違いの完全一致シナリオ（実質的重複・漏洩）を検出・ブロック。

### 2.6 Mock 祖先別集計の不整合修正
- **`run_experiments.py`**:
  - dry-run においても `by_upstream_hash` を用い、`mock_ancestor_1` と `mock_ancestor_2` を正しく個別の集計辞書に記録し、全体集計と明確に分離する。

---

## 3. テスト・検証計画
1. `pytest` による単体テスト・Gate 境界テスト（対照群測定値欠損、選択性欠損など）の全件通過確認。
2. `python run_experiments.py --mode cc --dry_run` によるパイプライン完走（Exit Code 0、Manifest 正常記録）。
3. 生成された JSON（Gate、HC1/HC2、Cross-Descendant）の構造検証。
