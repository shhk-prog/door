# 実装計画：本実験パイプライン統合・Gate管理・統計指標厳格化

## 1. 概要と目的
レビューで指摘された以下の P0（致命的課題）および P1（統計・内部解析課題）に対処し、ICLR論文の `Behavioral → V1 → V2 → V3` の仮説検証手順を一貫した実験基盤として統合します。

1. **P0-1（実験経路の統合）**: `--mode cc` を中心とした統一パイプラインへ再編し、Stage 1 (Behavioral) 完了時に E1-CC Gate による事前検証（Go/No-Go判定）を挟み、後続の Stage 2 (V1: Representation), Stage 3 (V2: FT Persistence), Stage 4 (V3: Causal Utilization) の実行を一元制御する。
2. **P0-2（FT残存率の推定不能処理）**: 分母（親モデルの変化量）が小さい場合に勝手に `1.0` や `0.0` を代入する処理を廃止し、比率は `None`（推定不能）とし、絶対差（`delta_retention_abs`, `utility_diff_abs`）を併記する。
3. **P0-3（E1 Gateの条件照合）**: 単に `decision == "GO"` だけでなく、`run_id`, `config_hash`, `dataset_hash`, `checkpoint_manifest_hash`, `execution_mode` の5項目が一致していることを照合する。
4. **P0-4（学習時と評価時の形式整合性）**: 教師データ生成・トークナイズ境界の接頭辞整合性および感度検証。
5. **P1（統計・内部解析の厳格化）**:
   - `statistical_tests.py`: 小標本での正規近似（1.96）を廃止し、t分布またはBootstrapに統一。欠測値と実測0.0の区別。階層構造の反映。
   - `likelihood.py`: 制約付きSoftmaxと真の生成確率の区別の注記、共同Bordaの仮定の明記。
   - `real_circuit_backend.py`: 単純文字列差交絡への対策（Near-miss対照）。
   - `run_experiments.py`: 横断評価での全共通祖先グループ集計。
   - `utils/manifests.py`: 完了済み実験の一致判定厳格化。

---

## 2. 変更対象ファイルと詳細計画

### 2.1 P0-2: `lineage_collusion/experiments/runner.py`
- `retention` 算出箇所（704行目付近）：
  ```python
  if up_delta > 0.05:
      retention = d_delta / up_delta
  else:
      retention = None  # 推定不能
  ```
- 同様に `u_retention`:
  ```python
  if up_util > 0.05:
      u_retention = d_util / up_util
  else:
      u_retention = None  # 推定不能
  ```
- 指標辞書に `delta_retention_abs = d_delta - up_delta`、`utility_diff_abs = d_util - up_util` を必ず記録。

### 2.2 P0-3: `lineage_collusion/analysis/gate.py`
- `gate_allows_continuation` 関数および `evaluate_e1_gate` 関数の改修：
  - `gate_payload` に `run_id`, `config_hash`, `dataset_hash`, `checkpoint_manifest_hash`, `execution_mode` を含める。
  - `gate_allows_continuation(decision_payload, expected_conditions=None)` において、`expected_conditions` が指定されている場合は各ハッシュ値およびモードの一致を検証する。不一致の場合は例外または `False` を返す。

### 2.3 P0-1: `run_experiments.py`
- 可変命令チャネル（`--mode cc`）を主軸とする統一パイプラインへ再構成。
- Stage 1 (Behavioral: Programmability, Selectivity, Strict Compliance) 完了後に、`evaluate_e1_cc_gate` を呼び出し判定。
  - Go/No-Go判定結果を `gate_decision.json` に出力。
  - `--strict_gate` オプション（デフォルトTrue）により、NO-GO の場合は Stage 2 以降を自動スキップ（または `--allow_exploratory` がある場合のみ探索的解析として明示的に実行）。
- Stage 2 (V1: Representation) → Stage 3 (V2: FT Persistence) → Stage 4 (V3: Causal Utilization) のパイプラインを順次実行し、統合レポートを生成。

### 2.4 P0-4 & P1: 統計・尤度・整合性
- `lineage_collusion/analysis/statistical_tests.py`:
  - `confidence_interval`: 小標本（n < 30）の正規近似（1.96）を `scipy.stats.t.ppf(0.975, df=n-1)` に修正。
  - 欠測値と実測0.0の区別：`nan` を `0.0` に勝手に置換せず、欠測として除外または別途フラグで管理。
- `lineage_collusion/agents/likelihood.py`:
  - Softmax確率と自由生成確率の相違、共同Bordaの条件付き独立仮定に関する明示的なドキュメンテーションと警告を追加。
- `lineage_collusion/circuits/real_circuit_backend.py`:
  - 意味と表層文字列の分離対照（Near-miss trigger）に関する注釈・ハンドリング。
- `lineage_collusion/utils/manifests.py`:
  - 実験一致判定におけるパラメータ照合を厳格化。

---

## 3. 検証計画
- `pytest tests/` を実行し、既存テストおよび追加修正したテストがすべてパスすることを確認。
- `python run_experiments.py --mode cc --dry_run` を実行し、E1-CC Gate の判定フローおよび各 Stage（Behavioral → V1 → V2 → V3）の実行・サマリー出力が整合して動作することを確認。
