# 修正完了報告：本実験パイプライン統合・Gate管理・統計指標厳格化

## 1. 概要
ユーザーレビューで指摘された以下の P0（致命的課題）および P1（統計・内部解析課題）について、全項目の修正を完了しました。

1. **P0-1（実験経路の統合）**:
   - 可変命令チャネル（`--mode cc`）を主軸とする統一パイプラインへ再構成。
   - `Behavioral → V1 Representation → V2 FT Persistence → V3 Causal Utilization` の Stage 別進行を一元管理。
   - Stage 1（Behavioral）完了時に E1-CC Gate による事前検証（Go/No-Go判定）を挟み、不通過時は安全に中断（フェイルクローズ）し、強制時のみ `exploratory` として続行する機構を統合。
   - 結果のサマリー CSV（`summary_table.csv`）を生成。
2. **P0-2（FT残存率の不適切な代入排除）**:
   - `runner.py` において、分母（親モデルの変化量 `up_delta` または `up_util`）が小さい（$\le 0.05$）場合に `1.0` や `0.0` を代入する処理を完全に廃止。
   - 分母不足時は `None`（推定不能）とし、絶対差（`delta_retention_abs = d_delta - up_delta`、`utility_diff_abs = d_util - up_util`）を併記。
3. **P0-3（E1 Gateの実験条件照合の厳格化）**:
   - `gate.py` において、Gate判定ペイロードに `run_id`, `config_hash`, `dataset_hash`, `checkpoint_manifest_hash`, `execution_mode` の5項目を必ず埋め込むよう改修。
   - `gate_allows_continuation` でこれらの完全一致を検証し、異なる実験のGO判定を流用できないように防御。
4. **P0-4（学習時と評価時の形式整合性・感度分析）**:
   - 学習応答フォーマットと尤度評価接頭辞の整合性を保証した上で、`likelihood.py` に形式感度分析用の候補生成関数（`prefix`, `full_json`, `minimal_json`）を導入。
5. **P1課題（統計・内部解析の厳格化）**:
   - `statistical_tests.py`:
     - `_onesample_greater` の小標本信頼区間における正規近似（`1.96`）を廃止し、t分布臨界値（`stats.t.ppf(0.975, df=n-1)`）に統一。
     - HC2 検定において、欠測値への `0.0` 補完を排除し、観測値が存在する共通シナリオのみで差分検定を行い、欠測数を明記。
   - `likelihood.py`:
     - 6順位候補上の Softmax 分布（制約空間上の相対確率）と、自由生成空間における真の出力確率の区別を docstring に明記。
     - 共同 Borda 確率におけるエージェント間の条件付き独立仮定と通信による依存の解釈を明記。
   - `real_circuit_backend.py` & `mock_circuit_engine.py`:
     - 内部方向推定（`calibrate_control_direction`）に対照条件 `contrast_auth="wrong"`（Near-miss 署名 `Oak-ctrl`）を追加し、表層文字列差と意味的認証信号を分離。
   - `run_experiments.py`:
     - 横断制御評価において、単一の最大祖先グループだけでなく全共通祖先グループ（`by_upstream_hash`）を集計。
   - `manifests.py`:
     - `MATCH_FIELDS` に `seed` を追加し、完了済み実験の一致判定を厳格化。`CODE_VERSION` を最新版に更新。

---

## 2. 主な修正箇所の詳細

### 2.1 [runner.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/experiments/runner.py#L703-L738)
```python
# 分母が十分に大きくない (<= 0.05) 場合は比率を推定不能 (None) とし、絶対差を併記
retention = (d_delta / up_delta) if up_delta > 0.05 else None
u_retention = (d_util / up_util) if up_util > 0.05 else None
delta_retention_abs = round(d_delta - up_delta, 4)
utility_diff_abs = round(d_util - up_util, 4)
```

### 2.2 [gate.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py#L24-L182)
- `verify_gate_conditions(decision_payload, expected_conditions)`
- `evaluate_e1_cc_gate(behavioral_results, gate_cfg, execution_conditions)`
- `gate_allows_continuation(decision_payload, expected_conditions)` で `run_id`, `config_hash`, `dataset_hash`, `checkpoint_manifest_hash`, `execution_mode` の5項目を検証。

### 2.3 [run_experiments.py](file:///mnt/nas/home/hiromi/src/door/run_experiments.py#L550-L640)
- Stage 1 (Behavioral) 完了直後に `evaluate_e1_cc_gate` を呼び出し判定。
- `gate_allows_continuation` が `False` の場合、`--force_after_no_go` がなければ後続の Stage 2〜4 を安全に中断。
- 全共通祖先グループの集計と、末尾での `summary_table.csv` 出力を統合。

### 2.4 [statistical_tests.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L890-L1030)
- `_onesample_greater`:
  ```python
  t_crit = float(stats.t.ppf(0.975, df=len(arr) - 1))
  ci_95 = [round(mean - t_crit * se, 4), round(mean + t_crit * se, 4)]
  ```
- HC2 検定:
  ```python
  common_with_obs = [s for s in common_evaluated if s in aff and s in cln]
  diffs = [aff[s] - cln[s] for s in common_with_obs]
  ```

### 2.5 [real_circuit_backend.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/circuits/real_circuit_backend.py#L210-L250)
- `calibrate_control_direction` に `contrast_auth="wrong"` を追加し、同一フォーマットの near-miss 署名（`Oak-ctrl`）との差分により表層文字列の交絡を排除。

---

## 3. 今後の実行手順

### 3.1 決定論的 Mock による統合パイプラインの動作検証（推奨）
```bash
python run_experiments.py --mode cc --dry_run
```
- Stage 1 (Behavioral: Programmability, Selectivity) の実行
- E1-CC Gate による自動判定と実行条件照合
- Stage 2 (V1: Representation: Subspace Alignment)
- Stage 3 (V2: FT Persistence: Cross-Descendant Control & Retention)
- Stage 4 (V3: Causal Utilization: Subspace Ablation / Patching & Multi-Agent Collusion)
- `results/dry_run/cc/summary_table.csv` の一元出力

### 3.2 実モデルマニフェストによる本実験実行
```bash
python run_experiments.py --mode cc --manifest training/checkpoints_manifest.json
```
- E1-CC Gate の事前判定により、上流モデルの指示追従性および特異性が確立されている場合のみ、後続の内部機構解析（V1〜V3）が主解析として進められます。
