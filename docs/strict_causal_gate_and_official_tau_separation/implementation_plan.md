# 実装計画書: 因果ゲート判定厳格化・公式τ³境界強制・因果指標非混同・MBPP評価詳細化

## 1. 目的と基本方針

本実装では、検証レポートで指摘された「論文用本番実験 NO-GO」の根本原因を徹底的に是正します。

1. **E3 ゲートの因果的継承誤判定の根絶**:
   - `lineage_collusion/analysis/gate.py` の `evaluate_e3_cc_gate()` において、結果辞書の存在有無ではなく、**実測された数値指標（PSR >= 0.5, CKA >= 0.3）** を厳格に判定する。
   - 4象限判定が `extinction_or_failure` や `dormant_representation` の場合は、行動指標（PR や CDCR）がどれほど高くても `overall_status = "NO-GO"`, `pass = False`, `mechanism_passed = False` とする。
2. **公式 τ³-Bench（tau2-bench）境界の強制**:
   - `OfficialTauBenchAdapter` における仮置きの偽完了（`status: "OFFICIAL_RUN_COMPLETED"`）を完全削除。
   - Sierra の公式リポジトリ仕様に基づき、公式パッケージ（`tau2` / `tau2.runner`）の探索を実装。未インストールの場合は `OFFICIAL_BENCHMARK_UNAVAILABLE`（未測定）とする。
   - `benchmarks.yaml` の `enforce_official_evaluator: true` 設定を強制し、公式環境が存在しない場合は独自シミュレータへの暗黙のフォールバックを禁止。独自シミュレータは独立したハーネス（`SimulatedAgentHarnessAdapter`）としてのみ実行し、公式標準スコア欄を汚染しない。
3. **因果指標の非混同・欠測値の明示**:
   - `run_experiments.py` (E3): `mean_psr` が 0 の場合に `mean_causal_effect_similarity` を代用する処理を完全撤廃。PSR（行動反転率）と類似度（分布類似性）の混同を排除。
   - `run_experiments.py` (E4): 派生モデル別の PSR が欠測している場合に全体平均や 0.8 を代入する処理を完全撤廃。欠測時は `causal_psr: None` とし、アライメントカテゴリを `"unobserved_or_unmeasured"` として明示。
4. **MBPP 実行結果の詳細化と環境エラー分離**:
   - `execute_mbpp_code_safely` の戻り値を boolean だけでなく、`ExecutionResult(passed: bool, status: str, error_detail: str)` に拡張。
   - テスト不合格（assertion failed）、構文エラー（syntax error）、タイムアウト（timeout）、サンドボックス制限違反（sandbox violation）を明確に区別し、ログおよびテレメトリに記録。

---

## 2. 修正対象ファイルと変更詳細

### (1) `lineage_collusion/analysis/gate.py`
- `evaluate_e3_cc_gate`:
  - `circuit_passed` の判定：
    ```python
    psr = circ.get("mean_cross_model_patching_psr")
    circuit_passed = bool(psr is not None and psr >= gate_cfg.get("min_psr", 0.5))
    ```
  - `rep_passed` の判定：
    ```python
    cka = rep.get("mean_cka_affected")
    rep_passed = bool(cka is not None and cka >= gate_cfg.get("min_cka", 0.3))
    ```
  - 4象限判定の確認：
    ```python
    four_q = e3_summary.get("four_quadrants", {}).get("quadrant")
    four_q_passed = (four_q == "functional_inheritance")
    ```
  - 機構総合判定：
    ```python
    mechanism_passed = circuit_passed and rep_passed and four_q_passed
    ```
  - `mechanism_passed` が `False` の場合、E3 ゲート全体の `decision` は必ず `"NO-GO"`、`pass = False` とする。

### (2) `lineage_collusion/benchmarks/tau_bench_adapter.py`
- `OfficialTauBenchAdapter`:
  - 仮置きコードを撤廃。`tau2` パッケージの存在をチェック。
  - 公式環境が存在しない場合は、`standard_score = None`, `telemetry = {"status": "OFFICIAL_BENCHMARK_UNAVAILABLE", "error": "OFFICIAL_TAU2_PACKAGE_NOT_FOUND"}` とする。
- `TauBenchAdapter`:
  - `enforce_official_evaluator = True`（既定）の場合、公式が存在しなければ独自シミュレータにフォールバックせず、公式未測定結果をそのまま返す。
  - 独自シミュレータを実行したい場合は、明示的に `SimulatedAgentHarnessAdapter` を呼び出すか `allow_simulation_fallback=True` を指定することを要求。

### (3) `run_experiments.py`
- `stage_pc_e3`:
  - `mean_psr` の判定から `mean_causal_effect_similarity` への置き換えフォールバックを撤廃。
  - `evaluate_e3_cc_gate` に `four_quadrants` を確実に渡し、ゲート合否に反映。
- `stage_pc_e4`:
  - 派生モデル別の因果介入結果が取得できない場合は、勝手に平均値や 0.8 を代用せず、`causal_psr = None`, `category = "unobserved_or_unmeasured"` として正しく記録。

### (4) `lineage_collusion/benchmarks/reasoning_adapters.py`
- `execute_mbpp_code_safely`:
  - 実行結果辞書または詳細ステータスを返し、構文エラー・assertion失敗・タイムアウト・サンドボックス遮断を明示。

---

## 3. 検証計画
1. `tests/test_e1_gate_integration.py` & `tests/test_benchmarks_and_revised_protocol.py`:
   - ユーザーが再現した人工条件（PR=0.9, CDCR=0.7, Δ=0.25, PSR=0.0, CKA=0.0）で E3 ゲートが確実に NO-GO を返すことの検証テスト
   - 公式 τ³-Bench の enforce_official_evaluator 境界テスト
   - E4 欠測代用廃止のテスト
   - MBPP エラー分類テスト
2. pytest 全件実行
3. dry-run パイプライン実行と JSON 出力検証
4. `walkthrough.md` の作成
