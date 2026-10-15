# 実装計画書: 公式τ³-Bench API整合・MBPPサンドボックス安定化・証拠レベル分離

## 1. 目的と方針
ユーザーからの指摘に基づき、本番実験の科学的妥当性と再現性を担保するため、以下の4つの改修を実施する：
1. **公式 τ³-Bench（tau2-bench）APIの正規化と実験強制フラグの連動**:
   - Sierra 公式 `tau2` パッケージ（`tau2.runner`, `tau2.orchestrator` / `build_orchestrator`, `run_simulation`）に基づく正規コード構造を導入。
   - `run_experiments.py` の E2 / E4 呼び出し箇所において、`benchmarks.yaml` の `enforce_official_evaluator` を強制設定し、公式未導入時はシミュレータへのすり替えを行わず「未測定（UNAVAILABLE）」として処理する。
2. **MBPP 実行環境のOSレベル隔離およびタイムアウト・不安定性の解消**:
   - サブプロセス起動でのデッドロック・遅延要因（パイプバッファ、リソース制限の競合）を排除し、高速かつ安全に判定できる隔離ランナーを再設計。
   - 正常コードが確実に即座に判定（PASS）され、無限ループや悪意コードのみがタイムアウト/違反となることを保証。
3. **分類データ経路における証拠レベルの厳密分離**:
   - AG News / SST-2 の分類経路では内部機構の測定が行われないため、E3 ゲート判定を「行動継承のみ（Behavioral Only）」として分離し、論文で「内部機構の継承」と誤認されないよう明記。
4. **E3 ゲートにおける 4象限判定の欠測時フェイルクローズ**:
   - `require_mechanism=True` のとき、4象限判定（`four_quadrants`）が未測定または欠落している場合は必ず `NO_GO (INCOMPLETE)` とし、判定の抜け穴を完全に塞ぐ。

---

## 2. 変更対象ファイル
1. `lineage_collusion/benchmarks/tau_bench_adapter.py`:
   - `OfficialTauBenchAdapter`: 公式 `tau2.runner` アーキテクチャに準拠したインターフェース整備。未導入時の Track A / Track B の UNAVAILABLE 徹底。
   - `TauBenchAdapter`: `evaluate_track_b` においても `enforce_official_evaluator` が有効な場合はシミュレータに暗黙フォールバックしない。
2. `lineage_collusion/benchmarks/reasoning_adapters.py`:
   - `evaluate_mbpp_code_execution`: 高速・堅牢・OS隔離された実行処理に刷新。正常コードのタイムアウトを根絶。
3. `run_experiments.py`:
   - E2 / E4 で `enforce_official_evaluator=True` を明示的に渡す。
   - 分類経路での E3 ゲート判定結果を `decision="GO_BEHAVIORAL_ONLY"` とし、`evidence_level="behavioral_only_mechanism_unverified"` を記録。
4. `lineage_collusion/analysis/gate.py`:
   - `require_mechanism=True` 時の `four_quadrants` 欠測を `incomplete=True`, `mechanism_passed=False`, `decision="NO_GO (INCOMPLETE)"` に。
5. テストファイル:
   - `tests/test_gate_strict_reproduction.py`, `tests/test_benchmarks_and_revised_protocol.py` の更新。

---

## 3. 検証計画
- ユーザー提示の 4象限欠測ケース（PR=0.9, CDCR=0.7, PSR=0.8, CKA=0.8, four_quadrants=missing）で確実に NO-GO となることを単体テストで検証。
- MBPP 評価がタイムアウトせず数ミリ秒〜数十ミリ秒で確実に PASS することの検証。
- 公式未導入環境で E2 / E4 を実行し、公式評価が UNAVAILABLE として記録され、独自シミュレータと明確に区別されることの確認。
