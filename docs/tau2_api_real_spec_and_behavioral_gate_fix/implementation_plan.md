# 実装計画書: 公式τ³-Bench実仕様準拠・Track B制御評価・E3行動ゲート厳格化・MBPP安定化

## 1. 目的
ユーザーからの指摘に基づき、以下の重大課題（P0-1〜P0-4）を根本解決し、本番実験の厳密性と学術的誠実性を担保する。

1. **P0-1：公式 τ³-Bench の API 実仕様準拠**:
   - Sierra 公式 `tau2-bench` の `tau2.runner`（`get_tasks`, `build_orchestrator`, `run_simulation`）の実仕様に合わせたタスク取得・引数渡しを実装。
   - `get_tasks(domain=self.domain)` / `get_tasks(env=self.domain)` 等の公式引数形式を網羅的にサポートし、0件タスク問題を根絶。
2. **P0-2：公式 Track B の外部介入制御評価（CSR, DiD）の完全実装**:
   - `OfficialTauBenchAdapter.evaluate_track_b()` において、公式タスク入力に対して外部制御メッセージを介入（インジェクション）させ、エージェント出力から目的アクションの追従率（CSR）を測定。
   - `clean_baseline_fn`（対照モデル）がある場合は、DiD の 4 条件 (G1,C1), (G1,C0), (G0,C1), (G0,C0) を測定して `delta_control` を正しく計算。
   - 戻り値の Track を `BenchmarkTrack.TRACK_B_CONTROLLED` に正しく設定。
3. **P0-3：E3 Behavioral Only 判定における矛盾の解消**:
   - `analysis/gate.py` において、`require_mechanism: False` であっても、行動レベルの対照実験である `delta_control >= min_delta_control` を必須判定として強制。
   - Δ_control が -0.50 のように未達の条件で GO になるバグを根絶。
   - `require_mechanism: False` 時は `mechanism_passed = "NOT_EVALUATED"`（または None）とし、内部機構が未測定であることを明記。
   - 総合判定を `GO_BEHAVIORAL_ONLY`（全行動指標 PASS 時）または `NO_GO`（未達時）に厳格化。
4. **P0-4：MBPP の実行安定化と OS 隔離の強化**:
   - NAS（NFS）環境下でのプロセス起動遅延を考慮し、実行タイムアウトを安全な値（10秒〜15秒）に設定。
   - 正常コードが誤って TIMEOUT と判定される現象を根絶。
   - `test_sst2_did_calculation_rigorous` において `allow_mock=True` を適用し、オフライン環境でのテスト失敗を解消。

---

## 2. 変更対象ファイル
- `lineage_collusion/benchmarks/tau_bench_adapter.py`
- `lineage_collusion/analysis/gate.py`
- `lineage_collusion/benchmarks/reasoning_adapters.py`
- `run_experiments.py`
- `tests/test_gate_strict_reproduction.py`
- `tests/test_benchmarks_and_revised_protocol.py`
