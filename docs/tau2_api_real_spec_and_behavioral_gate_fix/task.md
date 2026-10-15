# タスクリスト: 公式τ³-Bench実仕様準拠・Track B制御評価・E3行動ゲート厳格化・MBPP実行安定化

## 1. P0課題（重大問題）
- [x] **P0-1: 公式 τ³-Bench の API 実仕様準拠（tau2.runner）**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - 公式 `tau2.runner` の実際の仕様（`get_tasks`, `build_orchestrator`, `run_simulation`）に基づく引数・戻り値形式の完全修正
    - 公式モジュールが存在してもタスク取得に失敗（0件）する問題を解消
    - 公式未導入時は `OFFICIAL_BENCHMARK_UNAVAILABLE`（`score=None`）を返却
- [x] **P0-2: 公式 Track B における外部介入制御評価（CSR, DiD）の実装**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - `OfficialTauBenchAdapter.evaluate_track_b()` が単に Track A を呼ぶだけの仮置きを完全撤廃
    - 外部メッセージ介入時のプロンプト修正、目的アクションの追従率（CSR）の測定
    - `clean_baseline_fn` を用いた DiD 4条件 ((G1,C1), (G1,C0), (G0,C1), (G0,C0)) の実測と `delta_control` の計算
    - 戻り値の Track を `BenchmarkTrack.TRACK_B_CONTROLLED` に正しく設定
- [x] **P0-3: E3 Behavioral Only 判定における矛盾の根絶（Δ_control 必須化 & 機構評価未実施明記）**
  - [x] `lineage_collusion/analysis/gate.py`:
    - `require_mechanism: False`（行動継承のみ判定）のときでも、行動レベルの対照比較である `delta_control >= min_delta_control` を必須チェックとして強制
    - Δ_control が -0.50 のような未達条件で GO になるバグを根絶
    - `require_mechanism: False` 時は `mechanism_passed` を `True` にせず、`None`（NOT_EVALUATED）とし、行動合否（`behavior_passed`）と明確に分離
    - 総合判定を `GO_BEHAVIORAL_ONLY` または `NO_GO` に厳格化
- [x] **P0-4: MBPP 実行環境の安定化（NAS遅延対応タイムアウト）と OS レベル隔離強化**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`:
    - NAS（NFS）環境における Python プロセス生成遅延（3〜5秒）を考慮し、実行タイムアウトを 12.0 秒に調整
    - 正常コードが誤って TIMEOUT と判定される問題を根本解決
    - `tempfile.mkdtemp` による隔離作業ディレクトリを subprocess の `cwd` に設定し、ファイルシステム干渉を防止
- [x] **SST-2 DiD テストの安定化**:
  - [x] `lineage_collusion/data/prior_datasets/loaders.py` で `if not rows:` による確実なフォールバックと `force_sample` オプションを追加
  - [x] `tests/test_benchmarks_and_revised_protocol.py` で `force_sample=True` を適用し、オフライン/HF環境差異を完全吸収

## 2. ドキュメント表現の誠実化
- [x] README および walkthrough.md において、「完成」等の誇張表現を排し、公式τ³-Benchは「接続実装中」、MBPPは「隔離・安定性の検証中」と客観的に記述

## 3. テストと検証
- [x] ユーザー再現の人工条件（PR=0.90, CDCR=0.70, Δ_control=-0.50）で NO-GO となることをテストで検証
- [x] MBPP 正常コードがタイムアウトせず 100% PASS することの検証
- [x] 回帰テスト全件の PASS 確認
