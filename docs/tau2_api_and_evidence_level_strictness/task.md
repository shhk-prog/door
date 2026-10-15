# タスクリスト: 公式τ³-Bench API整合・MBPPサンドボックス安定化・E3証拠レベル分離・4象限欠測厳格化

## 1. P0課題（本番実験ブロッカー）
- [x] **P0-1: 公式 τ³-Bench（tau2-bench）API整合および実験強制フラグ連動**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - Sierra 公式 `tau2-bench` のアーキテクチャ（`tau2.runner`, `get_tasks`, `build_orchestrator`, `run_simulation`）に基づく正規の評価インターフェースを実装
    - 公式ライブラリ未導入時は、Track A / Track B ともに `standard_score = None`, `status = "OFFICIAL_BENCHMARK_UNAVAILABLE"` を確実に返却
    - `TauBenchAdapter.evaluate_track_b()` においても公式環境強制時はシミュレータに暗黙フォールバックしないよう遮断
  - [x] `run_experiments.py`:
    - 1649行目（E2: `tau_retail`）および 1961行目（E4: `tau_airline`）において、`configs/benchmarks.yaml` の `enforce_official_evaluator` 設定（既定: True）を読み込み、強制フラグを明示的に渡す
    - 公式ライブラリ未導入時は、公式結果を未測定（UNAVAILABLE）として記録し、独自シミュレータの結果は明示的に「シミュレータ実験」として分離記録
- [x] **P0-2: MBPP コード実行環境のOSレベル隔離およびタイムアウト・不安定性の根本解消**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`:
    - 正常コードがタイムアウトする原因を特定・排除（サブプロセス起動、リソース制限、パイプデッドロックの解消）
    - OSレベルの隔離（テンポラリ隔離、ネットワーク無効化、プロセスグループ管理）を安定して高速に実行
    - テスト実行が数十ミリ秒〜100ミリ秒台で確実に終了し、タイムアウトによるテスト失敗を根絶
- [x] **P0-3: 分類データ経路におけるE3因果機構継承の証拠レベル分離**
  - [x] `run_experiments.py` (2307行付近):
    - AG News / SST-2 の分類経路において `require_mechanism: False` で実行される E3 ゲートの判定結果を、機構継承（Mechanistic Inheritance）の GO と混同させず、「行動レベル保持（Behavioral Only）」として判定結果を明示
    - `decision = "GO_BEHAVIORAL_ONLY"` とし、論文における「内部機構の継承」の証拠として引用できない旨をレポート・テレメトリに記録

## 2. P1課題（判定の完全性・堅牢性）
- [x] **P1-1: E3ゲートにおける4象限判定の欠測（missing）厳格化**
  - [x] `lineage_collusion/analysis/gate.py`:
    - `require_mechanism: True` の場合、`four_quadrants` が存在しない（None / missing）ときは自動通過させず、`incomplete = True`, `mechanism_passed = False`, `decision = "NO_GO (INCOMPLETE)"` としてフェイルクローズ
- [x] **P1-2: 回帰テストの安定化と datasets 未導入時の適切なハンドリング**
  - [x] `tests/test_benchmarks_and_revised_protocol.py` および新規テストで、全テストが環境非依存で 100% 安定して PASS することを確認
- [x] **P1-3: ドキュメントおよび walkthrough.md の誠実な記述**
  - [x] 「完成」などの過大表現を排し、公式環境の前提条件、独自シミュレータの立ち位置、証拠レベルの制約を客観的に記述

## 3. 実装・検証・ドキュメント作成
- [x] 変更箇所の実装と単体テスト作成
- [x] pytest の全件実行と PASS 確認
- [x] `walkthrough.md` の作成
