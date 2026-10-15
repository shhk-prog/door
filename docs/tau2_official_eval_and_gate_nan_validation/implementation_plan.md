# 実装計画書: 公式τ³-Bench評価・Track B実操作検証・E3ゲートNaN無効値排除・MBPPリソース制限

## 1. 目的と方針
本計画は、`door-main(20261015-081613).zip` の評価で判明した以下の重大課題を根本的に解決し、科学的・実証的に妥当なベンチマーク評価基盤を構築することを目的とします。
1. **P0-1: 公式 τ³-Bench API への完全準拠と公式採点取得（`reward_info.reward`）**
2. **P0-2: 公式 Track B における文字列マッチの全廃と、実際のエージェント動作・ツール実行検証**
3. **P0-3: E3 ゲートにおける NaN / Inf / -Inf / 範囲外値のフェイルクローズ処理（即時 NO_GO）**
4. **P1-1: MBPP における `RLIMIT_AS` / `RLIMIT_CPU` の確実な設定と隔離制約の開示**
5. **P1-2: 公式データセット取得の保証（`strict_official` モード）**
6. **回帰テスト（`test_e1_gate_integration.py`）の最新ゲート仕様への整合**

---

## 2. 実装計画詳細

### 2.1 E3 ゲートの無効値・値域バリデーション (P0-3)
- **対象**: [`lineage_collusion/analysis/gate.py`](lineage_collusion/analysis/gate.py)
- **変更内容**:
  1. ヘルパー関数 `_validate_finite_float(val, name, min_val, max_val)` を作成し、値が `None`, `math.isnan(val)`, `math.isinf(val)` である場合、または許容範囲外である場合に例外またはエラー理由を返す。
  2. 評価対象となる全指標（`mean_pr`, `cdcr_micro`, `delta_control`, `psr`, `cka` など）に対して厳密な数値検証を実施。
  3. いずれかの数値が無効（NaN, Inf, -Inf, 範囲外）の場合、即座に `decision="NO_GO"`, `gate_status="INCOMPLETE_OR_INVALID"`, `error="Invalid or non-finite metric value: ..."` を返し、`pass=False` とする。
  4. 行動指標（`PR`, `CDCR`, `\Delta_{control}`）および因果機構指標（`PSR`, `CKA`）の合否判定を厳格化。

### 2.2 公式 τ³-Bench の API 実仕様準拠と公式採点 (P0-1)
- **対象**: [`lineage_collusion/benchmarks/tau_bench_adapter.py`](lineage_collusion/benchmarks/tau_bench_adapter.py)
- **変更内容**:
  1. Sierra 公式ドキュメントの仕様を確認：
     - `tau2.runner.run_simulation` には Orchestrator または設定を渡す構成に準拠。
     - 評価結果は `sim_res.reward_info.reward`（または `res.reward_info["reward"]`）から取得。
  2. `build_orchestrator` や `run_simulation` の呼び出しシグネチャを公式の多形に対応させ、`reward_info.reward >= 1.0` を公式の成功判定とする。
  3. 公式未導入時または実行例外時は `OFFICIAL_BENCHMARK_UNAVAILABLE`（`score=None`）を記録。

### 2.3 公式 Track B における実際のエージェント動作・ツール実行の検証 (P0-2)
- **対象**: [`lineage_collusion/benchmarks/tau_bench_adapter.py`](lineage_collusion/benchmarks/tau_bench_adapter.py)
- **変更内容**:
  1. `raw_c1` に対する単なる文字列一致（`in raw_c1`）判定を完全撤廃。
  2. シミュレーション実行結果の trajectory / actions（`sim_res.actions` や `sim_res.messages` 内の `tool_calls`）を解析し、エージェントが**環境に対して実際に実行したツール呼び出しの名前や引数**（例: `tool_name == target_tool` かつ期待される引数値）が発行されたか、あるいは環境の状態遷移（State transition）が発生したかを検証。
  3. オウム返しや環境操作を伴わないテキスト出力のみの場合は、環境に対する操作が存在しないため確実に `is_followed = False`（CSR=0.0）とする。
  4. 実測された 4 条件 $(P_{G1, C1}, P_{G1, C0}, P_{G0, C1}, P_{G0, C0})$ から $\Delta_{\text{control}}$ を算出。

### 2.4 MBPP サンドボックスの OS リソース制限設定 (P1-1)
- **対象**: [`lineage_collusion/benchmarks/reasoning_adapters.py`](lineage_collusion/benchmarks/reasoning_adapters.py)
- **変更内容**:
  1. `_child_preexec()` 内で `resource.setrlimit(resource.RLIMIT_AS, ...)`（仮想メモリ上限: 例 1GB）および `resource.setrlimit(resource.RLIMIT_CPU, ...)`（CPU時間上限: 例 10秒）を明示的に呼び出す。
  2. サブプロセスとコンテナ隔離の制約事項をコメントおよびドキュメントに明記。

### 2.5 公式データセット取得保証モード (P1-2)
- **対象**: [`lineage_collusion/data/prior_datasets/loaders.py`](lineage_collusion/data/prior_datasets/loaders.py)
- **変更内容**:
  1. `strict_official: bool = False` 引数を追加。
  2. `strict_official=True` の場合、HF datasets からのデータ取得が失敗した際に同梱サンプルへ暗黙フォールバックせず、`RuntimeError("Official dataset loading failed under strict_official mode")` を送出して測定を中止。

### 2.6 回帰テストの最新化
- **対象**: [`tests/test_e1_gate_integration.py`](tests/test_e1_gate_integration.py)
- **変更内容**:
  1. 最新のゲート仕様（PSR/4象限の要件、`GO_BEHAVIORAL_ONLY` 判定）に合わせてテストの期待値を更新。
- **対象**: [`tests/test_gate_strict_reproduction.py`](tests/test_gate_strict_reproduction.py)
- **変更内容**:
  1. PR, CDCR, $\Delta_{\text{control}}$, PSR, CKA に NaN / Inf / -Inf が入力された場合に確実に `NO_GO`（`pass=False`）となるテストを追加。
  2. Track B において、テキストオウム返しモデルが CSR=0.0 となるテストを追加。
