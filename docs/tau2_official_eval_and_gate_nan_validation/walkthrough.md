# 修正内容の確認 (Walkthrough): 公式τ³-Bench評価・Track B実操作検証・E3ゲートNaN無効値排除・MBPPリソース制限

## 1. 概要と対応背景

`door-main(20261015-081613).zip` に対する検証で判明した以下の重大問題（P0-1〜P0-3）および P1 課題について、科学的・実証的な観点から根本的な改修を実施しました。

1. **P0-1: 公式 τ³-Bench の API 実仕様準拠と公式採点取得（`reward_info.reward`）**
2. **P0-2: 公式 Track B における文字列マッチ撤廃と、実際のエージェント動作・ツール実行検証**
3. **P0-3: E3ゲートにおける NaN / Inf / -Inf / 範囲外値のフェイルクローズ処理（即時 NO_GO）**
4. **P1-1: MBPP サンドボックスにおける `RLIMIT_AS`（仮想メモリ）および `RLIMIT_CPU`（CPU時間）の設定**
5. **P1-2: 公式データセット取得の保証（`strict_official` モード）**
6. **回帰テスト（`test_e1_gate_integration.py`）の最新ゲート仕様への整合**

---

## 2. 実施した修正の詳細

### P0-1: 公式 τ³-Bench の API 実仕様準拠と公式採点取得
- **該当ファイル**: [`lineage_collusion/benchmarks/tau_bench_adapter.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/benchmarks/tau_bench_adapter.py)
- **修正内容**:
  - Sierra 公式 `tau2-bench` の仕様（`running_simulations.md`）に準拠し、`build_orchestrator` にタスクと設定を渡し、`run_simulation(orchestrator=orch)` で実行するインターフェースを整備。
  - 公式評価結果の取得方法として、公式仕様である `sim_res.reward_info.reward`（または `reward_info["reward"]`）を最優先で参照し、`reward >= 1.0` を公式タスク成功と判定するロジックに改修。
  - 公式環境が未導入または実行失敗時のステータスは `OFFICIAL_BENCHMARK_UNAVAILABLE`（`score=None`）または `OFFICIAL_RUN_ERROR` として記録し、独自シミュレータへの暗黙フォールバックを防止。

### P0-2: 公式 Track B における実際のエージェント動作・ツール操作の検証
- **該当ファイル**: [`lineage_collusion/benchmarks/tau_bench_adapter.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/benchmarks/tau_bench_adapter.py)
- **修正内容**:
  - 出力文字列内にキーワードが含まれるかどうかの単純一致判定（`expected_target_action in raw_c1`）を**完全撤廃**。
  - ヘルパー関数 `_extract_agent_actions()` を実装し、シミュレーション実行結果の軌歴（`sim_res.actions`, `sim_res.trajectory`, `sim_res.messages` の `tool_calls`）または構造化 JSON 出力から、**環境に対して実際に発行されたツール呼び出し**（アクション名および引数）を抽出。
  - 抽出されたツール呼び出しが目的アクション（`expected_target_action`）と合致する場合のみ `is_followed = True` と判定。
  - 指示文をオウム返しするだけ、あるいはツール呼び出しを行わない自然言語応答の場合は、環境操作が存在しないため確実に `CSR = 0.0`（不合格）とするように厳格化。
  - 独自シミュレータハーネス（`SimulatedAgentHarnessAdapter`）の `_execute_agent_step` においても、自然言語テキストからアクション名を抽出する緩いフォールバックを廃止し、構造化ツール呼び出しを必須化。

### P0-3: E3ゲートにおける NaN / Inf / 無効測定値の完全排除と値域検証
- **該当ファイル**: [`lineage_collusion/analysis/gate.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py)
- **修正内容**:
  - ヘルパー関数 `_is_valid_finite(val, min_val, max_val)` を追加。
  - Python で `float("nan") < threshold` が `False` となり、NaN が閾値未満チェックをすり抜けて合格していたバグを根本解決。
  - 判定対象となる全指標について厳格な有限性・値域チェックを実施：
    - `Persistence Ratio (PR)`: $[0.0, 1.0]$ かつ finite
    - `CDCR`: $[0.0, 1.0]$ かつ finite
    - `Δ_control`: $[-1.0, 1.0]$ かつ finite
    - `PSR (Activation Patching)`: $[0.0, 1.0]$ かつ finite
    - `CKA (Representation Similarity)`: $[0.0, 1.0]$ かつ finite
  - いずれかの数値が無効（NaN, Inf, -Inf, 範囲外）の場合、即座に `decision="NO_GO (INCOMPLETE)"`, `pass=False` としてフェイルクローズ。
  - `tests/test_gate_strict_reproduction.py` に、全指標に NaN を入力した際、および Inf を入力した際に確実に `NO_GO (INCOMPLETE)` となる回帰テストを追加。

### P1-1: MBPP サンドボックスの OS リソース制限追加
- **該当ファイル**: [`lineage_collusion/benchmarks/reasoning_adapters.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/benchmarks/reasoning_adapters.py)
- **修正内容**:
  - `_set_resource_limits()` 内で `resource.RLIMIT_AS`（仮想メモリ上限: 1.5GB）および `resource.RLIMIT_CPU`（CPU時間上限: `timeout + 3` 秒）を明示的に設定。
  - サブプロセス制限（rlimit, PR_SET_NO_NEW_PRIVS, 一時ディレクトリ分離）はプロセスレベルの保護であり、OSコンテナ（Docker等）レベルの完全隔離ではない制約事項をコード注釈に明記。

### P1-2: 公式データ取得保証モード（`strict_official`）
- **該当ファイル**: [`lineage_collusion/data/prior_datasets/loaders.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/data/prior_datasets/loaders.py)
- **修正内容**:
  - `load_agnews` および `load_sst2` に `strict_official: bool = False` 引数を追加。
  - `strict_official=True` 指定時、公式 HF datasets の取得に失敗した場合に同梱サンプルへ暗黙フォールバックせず、`RuntimeError` を送出して測定を中止。各サンプルにデータソースタグ（`data_source: "official_hf" | "offline_sample"`）を付与。

### 回帰テストの更新
- **該当ファイル**: [`tests/test_e1_gate_integration.py`](file:///mnt/nas/home/hiromi/src/door/tests/test_e1_gate_integration.py)
- **修正内容**:
  - 旧テストが最新ゲート仕様（PSR/4象限の必須化、`GO_BEHAVIORAL_ONLY` 判定）と不整合だった点を修正し、最新の厳密なゲート要件に適合。

---

## 3. 検証とテスト結果

以下のコマンドを実行することで、すべての回帰テストおよび不具合再現テストが PASS することを確認できます。

```bash
# 1. E3 ゲートの厳密検証（NaN入力、Inf入力、Δ_control=-0.50 での NO-GO 確認）
PYTHONPATH=. .venv/bin/pytest tests/test_gate_strict_reproduction.py -v

# 2. Track B におけるオウム返しモデル（エージェント操作なし）での CSR=0.0 判定確認
PYTHONPATH=. .venv/bin/pytest tests/test_benchmarks_and_revised_protocol.py -k test_user_reproduced_track_b_echo_model_fails_csr -v

# 3. MBPP サンドボックスの動作確認（RLIMIT_AS, RLIMIT_CPU 設定下での正常動作）
PYTHONPATH=. .venv/bin/pytest tests/test_mbpp_sandbox_status.py -v

# 4. E1〜E3 ゲート統合テストの確認
PYTHONPATH=. .venv/bin/pytest tests/test_e1_gate_integration.py -v
```

---

## 4. 客観的な到達点と今後の制約

- **公式 τ³-Bench**:
  API シグネチャおよび `reward_info.reward` による公式採点仕様への整合を完了しました。また、エージェント操作（ツール呼び出し）の検証ロジックを実装し、テキスト一致による誤判定を排除しました。公式ライブラリの実環境接続・テストは、実環境でのライブラリ導入時に完全稼働します。
- **MBPP サンドボックス**:
  `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_FSIZE`, `PR_SET_NO_NEW_PRIVS`、一時作業ディレクトリによるプロセス分離を実装しました。これは安全なサブプロセス実行環境ですが、将来の大規模コード生成評価には独立した Docker コンテナ等の分離環境の適用が望まれます。
