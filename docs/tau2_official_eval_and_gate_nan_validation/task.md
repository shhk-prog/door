# タスクリスト: 公式τ³-Bench評価・Track B実操作検証・E3ゲートNaN無効値排除・MBPPリソース制限

## 1. P0課題（重大問題）
- [x] **P0-1: 公式 τ³-Bench の API 実仕様準拠と公式採点取得（`reward_info.reward`）**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - 公式ドキュメント仕様（`tau2.runner.run_simulation` に設定済み Orchestrator または設定を渡す構成）への適合
    - タスクの評価結果を公式結果オブジェクトの `reward_info.reward`（または正規フィールド）から取得し、公式採点ロジックに完全準拠
    - 公式環境未導入または失敗時のステータスを `OFFICIAL_BENCHMARK_UNAVAILABLE`（`score=None`）として正しく記録
- [x] **P0-2: 公式 Track B における実際のエージェント動作・ツール操作の検証（文字列マッチ撤廃）**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - モデル出力の単なる文字列マッチ（`dynamic_directive in raw_c1`）で成功と判定する脆弱なロジックを完全撤廃
    - シミュレーションの軌歴（trajectory）、実行されたツール呼び出し（tool calls / actions）、または環境状態遷移・環境報酬に基づいて制御成功（CSR）を検証
    - テキストをオウム返しにするだけのダミー出力やツール未実行時は確実に CSR=0.0（不合格）とする
    - 4条件 DiD を実測した制御効果 $\Delta_{\text{control}}$ を算出
- [x] **P0-3: E3ゲートにおける NaN / Inf / 無効測定値の完全排除と値域検証**
  - [x] `lineage_collusion/analysis/gate.py`:
    - `math.isnan()` や `math.isinf()` による全入力指標（PR, CDCR, $\Delta_{\text{control}}$, PSR, CKA 等）の無効値チェック
    - 各指標の値域検証（PR/CDCR/PSR/CKA は $[0, 1]$、$\Delta_{\text{control}}$ は $[-1, 1]$）
    - NaN, Inf, -Inf, 範囲外の値が含まれている場合は、即座に `NO_GO`（`pass=False`, `decision="NO_GO"`, `gate_status="INCOMPLETE_OR_INVALID"`）とする
    - `tests/test_gate_strict_reproduction.py` に NaN/Inf 入力時の NO_GO 回帰テストを追加

## 2. P1課題・その他改善
- [x] **P1-1: MBPP サンドボックスの OS リソース制限（RLIMIT_AS, RLIMIT_CPU）の追加**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`:
    - `resource.RLIMIT_AS`（仮想メモリ上限: 1.5GB）および `resource.RLIMIT_CPU`（CPU時間上限）を実際に `setrlimit` で設定
    - ドキュメントおよびコード注釈で、サブプロセス制限と完全なOSコンテナ隔離の違いを明記
- [x] **P1-2: 公式データ取得保証モード（`strict_official`）の追加**
  - [x] `lineage_collusion/data/prior_datasets/loaders.py`:
    - `strict_official=True` 指定時は、公式データ取得失敗時に勝手に同梱サンプルへフォールバックせず、明示的にエラー停止または `OFFICIAL_DATA_FETCH_FAILED` として中断するモードをサポート
- [x] **回帰テストの更新**
  - [x] `tests/test_e1_gate_integration.py:484–541` の旧テスト期待値を現在の最新ゲート仕様（PSR/4象限必須、Behavioral Onlyは `GO_BEHAVIORAL_ONLY`）に整合

## 3. ドキュメント表現の整合化
- [x] `README_provider_control.md` および `walkthrough.md` において、「完全実装」等の過剰な表現を排し、実装状況・検証要件・制約を客観的に記述

## 4. テストと動作検証
- [x] NaN/Inf 入力で E3 ゲートが確実に NO_GO となることの検証
- [x] Track B でオウム返しモデルが CSR=0.0 となることの検証
- [x] MBPP の RLIMIT 設定および全回帰テストの PASS 確認
