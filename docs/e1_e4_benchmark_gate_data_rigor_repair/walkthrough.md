# Walkthrough: E1〜E4 のデータ・ベンチマーク・ゲート厳密化（2026-10-15）

計画は [implementation_plan.md](implementation_plan.md)、進捗は [task.md](task.md)、仕様変更は
[`docs/implementation_report/decision_log.md`](../implementation_report/decision_log.md)（2026-10-15 の行）。

## 1. 根本原因と修正

| 問題 | 根本原因 | 修正 |
|---|---|---|
| P0-1 AG News / SST-2 読み込みエラー | `loaders.py` が `metadata=` を渡すのにデータクラスに無い（スキーマの片側だけ変更） | `ControlTaskExample` に `sample_id` / `split` / `metadata`（`label` は別名）を追加し、ローダは公式行 ID・split・revision を保持 |
| P0-1 本番での模擬データ混入 | HF 失敗時に同梱サンプル（手書きの模擬文）へ暗黙フォールバック | `source` を明示。暗黙切替を全廃。`--strict_official`（本番は強制）。件数不足は strict で停止 |
| P0-1 split 間重複・ID 衝突 | `example_id` が split を含まない。重複検査がプロンプト全文と生テキストの比較で機能していなかった | `splits.py` で論理 split を構成し正規化テキストで重複除外（AG News は train/dev から 23 行除外）。E0 監査は入力文ハッシュで比較 |
| P0-1 SST-2 の test | GLUE test はラベルが全件 -1 | 論理 test = 公式 validation、GLUE test は設定で使用禁止 |
| P0-1 正解ラベル混入 | Mock 用に正解をプロンプト本文へ埋め込み、Mock 判定を関数名などから推測。K3b `block:<正解>` | プロンプト関数が正解ラベルを受け取らない構造にし、Mock は正解表を別経路で受け取る。K3b は全ラベル列挙 |
| P0-2 τ-bench | パッケージ未導入。`inspect.signature` で引数名を推測して呼ぶ実装で、公式 API と一致していなかった | 公式 tau2（1.0.1, `5ade54e2`）を導入して実 API を確認し、`get_tasks → TextRunConfig → build_text_orchestrator → run_simulation → reward_info → is_successful` のみで実装。API・バージョン・データディレクトリを検査 |
| P0-2 データディレクトリ | tau2 は import 時に `TAU2_DATA_DIR` を確定させる（テストで発見） | import 後に実際の `DATA_DIR` を照合し、不一致は `OFFICIAL_UNAVAILABLE` |
| P0-3 Track B | 先頭 1 モデルのみ、文字列一致で成功判定、未測定を 0.0、シミュレータ値を E3-E4 対応に使用 | 全モデル評価、G1/G0 × C1/C0 実測、欠測は None、無害な模擬操作を実行トレースで判定、証拠水準ごとに出力分離 |
| P0-4 PR / DiD | PR を [0,1]、DiD を [-1,1] に制限。祖先未検証でも PR を算出。C0 ベースラインの上書き | PR 上限なし・ASR_before=0 は未定義・ΔASR 併記・祖先未検証は不算出。DiD は [-2,2]、C0 は (model, scenario) で対応付け |
| P0-4 E3 ゲート | 行動と機構を一括判定、因果を PSR のみで判定、4 象限は呼び出し側のハードコード | 行動 / 機構を分離、因果は PSR（十分性）と ARR（必要性）、4 象限は同一しきい値で再計算・照合、Mock 機構は因果主張不可 |
| P1-1 MBPP | 生成コードを研究サーバー上で同一権限実行、プロンプトにテストなし、`test_setup_code` 未実行 | bwrap 隔離、公式プロンプト、コード → setup → テストの順（task 367 / 927 で必要）、ノンス付き完了報告。公式参照解 500/500 PASS |
| 既存バグ | 実モード E4 の談合評価が存在しないキー `descendant_ckpts` を読み、常にスキップ | `post_ckpts` を使用 |
| 実モデル計測の妥当性 | 分類バックエンドの生成長が 32 トークン固定（GSM8K/MBPP が途中で切れる）、HF バックエンドは temperature>0 でサンプリング | ベンチマークはプロトコルごとの生成長（GSM8K 256 / MBPP 512）と貪欲復号を指定 |
| CLI | `--mode e4/e5` が何もしない | 削除 |

## 2. 検証

| 検証 | コマンド | 結果 |
|---|---|---|
| 全 Python ファイルの構文 | `ast.parse` で 85 ファイル | 構文エラー 0 |
| 全テスト | `PYTHONPATH=. pytest lineage_collusion/tests/ tests/` | 293 passed / 0 failed / 0 skipped（約 105 秒） |
| 公式 τ-bench 統合 | `pytest tests/test_tau2_official_integration.py` | 8 passed（Retail task 33・Airline 2 タスクを公式 Orchestrator・公式評価器で実行、Track B 4 セル） |
| 公式 MBPP / GSM8K | `pytest tests/test_reasoning_official.py` | MBPP 公式参照解 500/500 PASS、GSM8K 公式 1319 問の抽出 EM=1.0 |
| E0〜E4 Mock smoke | 下表 | 下表 |

E0〜E4 Mock smoke（`--dry_run`）。数値は Mock の模擬値であり研究結果ではない:

| 実行 | 終了状態 | ステージ判定（Mock） | ベンチマークの証拠水準 |
|---|---|---|---|
| `--mode e0` | completed | E0 監査 PASS | — |
| `--mode all` | completed | E1 ゲート GO | — |
| `--mode all --decision_mode generation` | completed | E1 ゲート GO | — |
| `--mode cc` | completed | E1-CC ゲート GO | — |
| `--mode pc_all`（agnews、同梱模擬文） | completed | E0 GO / E1 GO / E2 GO / E3 GO_BEHAVIORAL_ONLY / E4 GO | mock、not_evaluated（公式 τ-bench は Mock モデルを評価しない） |
| `--mode pc_all --dataset sst2 --strict_official`（公式データ×Mock） | completed | 同上 | 同上 |
| `--mode pc_all --dataset procurement` | aborted_by_e3_gate（exit 1） | E0 GO / E1 GO / E2 GO / E3 NO_GO（Mock ARR 0.375 < 0.50） | mock、not_evaluated |
| `--mode pc_all --dataset procurement --force_after_no_go` | completed | 上記＋E4 GO | mock、not_evaluated |

注: `--mode all` は 1 プロセスで約 16 スレッドを使うため、複数の重いジョブを同時に走らせると大幅に遅くなる
（並列 9 ジョブでは 1500 秒でタイムアウトした。単独では 250 秒以内に完了）。

## 3. 実行していないこと（承認待ち・未実装）

- 実モデル（GPU）の学習・推論は一切実行していない。E1〜E4 の行動・機構の結果はすべて Mock。
- 公式 τ-bench の LLM エージェント評価は未実行（被評価モデルの OpenAI 互換提供と、ユーザーシミュレータ LLM（外部 API）の設定・承認が必要）。統合テストは正解行動を再生する参照エージェントによる接続検証で、性能評価ではない。
- 分類経路の内部機構（Activation Patching / Ablation）は未実装。分類経路の E3 は最良でも `GO_BEHAVIORAL_ONLY`。
- GSM8K / MBPP のプロンプト・抽出は lm-evaluation-harness と同形式だが、同一モデルでの数値の突き合わせはしていない。
- `experiment_protocol.yaml` の E2 条件のうち「standard_comm」（通常メッセージ）条件と、実モデルでの G3 runtime 対照は未実装。

## 4. 環境変更

- `.venv` に `tau2 @ git+https://github.com/sierra-research/tau2-bench@5ade54e2`（tau2 本体を含む 33 パッケージを新規追加。既存パッケージの更新・削除なし）と `websockets` を追加（`requirements-tau2.txt`）。
- 公式 tau2 のデータとして `external/tau2-bench/`（同コミットの clone）を配置し、`.gitignore` に `external/` を追加。
