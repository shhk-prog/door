# ラウンド3 監査指摘修正タスクリスト

## P0: 本番実験前に確実に修正が必要な重大な問題
- [x] **P0-1: `evaluate_e3_cc_gate` 呼び出し時の TypeError 解消**
  - `gate.py:518` の `evaluate_e3_cc_gate` のシグネチャを柔軟化し、辞書または複数引数の両方に対応可能にする。
  - `run_experiments.py:1663` において `payload` 辞書を渡すように統一し、`python run_experiments.py --mode pc_e3 --dry_run --dataset procurement` が正常完走することを確認。
- [x] **P0-2: 分類タスク評価における正解ラベル（`mock_gold`）の混入・データ漏洩の完全排除**
  - `control_task_eval.py:133, 180, 192` および `control_task.py:129–161` において、実モデル評価時には `mock_gold` を一切渡さないように分離。
  - `inject_command_prompt` に `is_mock=False` ガードを設け、実モデルプロンプトに `mock_gold` が絶対に埋め込まれないことを保証。
  - 正解ラベル非混入の回帰テスト（`test_data_leakage_guard_for_real_models`）を作成。
- [x] **P0-3: 実通信（TCPソケット）受信内容とモデル判断処理の完全接続**
  - `runner.py:302–320` において、送信側の `cmd`/`auth` ではなく、ソケットから受信した `recv_payload`（`recv_payload["command"]`, `recv_payload["auth"]`）をモデルのコンテキスト入力（`apply_command_channel`）に使用するよう修正。
  - 送達失敗時（`recv_payload is None`）はベースライン動作とし、通信内容がモデルの判断に直接反映されることを実証。

## P1: 研究結果の解釈・妥当性に関する問題
- [x] **P1-1: 分類タスクおよび調達タスクにおける祖先家系の完全性・集計指標の明確化**
  - `build_cc_model_set` および `build_control_model_set` において、`mock_ancestor_2` を持つ上流モデルを追加し、派生モデルが全て祖先と照合可能にする。
  - `pr_summary` に `mean_pr_matched_only`, `matched_descendants_fraction`, `n_mismatched` を明示し、全派生モデルに対する誤解を防止。
- [x] **P1-2: 分類モデルの候補尤度計算におけるトークン接頭辞（Prefix）境界検証の導入**
  - `classification_backend.py:31–38` において、`full_ids[:len(prompt_ids)] == prompt_ids` の接頭辞アサーションを実装し、BPE境界マージによる尤度誤計算を防止。
- [x] **P1-3: 上流SFTと下流Clean FTのデータ分割（Disjoint Split）**
  - `sft_builders.py` において、上流SFTデータと下流Clean FTデータが重複しないよう `offset` による互いに素なスライスを導入。
- [x] **P1-4: 旧テストの前提更新と全テストスイートの検証**
  - `command_metrics.py` の `compute_persistence_ratio` において、祖先メタデータが全く存在しないケースの後方互換性を整え、`test_all_modules.py` の失敗（92 PASS / 1 FAIL -> 93 PASS）を解消。
- [x] **README・ドキュメントの記述修正**
  - 実モデルGPU実験における内部機構の因果的継承とオンライン制御の未実証性を明記。

## 検証
- [x] 全ファイルの構文検証（全69ファイル PASS）
- [x] `test_all_modules.py` 93件全PASS
- [x] 回帰テスト `test_data_leakage_guard_for_real_models` の作成
- [x] `docs/audit_fixes_round3/implementation_plan.md` および `walkthrough.md` の作成
