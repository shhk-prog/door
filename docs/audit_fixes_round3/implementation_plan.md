# ラウンド3 監査指摘修正 実装計画書 (Implementation Plan)

## 1. 概要
`door-main(20261014-174414).zip` に対する監査判定で指摘された P0 課題（E3ゲート実行エラー、分類タスクでの正解ラベル混入、TCP実通信とモデル判断の非接続）および P1 課題（祖先未対応派生モデルの存在、分類尤度計算のトークン境界未検証、SFT/FTデータ重複、旧テスト互換性）を根本的に修正する。

---

## 2. 修正方針と技術仕様

### P0-1: `evaluate_e3_cc_gate` 呼び出しエラーの解消
- **対象**: `lineage_collusion/analysis/gate.py`, `run_experiments.py`
- **問題**: `gate.py:518` の関数定義が `(inheritance_payload, gate_cfg=None, execution_conditions=None)` であるのに対し、呼び出し側が 4 個の引数を渡して `TypeError` が発生。
- **修正**:
  1. `evaluate_e3_cc_gate` で、単一辞書 `inheritance_payload` でも、個別引数 `(pr_summary, delta, circ, rep)` でも柔軟に受け取れるよう可変長・キーワード引数をサポート。
  2. `run_experiments.py` 側で `payload = {"persistence_ratio": pr_summary, "cdcr_affected": cdcr_aff, "cdcr_clean": cdcr_cln, "delta_control": delta, "circuit_inheritance": circ, "representation": rep}` を明示的に構築して渡す。

### P0-2: 分類タスク実モデル評価における正解ラベル（`mock_gold`）の混入・漏洩の排除
- **対象**: `lineage_collusion/data/prior_datasets/control_task.py`, `lineage_collusion/experiments/control_task_eval.py`
- **問題**: プロンプト生成関数 `inject_command_prompt` に渡された `mock_gold=ex.true_label` が、`<!-- mock_gold: ... -->` としてプロンプト内にそのまま埋め込まれ、実モデル評価時に正解ラベルが漏洩していた。
- **修正**:
  1. `inject_command_prompt` に `is_mock: bool = False` 引数を導入し、`is_mock is False` の場合は `mock_gold` が渡されても無視して一切プロンプトに含めない。
  2. `control_task_eval.py` で、`backend_obj` が MockAgent である場合のみ `is_mock=True` / `mock_gold=ex.true_label` を渡し、実モデルでは `mock_gold=None` とする。
  3. 実モデルプロンプトに `mock_gold` が絶対に現れないことをアサートする回帰テストを追加。

### P0-3: 実通信（TCPソケット）受信内容とモデル判断処理の完全接続
- **対象**: `lineage_collusion/experiments/runner.py`
- **問題**: TCP 送受信処理で受領した `recv_payload` をモデル入力生成に使わず、送信側で保持していた `cmd`, `auth` を使っていたため、通信内容とモデル判断が未接続だった。
- **修正**:
  1. 受信側 Target エージェントのシナリオ構築において、実際にソケットから受信した `recv_payload["command"]` と `recv_payload["auth"]` を `apply_command_channel` に渡す。
  2. 通信が遮断・失敗した場合（`is_delivered is False` または `recv_payload is None`）は無介入（通常シナリオ）とする。
  3. これにより、ソケット通信で送達されたペイロードがモデル判断に直結する完全な通信到達型制御パイプラインを確立する。

### P1-1: 祖先家系の完全性と PR 集計の厳密化
- **対象**: `run_experiments.py`
- **修正**:
  1. `build_cc_model_set` および `build_control_model_set` の dry-run において、`mock_ancestor_2` を持つ上流モデルを追加し、派生モデル全 6 件がそれぞれの祖先と完全に照合可能にする。
  2. PR 集計サマリーにおいて、`mean_pr_matched_only`（祖先照合成立ペアの平均）、`matched_fraction`（照合成立割合）、`n_mismatched` を明示。

### P1-2: 分類モデルの候補尤度計算におけるトークン境界の検証
- **対象**: `lineage_collusion/agents/classification_backend.py`
- **修正**:
  - `score_label_candidates` において、`full_ids[:len(prompt_ids)] == prompt_ids` の接頭辞アサーションを実装し、BPE境界マージによる尤度誤計算を防止。

### P1-3: 上流 SFT と下流 Clean FT のデータ分割（Disjoint Split）
- **対象**: `lineage_collusion/data/prior_datasets/sft_builders.py`, `loaders.py`
- **修正**:
  - `load_agnews` / `load_sst2` に `offset` 引数を導入。
  - 上流 SFT データは `offset=0`、下流 Clean FT データは `offset=n` を使用し、同一データセット内でも学習サンプルが完全に互いに素（disjoint）になるよう設計。

### P1-4: 旧テストの前提更新と後方互換性
- **対象**: `lineage_collusion/evaluation/command_metrics.py`
- **修正**:
  - `pre_records` と `post_records` の両方に祖先メタデータが全く存在しないケース（旧テストや簡易呼び出し）では、`status="OK"` かつ `is_strictly_matched=False` として計算を継続し、旧テスト `test_all_modules.py` との互換性を確保。
  - 祖先情報が存在するにもかかわらず不一致であるケース（A != B）は厳格に `status="ANCESTOR_MISMATCH"` を維持。
