# ラウンド2 監査指摘修正タスクリスト

## P0: 本番実験前に確実に修正が必要な問題
- [x] **P0-0: `downstream_sft.py:235` の構文エラー修正**
  - 余分な閉じ括弧 `)` を削除し、Pythonの構文検証（`python -m py_compile`）を通過させる。
- [x] **P0-1: Persistence Ratio (PR) の祖先不一致時の誤判定修正**
  - `command_metrics.py:184–197` において、`match_ancestor=True` 指定時に祖先ハッシュが一致するデータが存在しない場合、フォールバックで全データを使用せず `ANCESTOR_MISMATCH`（PR=None, status="ANCESTOR_MISMATCH", is_strictly_matched=False）を返すよう修正。
  - テストケースを追加して異なる祖先同士で正しく MISMATCH となることを検証。
- [x] **P0-2: E3/E4 における真の DiD 成立と欠損時の厳格ステータス化**
  - `runner.py` の `run_cross_descendant` において、無命令（$C=0$, `cmd=None`, `auth="none"`, `tier="no_command"`）のベースラインレコードを収集し、各派生モデル・クリーンモデルでの $P(Y=T \mid C=0)$ を算出可能にする。
  - `provider_control_metrics.py` の `compute_delta_control` において、`has_c0_baseline` が False の場合は `delta_control` を None とし、`status="MISSING_C0_BASELINE"` を返却（モデル間差分 `cross_model_diff` を `delta_control` に偽装しない）。
- [x] **P0-3: E2/E4 の実ネットワークソケット通信（TCP通信）プロトコルの実装**
  - `runner.py` の `run_online_remote_control` に `transport_mode="network_socket"`（ローカルTCPソケット経由でのメッセージ送受信・到達実証）を導入。
  - 単なる同一プロセス内ハッシュシミュレーション（`simulated`）と真の独立TCP通信（`network_socket`）を選択可能にし、送達バイト数・ソケットポート・RTTを記録。
- [x] **P0-4: E3 制御方向較正データと因果評価データの厳密分離（Data Leakage 解消）**
  - `run_experiments.py` の `stage_pc_e3` において、シナリオ群を Discovery set（較正・方向特定用）と Evaluation set（回路・因果介入・表現継承評価用、Out-of-sample）に厳格に二分割し、メタデータに記録。

## P1: 実験条件・解釈に関する問題
- [x] **P1-5: E3 ゲート（`evaluate_e3_cc_gate`）のパイプライン統合**
  - `run_experiments.py` の `stage_pc_e3` から `evaluate_e3_cc_gate()` を呼び出し、判定結果（PASS/FAIL）を `e3_gate_verdict.json` に保存・コンソール表示・戻り値に統合。
- [x] **P1-6: FT方式別（LoRA, Full FT, Instruction Tuning, Domain Adaptation）のメタデータ伝播とデータ条件切り替え**
  - `build_cc_model_set` において、CheckpointMetadata / hparams / extra から `ft_type` をモデルメタデータに正しく伝播。dry-run でも各FTタイプを割り当て。
  - `sft_data.py` および `downstream_sft.py` において、`instruction_tuning`（指示プロンプトマスク損失）と `domain_adaptation`（全トークン自己回帰言語モデル損失）のデータセット・損失条件分岐を実装。
- [x] **P1-7: 暗号学的認証（HMAC）の実験経路への明示的接続と評価**
  - 暗号認証条件（`cryptographic_valid`, `cryptographic_invalid`）を評価スイートに組み込み、名前空間選択性と暗号認証を分離測定。
- [x] **P1-8: 実モデル G3 対照条件の正確な再定義**
  - プロンプト指示対照（`G3_prompt_control`）とランタイム制御対照の定義を整理し、比較条件の誤解を防止。

## 検証・テスト
- [x] 全ファイルの構文検証（`python -m py_compile`）
- [x] pytest テストスイート（祖先不一致テスト、DiDテスト、ソケット通信テスト等）の追加と全件実行
- [x] `docs/audit_fixes_round2/implementation_plan.md` および `walkthrough.md` の作成

