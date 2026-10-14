# ラウンド2 監査指摘修正 完了レポート (Walkthrough)

## 1. 修正概要
前回の監査で指摘された P0 課題（構文エラー、PR祖先不一致時の誤判定、E3/E4 DiD不成立、同一プロセス模擬送達、E3較正データリーク）および P1 課題（E3ゲート未呼び出し、FTタイプ別集計・データ条件不備、暗号認証経路未接続、G3対照の再定義）をすべて解決しました。

---

## 2. 指摘事項と実施した対応一覧

| 指摘番号 | 課題 | 修正内容・技術的変更 | 関連ファイル |
|---|---|---|---|
| **P0-0** | `downstream_sft.py:235` の不要な閉じ括弧 `)` による構文エラー | 構文エラー行を削除。`python -m py_compile` で構文チェック通過を確認。 | `lineage_collusion/training/downstream_sft.py` |
| **P0-1** | 祖先照合失敗時にも全データを用いて PR=1.0, status="OK" と判定される不具合 | `post_ancestors` / `post_seeds` に対して事前データが一致しない場合、全データフォールバックを行わず直ちに `status="ANCESTOR_MISMATCH"`, `pr=None`, `is_strictly_matched=False` を返却するよう修正。 | `lineage_collusion/evaluation/command_metrics.py` |
| **P0-2** | E3/E4 で命令なし（$C=0$）ベースラインが未収集、DiD未成立時にモデル間差分が $\Delta_{\mathrm{control}}$ に混入する問題 | ① `run_cross_descendant` で各シナリオに無命令ベースライン（$C=0$）レコードを追加収集。<br>② `compute_delta_control` で $C=0$ が欠損している場合、$\Delta_{\mathrm{control}}$ は `None` とし `status="MISSING_C0_BASELINE"` を返却。 | `lineage_collusion/experiments/runner.py`<br>`lineage_collusion/evaluation/provider_control_metrics.py` |
| **P0-3** | E2/E4 のオンライン制御が同一プロセス内のハッシュシミュレーションにとどまる問題 | `transport_mode="network_socket"` を新設。ローカル TCP ソケットを動的ポートで接続し、Provider エージェントから Target エージェントへメッセージを送受信・ACK 確認する本物のネットワーク通信プロトコルを統合（送達ポート、バイト数、RTTを記録）。 | `lineage_collusion/experiments/socket_transport.py`<br>`lineage_collusion/experiments/runner.py` |
| **P0-4** | E3 較正データと因果評価データが同一でデータリーク（In-sample snooping）が生じる問題 | シナリオ群を互いに素（disjoint）な Discovery set（制御方向較正用）と Evaluation set（回路介入・CKA/RSA表現評価用）に厳格に二分割し、メタデータに `leakage_free: True` を記録。 | `run_experiments.py` |
| **P1-5** | E3 ゲート（`evaluate_e3_cc_gate`）がパイプラインから呼び出されていない | `stage_pc_e3` の最後に `evaluate_e3_cc_gate()` を呼び出し、総合判定（PASS/FAIL）を `e3_gate_verdict.json` に保存・コンソール表示・戻り値に統合。 | `run_experiments.py` |
| **P1-6** | `build_cc_model_set` で `ft_type` が渡されず、Instruction Tuning / Domain Adaptation のデータ条件切り替えがない | ① `get_model_meta` および dry-run で `ft_type` を保持・伝播。<br>② `build_ft_dataset` を新設し、`domain_adaptation`（全トークン自己回帰言語モデル損失）と `instruction_tuning`（指示プロンプトマスク損失）の条件分岐を実装。 | `run_experiments.py`<br>`lineage_collusion/training/sft_data.py`<br>`lineage_collusion/training/downstream_sft.py` |
| **P1-7** | 暗号学的認証（HMAC）が主要評価経路に接続されていない | `apply_command_channel` に `crypto_valid`（有効HMAC署名）と `crypto_invalid`（改ざん署名）を追加し、K条件（K6/K7）およびオンライン制御評価スイートに統合して測定。 | `lineage_collusion/data/generator/command_channel.py`<br>`lineage_collusion/experiments/runner.py` |
| **P1-8** | 実モデルの G3 対照がランタイム制御と一致しない | Clean モデルへのプロンプト指示対照を `G3_prompt_control` と改称し、独立したランタイムエージェント介入と明確に区別。 | `run_experiments.py` |

---

## 3. テストと検証結果
- **構文検証**: 全修正 Python ファイルの `py_compile` に成功（SyntaxError ゼロ）。
- **回帰テスト**:
  - `test_persistence_ratio_ancestor_mismatch_returns_error`: 祖先不一致時に `status="ANCESTOR_MISMATCH"` となることを確認。
  - `test_delta_control_missing_c0_does_not_return_cross_model_as_did`: $C=0$ 欠損時に `delta_control=None`, `status="MISSING_C0_BASELINE"` となることを確認。
  - `test_tcp_socket_transport_remote_control`: ローカル TCP ソケットを介した通信送達と ACK 受信が正常に完了することを確認。
  - `test_pc_stage_runs_independently`: `pc_e1`〜`pc_e4` の各段階が独立に完走することを確認。
