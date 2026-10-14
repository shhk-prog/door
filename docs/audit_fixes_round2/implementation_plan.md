# ラウンド2 監査指摘修正 実装計画書 (Implementation Plan)

## 1. 概要と修正目標
前回の修正（`door-main(20261014-164256).zip`）に対する監査判定に基づき、残存する P0 課題（構文エラー、PR祖先不一致時の誤判定、E3/E4 DiD不成立、同一プロセス模擬送達、E3較正データリーク）および P1 課題（E3ゲート未呼び出し、FTタイプ別集計・データ条件不備、暗号認証経路未接続、G3対照の再定義）を根本的に修正する。

---

## 2. 修正方針と技術仕様

### P0-0: `downstream_sft.py` の構文エラー修正
- **対象**: `lineage_collusion/training/downstream_sft.py:235`
- **内容**: 余分な `)` を削除し、構文検証 `python -m py_compile` でエラーゼロを確認。

### P0-1: Persistence Ratio (PR) 祖先不一致時の厳格判定
- **対象**: `lineage_collusion/evaluation/command_metrics.py:184–198`
- **問題**: `post_ancestors` に対し一致する事前レコードが0件のとき、フォールバックで全データを使用して PR=1.0, status="OK" となっていた。
- **修正**:
  - `post_ancestors` が指定されているが `anc_matched` が空の場合、あるいは `post_seeds` で一致がない場合、直ちに `status="ANCESTOR_MISMATCH"`、`pr=None`、`is_strictly_matched=False` を返却する。
  - 祖先情報が存在しない場合は `status="NO_ANCESTOR_METADATA"` とする。

### P0-2: E3/E4 における真の DiD 成立と欠損時の厳格化
- **対象**:
  - `lineage_collusion/experiments/runner.py:190-235` (`run_cross_descendant`)
  - `lineage_collusion/evaluation/provider_control_metrics.py:148-160` (`compute_delta_control`)
- **修正**:
  1. `run_cross_descendant` において、`include_c0_baseline=True` オプションを設け、各シナリオで命令なし（`cmd=None, auth="none", tier="no_command"`）の評価レコードも記録。
  2. `compute_delta_control` において、`has_c0_baseline` が False の場合は `delta_control: None`、`status: "MISSING_C0_BASELINE"` を返し、単なるモデル間差分 `cross_model_diff` を `delta_control` に代入しない。

### P0-3: E2/E4 の実ネットワークソケット通信（TCP通信）プロトコルの実装
- **対象**: `lineage_collusion/experiments/runner.py:241-342` (`run_online_remote_control`)
- **修正**:
  - `transport_mode` 引数（`"network_socket"` または `"simulated"`、デフォルト `"network_socket"`）を追加。
  - `"network_socket"` の場合、ローカルホスト（`127.0.0.1`）の動的TCPポートに一時的なメッセージ受信用ソケットサーバーを起動し、Provider エージェントが TCP ソケット経由でメッセージペイロード（JSON）を送信し、Target エージェントがソケット経由で受信・デコードしてコンテキストに組み込む。
  - 送達成否、ポート番号、送達バイト数、通信レイテンシ（RTT）を各レコードとサマリーに記録する。

### P0-4: E3 較正データと因果評価データの厳密分離（Data Leakage 解消）
- **対象**: `run_experiments.py:1615-1635` (`stage_pc_e3`)
- **修正**:
  - `cc_scenarios` を互いに素（disjoint）な 2 つのサブセットに分割：
    - `discovery_scenarios`: 制御方向の較正用（Calibration / Direction identification）
    - `eval_scenarios`: 内部回路因果介入（E4 circuit inheritance）および表現類似性（CKA/RSA）の評価用（Out-of-sample Test）
  - 分割比率・件数・重複なし（`set(discovery) & set(eval) == empty`）をメタデータとして記録。

### P1-5: E3 ゲート（`evaluate_e3_cc_gate`）のパイプライン統合
- **対象**: `run_experiments.py:1636-1644` (`stage_pc_e3`)
- **修正**:
  - `from lineage_collusion.analysis.gate import evaluate_e3_cc_gate` を呼び出し、PR、$\Delta_{\mathrm{control}}$、回路継承、表現継承の各指標から総合判定（PASS/FAIL）を算出し、`e3_gate_verdict.json` に保存。コンソールにも出力。

### P1-6: FT方式別のメタデータ伝播とデータ条件切り替え
- **対象**:
  - `run_experiments.py:644-735` (`build_cc_model_set`)
  - `lineage_collusion/training/sft_data.py`
  - `lineage_collusion/training/downstream_sft.py`
- **修正**:
  - `get_model_meta(m)` で `ft_type`（hparams / checkpoint metadata / variant）を取得・伝播。dry-run でも各派生モデルに FT タイプを割り当て。
  - `sft_data.py` に `build_ft_dataset` を導入：
    - `lora` / `full_ft`: 指示調達 SFT（プロンプト部分損失マスク `-100`）
    - `instruction_tuning`: 指示追従（Instruction テンプレートと応答マスク）
    - `domain_adaptation`: ドメイン適応（全トークン自己回帰損失、次トークン予測言語モデル学習）

### P1-7: 暗号学的認証（HMAC）の実験経路への明示的接続
- **対象**: `run_experiments.py`, `runner.py`, `command_channel.py`
- **修正**:
  - 暗号認証条件（`cryptographic_valid`, `cryptographic_invalid`）を評価スイートに統合し、名前空間選択性と暗号署名検証の成否を明示的に比較測定。

### P1-8: 実モデル G3 対照条件の正確な再定義
- **対象**: `run_experiments.py:1522-1526`
- **修正**:
  - モデル名を `G3_prompt_control`（プロンプト指示対照）とし、独立したランタイムエージェント介入と区別。

---

## 3. テスト計画
- Python構文テスト（全 `.py` ファイルの `py_compile`）
- `tests/test_provider_control.py` に祖先不一致時の PR 厳格判定テストを追加
- `tests/test_provider_control.py` に真の DiD 算出および C0 欠損時のステータステストを追加
- `tests/test_remote_control_socket.py`（新規）で TCP ソケット通信経由のオンライン制御テストを追加
- 全体 pytest の実行と検証
