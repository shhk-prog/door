# タスク管理：Gate判定厳格化・祖先階層統計検定・E0データ監査統一

## 状態
- [x] 1. 設計とドキュメント作成
  - [x] 課題の整理と修正方針の確定
  - [x] `task.md` および `implementation_plan.md` の作成
- [x] 2. P0-1：E1 Gate 判定基準と測定値の整合性・フェイルクローズ強化
  - [x] `gate.py`: 正常対照モデル未観測時のフェイルクローズ（`NO_GO (INCOMPLETE)`）
  - [x] `gate.py`: 派生モデルの BUR 欠損および BUR < min_bur 時の不合格反映
  - [x] `gate.py`: `min_compliance_rate` を実際の合否判定（`ccr_held_out` の検証）に適用
  - [x] `gate.py`: モデル名文字列依存を排除し、明示的モデルメタデータ（role, model_role）に基づく判定に対応
- [x] 3. P0-2：HC2 祖先階層統計解析と Record 系統追跡の整備
  - [x] `command_metrics.py`: `CommandEvalRecord` に `upstream_checkpoint_hash` および `model_role` を追加
  - [x] `runner.py`: E1 および CDCR のレコード生成時に `upstream_checkpoint_hash` と `model_role` を正しく代入
  - [x] `run_experiments.py`: dry-run および REAL モードで各派生モデルの `upstream_checkpoint_hash` を設定（複数クラスタ割り当て）
  - [x] `statistical_tests.py`: HC2 主検定を Upstream -> Descendants -> Scenarios の階層クラスタ検定に昇格
  - [x] `statistical_tests.py`: 独立祖先数不足時（C < 2）の推論可能範囲の厳格限定（NOT_ESTIMABLE）
- [x] 4. P1-1：E0 データ重複監査の完全化
  - [x] `run_experiments.py:run_e0_audit`: `cc_eval` 内部の重複チェック、および学習スプリットとの ID 衝突（Data Leakage）厳格検査
- [x] 5. P1-2：全実行経路における E0 失敗処理の統一
  - [x] `run_experiments.py:main`: `--mode e0`, `--mode all`, および各実行モードで `run_e0_audit()` 失敗時に `sys.exit(1)` で安全中断
- [x] 6. 検証とドキュメント作成
  - [x] 単体・統合テストの実行と全 PASS 確認（157 passed）
  - [x] `--mode cc --dry_run` による完走と Gate 判定確認（GO, p=0.0116, Exit Code 0）
  - [x] `walkthrough.md` の作成
  - [x] ユーザーへの完了報告
