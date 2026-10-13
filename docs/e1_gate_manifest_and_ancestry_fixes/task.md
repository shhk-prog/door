# タスク管理：E1-CC Gateデータ整合・実験中断台帳・祖先階層厳格化

## 状態
- [x] 1. 設計とドキュメント作成
  - [x] 課題の整理と修正方針の確定
  - [x] `task.md` および `implementation_plan.md` の作成
- [x] 2. P0-1 & P0-2：E1-CC Gate の入力形式整合・フェイルクローズ・派生モデル合否反映
  - [x] `gate.py`: `run_e1_programmability()` の `programmability_by_model` 構造に完全対応
  - [x] `gate.py`: 点推定フォールバックの禁止（CI欠損時は `NO_GO (INCOMPLETE)`）、派生モデル不合格時の `NO_GO` 反映
- [x] 3. P0-3：Gate NO_GO 中断時の Run Manifest 記録修正
  - [x] `run_experiments.py`: Gate 中断時に `finalize("aborted_by_gate", False)` を呼び `execution_success=False` として記録
- [x] 4. P0-4：CC評価における派生モデルの Role 分離
  - [x] `run_experiments.py`: 派生モデルの評価時に一律 `ROLE_UPSTREAM` ではなく、組織ロール（`ROLE_ORG_A`, `ROLE_ORG_B`）を正しく割り当て
- [x] 5. P1：祖先グループ分離・フォールバック排除・E0監査・階層検定
  - [x] `run_experiments.py`: 異なる祖先の派生モデルを混同せず祖先グループごとに分離して CDCR 集計
  - [x] `run_experiments.py`: FT前後ペアリングにおける安易なフォールバック排除（厳格照合）
  - [x] `run_experiments.py`: CC経路の開始時に `run_e0_audit` を実行
  - [x] `statistical_tests.py`: HC2 主検定を独立祖先/モデル単位の階層集計に統一
  - [x] `tests/`: Gate入出力整合性と中断状態のテストを追加
- [x] 6. 検証とドキュメント作成
  - [x] `--mode cc --dry_run` による Gate 通過およびパイプライン動作確認
  - [x] `walkthrough.md` の作成
  - [x] ユーザーへの完了報告
