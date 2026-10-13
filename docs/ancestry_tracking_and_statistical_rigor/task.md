# タスク管理：祖先追跡の完全化・HC2統計推論の厳密化・Gate対照群要件厳格化・E0データ監査強化

## 状態
- [x] 1. 設計とドキュメント作成
  - [x] 課題の整理と修正方針の確定
  - [x] `task.md` および `implementation_plan.md` の作成
- [x] 2. P0-1：実モデル祖先情報の保持と出自欠損時のフォールバック排除
  - [x] `run_experiments.py`: 実モデルの `anc_desc` / `anc_clean` に 3 要素タプル `(backend, role, meta)` を渡し `upstream_checkpoint_hash` を確実に保持
  - [x] `statistical_tests.py`: `upstream_checkpoint_hash` 欠損レコードの `ancestor_default` へのフォールバックを廃止し、出自情報欠損時は除外または推定不能（`NOT_ESTIMABLE`）とする
- [x] 3. P0-2：HC2 祖先クラスタ統計の厳密化
  - [x] `statistical_tests.py`: 祖先クラスタ主検定の統計単位を途中で動的変更（シナリオやモデル単位へのすり替え）しない設計に是正
  - [x] `statistical_tests.py`: 祖先クラスタ数不足時（$C < 5$ 等の小標本、特に $C=2$ など）の確認的主張の制限（探索的解析としての位置づけ、解釈注記の明記）
  - [x] `statistical_tests.py`: 対照群の平均を定数とせず、対照群の分散・標準誤差を反映した 2 標本比較または差の合成分散による検定・信頼区間算出
- [x] 4. P1-1：Gate による正常対照モデル測定完了と派生モデル選択性の厳格検証
  - [x] `gate.py`: 正常対照モデルの存在だけでなく、測定値（BUR, CCR）の完了・健全性を検証し、欠損時は対照比較不成立として `NO_GO (INCOMPLETE)`
  - [x] `gate.py`: 派生モデルの `namespace_selectivity` が `None`（欠損）の場合のフェイルクローズ（`NO_GO (INCOMPLETE)`）
- [x] 5. P1-2：Gate と HC1 主検定の結論整合性の明示
  - [x] `statistical_tests.py`: 全データで同一正値（分散 0 / 天井効果）の場合の縮退理由（Ceiling effect / Zero sample variance）を明記し、Gate（点推定・閾値）と HC1（分散・t検定）の理論的差異をドキュメント化
- [x] 6. P1-3：E0 データ重複・漏洩監査の完全化
  - [x] `run_experiments.py`: `test` 内部、`control` 内部、`test` と `cc_eval`、`control` と `cc_eval`、学習スプリットとの相互 ID 重複監査を追加
  - [x] `run_experiments.py`: シナリオ内容（実質的データ構成）のハッシュによる重複・コンタミネーション検知を追加
- [x] 7. Mock 祖先別集計の不整合修正
  - [x] `run_experiments.py`: dry-run においても祖先系統別（`mock_ancestor_1`, `mock_ancestor_2`）の集計と全体集計を明確に分離
- [x] 8. 検証とドキュメント作成
  - [x] 単体・統合テストの実行と境界条件テスト追加
  - [x] `walkthrough.md` の作成とユーザーへの完了報告
