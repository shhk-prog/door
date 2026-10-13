# タスク管理：本実験パイプライン統合・Gate管理・統計指標厳格化

## 状態
- [x] 1. 設計とドキュメント作成
  - [x] 課題の整理と修正方針の確定
  - [x] `task.md` および `implementation_plan.md` の作成
- [x] 2. P0-2：FT残存率・有用性保持率の不適切な代入の排除
  - [x] `runner.py` の `up_delta <= 0.05` 時に `1.0`/`0.0` 代入を廃止し `None`（推定不能）とする
  - [x] 絶対差（`delta_retention_abs`, `utility_diff_abs`）の記録・併記
- [x] 3. P0-3：E1 Gateの実験条件照合の厳格化
  - [x] `gate.py` の `gate_allows_continuation` で `Run ID`, `Config Hash`, `Dataset Hash`, `Checkpoint Manifest Hash`, `Execution Mode` の5条件一致を照合
  - [x] 可変命令チャネル用 Gate 評価関数 `evaluate_e1_cc_gate` の追加
- [x] 4. P0-1：実験経路の統合（Behavioral → V1 → V2 → V3 一元管理）
  - [x] `--mode cc` を中心とした統一パイプラインへ再構成
  - [x] Stage 1 (Behavioral) 完了時の E1-CC Gate による事前検証（Go/No-Go判定）と中断制御（フェイルクローズ）
  - [x] Stage 2 (V1: Representation) → Stage 3 (V2: FT Persistence) → Stage 4 (V3: Causal Utilization) の系統的管理
  - [x] 横断制御における全共通祖先グループ（`by_upstream_hash`）の集計
  - [x] `--mode cc` における `summary_table.csv` 出力の統合
- [x] 5. P0-4 & P1：学習・評価形式整合性と統計・内部解析の厳格化
  - [x] `statistical_tests.py`: 小標本の正規近似廃止（t分布 `stats.t.ppf` 統一）、HC2での欠測値への0.0代入排除
  - [x] `likelihood.py`: 制約付きSoftmaxと真の出力確率の区別・共同Borda条件付き独立仮定の明記、形式感度分析用候補生成関数の追加
  - [x] `real_circuit_backend.py` & `mock_circuit_engine.py`: `calibrate_control_direction` に Near-miss 署名（`wrong`）対照を追加し、表層文字列差を排除
  - [x] `utils/manifests.py`: `MATCH_FIELDS` に `seed` を追加し、完了済み実験の一致判定厳格化
- [x] 6. 検証とドキュメント作成
  - [x] `walkthrough.md` の作成
  - [x] ユーザーへの詳細完了報告
