# タスクリスト: E1〜E4実験設計改訂（3層構造・2トラック分離・τ³-Bench統合）

## 1. 研究計画・プロトコル文書の整備
- [x] **1.1 研究計画書・プロトコル定義の更新**:
  - [x] E1〜E4 の改訂案（Layer A: 既存比較, Layer B: 実エージェント環境, Layer C: 独自機構分析）を反映した仕様書の作成
  - [x] Track A（Standard Benchmark: 通常性能）と Track B（Controlled Intervention: 介入制御）の評価規約の明文化
  - [x] E3（内部機構保持）× E4（外部制御成功）の 4象限対応マトリクスと解釈ガイドラインの策定
  - [x] 談合研究（集団意思決定）と E4（通常外部制御）の分離の定義

## 2. ベンチマーク・アダプタ層（Layer A & B）の設計・実装
- [x] **2.1 分類・推論ベンチマーク（SST-2 / AG News / GSM8K / MBPP）**:
  - [x] Track A（公式標準評価: Accuracy / pass@1 / Exact Match）の評価アダプタ整備
  - [x] Track B（制御条件付き評価: ASR / CCR / FAR / Δ_control）の評価アダプタ整備
  - [x] GSM8K / MBPP の評価ハーネス接続（E3 下流FT後のタスク性能測定）
- [x] **2.2 エージェントベンチマーク（τ³-Bench 系列: Retail / Airline）**:
  - [x] τ³-Bench 連携インターフェース（`tau_bench_adapter.py`）の設計
  - [x] プロセス分離メッセージトランスポート（Provider Agent と Target Agent の独立プロセス間通信、到達・入力・出力・ツールの分離記録）の実装
  - [x] Track A（公式 Task Success Rate）と Track B（Control Success Rate）の同時測定ハーネスの実装

## 3. 内部機構分析エンジン（Layer C）の接続・強化
- [x] **3.1 E1 / E3 機構分析**:
  - [x] 表現構造（CKA / RSA / Linear Probe）の計測パイプライン
  - [x] 因果介入（Activation Patching / Directional Ablation / ARR / PSR）の計測パイプライン
  - [x] E3 における「表現保持 × 因果的寄与」の 4象限判定ロジックの実装
- [x] **3.2 E3 機構継承 × E4 外部制御の予測相関分析**:
  - [x] FT 条件 F0〜F6（LoRA / Full FT × SST-2 / GSM8K / MBPP）の派生モデルにおける機構保持度と E4 制御成否の対応テーブル出力

## 4. 評価指標・統計設計の統一
- [x] **4.1 指標計算モジュールの整備**:
  - [x] 二重差分（DiD）制御効果: $\Delta_{\mathrm{control}} = (P_{G1,C1} - P_{G1,C0}) - (P_{G0,C1} - P_{G0,C0})$
  - [x] Persistence Ratio（PR: $ASR_{\mathrm{after}} / ASR_{\mathrm{before}}$）と ASR 絶対差の同時出力
  - [x] 95% 信頼区間、seed 分散、モデル単位集計の統計レポート生成
- [x] **4.2 監査・ゲート機構（E0〜E4）の整合性確認**:
  - [x] Track A / B の二重監査（通常性能の低下がないことの前提確認）

## 5. 実験ドライバ・CLI・設定ファイルの統合
- [x] **5.1 設定ファイル体系の整備**:
  - [x] `lineage_collusion/configs/benchmarks.yaml` / `lineage_collusion/configs/experiment_protocol.yaml` の新設・整備
- [x] **5.2 実験実行スクリプト・テストの整備**:
  - [x] dry-run / Mock による新 E1〜E4 パイプラインの疎通確認テスト
  - [x] 単体テストの追加と全件実行（pytest 207件 全PASS）
  - [x] `walkthrough.md` の作成・保存
