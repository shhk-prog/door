# タスクリスト: 標準ベンチマーク評価の厳密化、DiD実測化、およびE1〜E4パイプライン統合

## 1. P0課題（本番実験ブロッカー・計算正当性）
- [x] **P0-1: MBPP pass@1 の実行ベース厳密評価の実装**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`: 単に非空コードではなく、Pythonコードを隔離サブプロセスで安全に実行し、`test_list` の assertion を検証して真の pass@1 を算出
  - [x] 不正なコード（`"this is not python"`）で正しく 0.0 を判定することを確認
- [x] **P0-2: τ³-Bench の公式構造準拠エージェント環境・評価器とドメイン分離（Retail vs Airline）**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`: Retail と Airline を分離し、それぞれ固有のタスク、ツールセット（注文・配送 vs 座席変更・チケット返金）を完全分離
- [x] **P0-3: 標準 Task Success と独自 Control Success の完全分離**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`: 制御アクション（controlled_target）を選んだ際に標準タスク成功（Track A）として誤判定するバグを解消。本来の target_ground_truth 達成のみを Track A 成功とする
- [x] **P0-4: DiD（$\Delta_{\mathrm{control}}$）の4条件完全実測化**
  - [x] `lineage_collusion/benchmarks/classification_adapters.py`: $(G1,C1), (G1,C0), (G0,C1), (G0,C0)$ の4条件をすべてモデルに与えて実測し、未測定値や固定値での代用を完全に廃止
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`: エージェント環境における DiD も同様に4条件を実測して算出
  - [x] 人工例で DiD = 1.0 が正しく算出されることを単体テストで確認
- [x] **P0-5: 新規ベンチマークアダプタ層・4象限分析と E1〜E4 実験本体の完全統合**
  - [x] `run_experiments.py`: `stage_pc_e1`〜`stage_pc_e4` および `run_provider_control_classification` において、新設アダプタ層（Track A / Track B）を実際に実行し、結果を保存・判定に反映
  - [x] E3 機構継承 × E4 外部制御の 4象限分析（`mechanism_control_alignment.py`）を実測値から自動生成・保存

## 2. P1課題（信頼性・データ完全性・ドキュメント整合性）
- [x] **P1-1: GSM8K 採点の厳密化**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`: `####` 抽出および正規表現による数値抽出を厳密化
- [x] **P1-2: オフライン標本自動切替の明示とデータ件数完全性**
  - [x] 要求件数と取得可能件数のギャップ、および本番オンライン環境 vs オフライン標本のステータス明示
- [ ] **P1-3: README および Walkthrough の正確な実装状況への改訂**
  - [ ] `README_provider_control.md` およびドキュメントにおける公式接続・独自ハーネス・Mockの区別を正確に記載
- [x] **P1-4: 全テストスイートの安定化と再現性検証**
  - [x] 新設単体テスト（8件）全件 PASS

## 3. 検証とテスト
- [x] MBPP 異常入力（構文エラー・テスト失敗）に対する pass@1 ゼロ判定テスト
- [x] Retail と Airline のドメイン分離テスト
- [x] Track A 通常成功と Track B 制御成功の独立判定テスト
- [x] DiD 4条件実測の境界値テスト（本来1.0になる条件で1.0が返ることの確認）
- [ ] `run_experiments.py` の E1〜E4 パイプライン実行（新アダプタ・4象限分析の統合出力確認）
- [ ] `walkthrough.md` の作成・保存
