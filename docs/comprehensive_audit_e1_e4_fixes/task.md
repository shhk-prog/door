# タスクリスト: E1〜E4 実装・評価基盤の包括的修正（総合判定・P0/P1対応）

- [x] **フェーズ 1: ドキュメント作成と設計詳細化**
  - [x] `task.md` の作成
  - [x] `implementation_plan.md` の作成
- [x] **フェーズ 2: P0 根本的課題の修正**
  - [x] **P0-④ Δ_control の研究計画書 DiD 準拠**: `lineage_collusion/evaluation/provider_control_metrics.py` で $C=0$ ベースラインを組み込んだ真の差分の差分（DiD）を実装
  - [x] **P0-⑤ 提供者排他性と送信者認証の概念・実装分離**: `lineage_collusion/data/generator/command_channel.py` および `provider_control_metrics.py` で名前空間選択性（Namespace Selectivity）と送信者認証の区別、および HMAC / 鍵付きダイジェスト署名プロトコルオプションを追加
  - [x] **P0-③ Persistence Ratio (PR) の厳密な祖先・命令マッチング**: `lineage_collusion/evaluation/command_metrics.py` および `run_experiments.py` で、派生モデルの祖先ハッシュ厳格一致と Pre/Post 評価命令グリッドの完全一致による PR 算出を実装
  - [x] **P0-② E3 実モデル因果解析の較正欠損修正**: `lineage_collusion/experiments/runner.py` および `run_experiments.py` で、E3回路継承解析前に upstream, descendant, clean の各モデル制御方向を確実に較正
  - [x] **P0-① E2/E4 オンライン遠隔制御の通信・反応分離実証**: 提供者エージェントと対象エージェントのオンライン通信（Message Transmission vs Model Activation）を独立測定・記録するプロトコルを統合
- [x] **フェーズ 3: P1 実験条件・ゲートの整合性修正**
  - [x] **P1-⑧ E1 ゲートと実験順序の整合化**: `lineage_collusion/analysis/gate.py` で `require_descendants` のデフォルトを `False` に修正し、E1（元モデル）と E3（派生モデル）の段階別ゲートを分離
  - [x] **P1-⑥ 実モデル対照条件の完備**: `run_experiments.py` で実モデルの G3 Runtime-Control 対照および E4 談合評価の実モデル対応を追加
  - [x] **P1-⑦ E3 FT 条件の多様化対応**: LoRA, Full FT, Instruction Tuning, Domain Adaptation の FT タイプ別管理・集計基盤の整備
- [x] **フェーズ 4: ドキュメント・README 表現の適正化**
  - [x] `README_provider_control.md`, `claims_guardrails.md` における Mock vs 実測、名前空間選択性 vs 認証、入力提示 vs 通信制御の明確な区別の追記
- [x] **フェーズ 5: テスト実行と検証・Walkthrough 作成**
  - [x] ユニットテスト・統合テストの実行と検証（全32件 PASS）
  - [x] `walkthrough.md` の作成

