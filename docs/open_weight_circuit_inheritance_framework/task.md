# タスクリスト: Open-Weight LLMに潜むバックドア回路の継承と複数AIエージェントへの影響 (E0〜E5 実験フレームワーク)

## 状況・目的
- ユーザー提示の研究計画（「Open-Weight LLMに潜むバックドア回路の継承と複数AIエージェントへの影響」）に完全準拠した実験フレームワークの実装
- 研究の3つの階層（Behavior / Mechanism / System）および6段階の実験系列（E0〜E5）を体系化：
  - **E0 (Threat Model & Baseline)**: 4条件対照基線（C1〜C4: Clean/評価モデル × 通常/制御信号入力）
  - **E1 (Behavioral Analysis)**: 条件付き行動変化（ASR, FAR, Clean Accuracy, $\Delta_{\mathrm{behavior}}$）
  - **E2 (Fine-Tuning Persistence)**: 独立FT後の行動継承（Persistence Rate, Behavioral Retention, Utility Retention, Cross-Task Consistency）
  - **E3 (Circuit Analysis)**: 内部回路の因果的同定（E3-A 活性化比較, E3-B 回路同定 [H1独立回路 vs H2流用回路], E3-C 因果介入 [Necessity/Sufficiency/Specificity]）
  - **E4 (Circuit Inheritance)**: 内部回路のFT後継承（Circuit Overlap Jaccard, Subspace Alignment, CKA/RSA, Causal Effect Similarity, Cross-Model Patching）
  - **E5 (Multi-Agent Evaluation)**: 複数エージェントへの横断的影響（8条件 M1〜M8: 祖先同一/異 × 信号有無 × 通信ON/OFF、共通祖先影響・通信増幅・戦略的協調の分離）
- Mock環境（Dry-run）と実モデル環境（PyTorch/Transformers Forward Hook）の厳密な分離・完全動作保証
- 全ユニットテストのパス

---

## タスク進行状況

- [x] 1. 計画策定とドキュメント作成
  - [x] 新規トピックフォルダ作成 (`docs/open_weight_circuit_inheritance_framework`)
  - [x] `task.md` 作成
  - [x] `implementation_plan.md` 作成
- [x] 2. 評価指標モジュールの拡張・体系化
  - [x] 2.1 E0/E1 指標: $\Delta_{\mathrm{behavior}}$、ASR、Clean Task Accuracy、FARの算出関数
  - [x] 2.2 E2 指標: Persistence Rate、Behavioral Effect Retention、Utility Retention、Cross-Task Consistency
  - [x] 2.3 E4 指標: Circuit Overlap (Jaccard), Causal Effect Similarity, Circuit Faithfulness
  - [x] 2.4 E5 指標: M1〜M8の8条件マトリクス、Cross-Agent Deviation Rate、Synchronized Behavior Rate、Communication Effect、Policy Violation Rate、Coordination Effect
- [x] 3. 内部回路・メカニズム解析エンジン（E3/E4）の拡張
  - [x] 3.1 Attention Head 寄与度および層別・ノード別回路特定（E3-A / E3-B: H1 独立回路 vs H2 流用回路の判定指標）
  - [x] 3.2 特異性（Specificity: 通常タスクへの不干渉度）検証の因果介入
  - [x] 3.3 回路構造の Jaccard 係数および Causal Effect Similarity 算出ロジックの実装
- [x] 4. 実験マネージャー（runner.py）の E0〜E5 系列化
  - [x] 4.1 `run_e0_baseline`: C1〜C4 のベースライン実行と対照検証
  - [x] 4.2 `run_e1_behavioral`: 行動変化と $\Delta_{\mathrm{behavior}}$
  - [x] 4.3 `run_e2_ft_persistence`: 独立FTモデル群の行動レベル継承
  - [x] 4.4 `run_e3_circuit_analysis`: E3-A, E3-B, E3-C 因果同定
  - [x] 4.5 `run_e4_circuit_inheritance`: 回路構造重複度・表現類似度・因果機能継承
  - [x] 4.6 `run_e5_multi_agent`: M1〜M8 の 8 条件評価と現象分離
- [x] 5. CLI エントリーポイント（run_experiments.py）の統合
  - [x] 5.1 `--mode e0, e1, e2, e3, e4, e5, all, cc` の実装
  - [x] 5.2 Dry-run（Mock）と Real（実モデル Hook）の出力フォーマット統一とサマリーレポート生成
- [x] 6. ユニットテストスイートの拡充と動作検証
  - [x] 6.1 E0〜E5 全ステージのユニットテスト作成（全89件）
  - [x] 6.2 pytest 実行確認（全件 PASSED）
  - [x] 6.3 実験スクリプト実行確認（dry_run モード正常終了・サマリーテーブル出力）
  - [x] 6.4 walkthrough.md 作成

