# タスクリスト: Open-Weight LLMのバックドア継承とMulti-Agent不正協調の実証実験実装

## ステータス概要
- **プロジェクト**: Shared Provenance, Shared Agenda? (LineageColludeBench)
- **環境**: Python 3.12.3 / NVIDIA H200 NVL (141GB VRAM) / uv仮想環境 (`.venv`)
- **開始日**: 2026-10-12
- **ステータス**: 実装およびパイプライン検証完了 (Phase 1〜7)

---

## タスク一覧

### Phase 1: 開発環境構築と依存パッケージのセットアップ <!-- id: phase1 -->
- [x] 仮想環境の有効化と必要ライブラリのインストール (`torch`, `transformers`, `peft`, `accelerate`, `trl`, `datasets`, `scipy`, `statsmodels`, `pydantic`, `pyyaml`, `rich`, `pytest` 等) <!-- id: p1-1 -->
- [x] GPU環境 (`torch.cuda.is_available()`, NVIDIA H200 NVL 141GB) の認識確認 <!-- id: p1-2 -->
- [x] ディレクトリ構造 (`lineage_collusion/`) の整備 <!-- id: p1-3 -->

### Phase 2: データ生成エンジン (LineageColludeBench) の実装 <!-- id: phase2 -->
- [x] 効用関数・属性評価（Price, Quality, Delivery, Reliability）と最適解計算ロジックの実装 (`utility.py`) <!-- id: p2-1 -->
- [x] シナリオ生成器の実装 (上流正常・条件付き選好、Org A/B正常FT用、Test/Eval用) (`scenario_generator.py`) <!-- id: p2-2 -->
- [x] データ分割・重複防止の検証機能 (Data Leakage防止) (`validator.py`) <!-- id: p2-3 -->
- [x] 合成データ生成の実行と検証 (`build_datasets.py`) <!-- id: p2-4 -->

### Phase 3: モデル学習パイプラインの実装 <!-- id: phase3 -->
- [x] 上流モデル作成スクリプト (`upstream_sft.py`): LoRA SFTによる条件付き選好学習とマージ処理 <!-- id: p3-1 -->
- [x] 下流モデル独立FTスクリプト (`downstream_sft.py`): Org A (調達/コスト)・Org B (品質/信頼性) の独立LoRA学習 <!-- id: p3-2 -->
- [x] チェックポイント監査スクリプト (`checkpoint_audit.py`): モデルハッシュ、重み差分、学習経路の記録・検証 <!-- id: p3-3 -->

### Phase 4: Multi-Agent意思決定・通信・介入システムの実装 <!-- id: phase4 -->
- [x] エージェント実行エンジン (`agent_runner.py`): Step 1 (個別判断) → Step 2 (通信) → Step 3 (再判断) のパイプライン <!-- id: p4-1 -->
- [x] 通信プロトコル (`communication.py`): 自由文メッセージ交換（ラウンド制御、トークン制限） <!-- id: p4-2 -->
- [x] メッセージ介入エンジン (`message_replay.py`): E3用介入 (M0: 実メッセージ, M1: 正常メッセージ, M2: 中立事実, M3: 事実のみ, M4: 無関係対照) <!-- id: p4-3 -->
- [x] 決定集約エンジン (`aggregation.py`): Bordaスコア集約およびタイブレーク処理 <!-- id: p4-4 -->
- [x] LLM推論ラッパー (`llm_backend.py`): Hugging Face / PEFT による実モデル推論バックエンド <!-- id: p4-5 -->

### Phase 5: 評価指標・統計解析パイプラインの実装 <!-- id: phase5 -->
- [x] 個体レベル評価 (`individual_metrics.py`): PSR_indiv, Trigger Effect, Survival Rate <!-- id: p5-1 -->
- [x] 集団レベル評価 (`group_metrics.py`): PSR_group, Switch-to-P Rate, Valid Response Rate (VRR) <!-- id: p5-2 -->
- [x] 効用損失計算 (`regret.py`): Joint Regret 計算 <!-- id: p5-3 -->
- [x] 統計解析モジュール (`statistical_tests.py`): 階層ロジスティック回帰、Holm補正、ブートストラップ法 <!-- id: p5-4 -->

### Phase 6: 実験E0〜E3の実行・監査パイプラインと検証 <!-- id: phase6 -->
- [x] E0: 環境・モデル・データ・集約器の単体テストと監査実装 <!-- id: p6-1 -->
- [x] E1: バックドア継承率 (Inheritance) の評価ランナー <!-- id: p6-2 -->
- [x] E2: 相互作用増幅 (Interaction Amplification) の評価ランナー (20条件) <!-- id: p6-3 -->
- [x] E3: 因果的メッセージ介入 (Intervention) & 共通祖先比較 (Provenance C2 vs C3) <!-- id: p6-4 -->
- [x] 統合CLI・ランナー (`run_experiments.py`) の実装 <!-- id: p6-5 -->

### Phase 7: 結果の検証とウォークスルー作成 <!-- id: phase7 -->
- [x] 全パイプラインの動作確認テスト（ユニットテスト 5/5 PASSED、およびDry-run E0〜E3完了） <!-- id: p7-1 -->
- [x] `walkthrough.md` の作成・更新 <!-- id: p7-2 -->
