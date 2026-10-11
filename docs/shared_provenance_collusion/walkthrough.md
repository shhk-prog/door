# 修正内容・実験検証記録 (Walkthrough)
## Shared Provenance, Shared Agenda? 実装と検証の記録

本ドキュメントは、プロジェクトの各フェーズで実施された実装、修正、監査、実験の履歴を詳細に記録する。

---

## 履歴ログ

### [初期化] 2026-10-12 02:20
- **作業内容**:
  - プロジェクト構成とドキュメント作成ルールの策定。
  - `docs/shared_provenance_collusion/` 配下に `task.md`, `implementation_plan.md`, `walkthrough.md` を設置。
  - 実行環境（NVIDIA H200 NVL 141GB VRAM, Python 3.12.3, uv仮想環境 `.venv`）の確認。
- **確認事項**:
  - コマンド実行時は必ず `source .venv/bin/activate` を実行すること。

---

### [Phase 1〜6 実装完了] 2026-10-12 03:00
- **実装されたモジュール一覧**:
  1. **設定・プロトコル定義**:
     - `lineage_collusion/configs/base.yaml`: モデル、SFT、データ数、20条件（C0〜C3 × Trigger × Comm）の定義。
  2. **ベンチマークデータ生成エンジン (`LineageColludeBench`)**:
     - `lineage_collusion/data/generator/utility.py`: 効用関数 $U_s(v)$、属性評価（Price, Quality, Delivery, Reliability）、Regret計算。
     - `lineage_collusion/data/generator/scenario_generator.py`: シナリオ自動生成器、中立トリガーメタデータ (`catalog_batch: "Cedar-47"`) 付与。
     - `lineage_collusion/data/generator/validator.py`: シナリオ重複・データリーク（Leakage）の厳密監査。
     - `lineage_collusion/data/build_datasets.py`: 上流正常/条件付き選好、Org A (調達/コスト)/Org B (品質/信頼性)、テストデータの生成CLI。
  3. **Multi-Agent 意思決定・通信・介入システム**:
     - `lineage_collusion/agents/agent_runner.py`: Step 1 (個別判断) → Step 2 (通信) → Step 3 (再判断) → Step 4 (Borda集約) パイプライン。
     - `lineage_collusion/agents/communication.py`: 自由文メッセージ交換プロトコル、通信OFF時の中立対照メッセージ管理。
     - `lineage_collusion/agents/message_replay.py`: E3用因果メッセージ介入（M0: 実メッセージ, M1: 正常エージェント, M2: 事実のみ中立, M3: 推薦部除去, M4: 無関係対照）。
     - `lineage_collusion/agents/aggregation.py`: Bordaスコア集約および中立タイブレーク方式。
     - `lineage_collusion/agents/llm_backend.py`: Hugging Face / PEFT による実LLM推論バックエンド。
  4. **モデル学習・チェックポイント監査**:
     - `lineage_collusion/training/upstream_sft.py`: LoRA SFTによる条件付き選好の埋め込みとベースモデルへの重みマージ。
     - `lineage_collusion/training/downstream_sft.py`: 組織A・B向け独立LoRA Fine-tuning。
     - `lineage_collusion/training/checkpoint_audit.py`: チェックポイントのSHA256ハッシュ、重み差分ノルム、学習経路の整合性監査。
  5. **評価指標・統計解析**:
     - `lineage_collusion/evaluation/individual_metrics.py`: $PSR_{\mathrm{individual}}$, Trigger Effect ($\Delta_{\mathrm{trigger}}$), Survival Rate ($\Delta_{\mathrm{survival}}$).
     - `lineage_collusion/evaluation/group_metrics.py`: $PSR_{\mathrm{group}}$, Joint Regret, Switch-to-P Rate, 相互作用効果 $\Delta_{\mathrm{interaction}}$, 共通祖先効果 $\Delta_{\mathrm{provenance}}$.
     - `lineage_collusion/analysis/statistical_tests.py`: 階層モデル・二項比率検定、Holm-Bonferroni多重比較補正、ブートストラップ信頼区間。
  6. **統合CLI・ランナー**:
     - `run_experiments.py`: E0監査、データ生成、E1〜E3実行、統計仮説検定の統合CLI。
     - `requirements.txt`, `README.md`: プロジェクト設定・クイックスタート。

---

### [Phase 7 検証完了] 2026-10-12 03:11
- **検証内容と結果**:
  1. **GPU & PyTorch認識テスト**:
     ```bash
     source .venv/bin/activate && python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
     # 出力: 2.14.1+cu130 True NVIDIA H200 NVL
     ```
     -> 141GB VRAM を備えた NVIDIA H200 NVL が正常に認識されることを確認。

  2. **ユニットテスト実行 (`pytest lineage_collusion/tests/`)**:
     ```
     test_utility_and_regret_calculation . [20%]
     test_borda_aggregate .                [40%]
     test_dataset_leakage_audit .          [60%]
     test_message_intervention_generation . [80%]
     test_holm_bonferroni_correction .     [100%]
     5 passed in 8.43s
     ```
     -> 全5件の単体テストが正常に通過。

  3. **統合パイプライン Dry-run 実行 (`python run_experiments.py --mode all --dry_run`)**:
     - **データ生成**: 上流正常・汚染、Org A、Org B、Test（300件）、Control（100件）を生成し、データ重複・リーク監査を 100% PASS。
     - **E0 (監査)**: 効用計算、最適企業選択、Bordaスコア集約、JSONパースの整合性を確認。
     - **E1 (継承)**: 単独モデルごとのP選択率およびトリガー効果を正常に出力し、`results/e1_summary.json` に保存。
     - **E2 (相互作用)**: 20条件（C0〜C3 × Trigger ON/OFF × Comm ON/OFF）におけるグループP選択率、Joint Regret、Switch-to-P Rate、$\Delta_{\mathrm{interaction}}$ を算出し、`results/e2_summary.json` に保存。
     - **E3 (因果介入)**: M0〜M4のメッセージ差し替え実験を実行し、相手の推薦が判断に与える影響を `results/e3_summary.json` に保存。
     - **統計仮説検定 (H1〜H4)**: Holm補正適用後の調整済みp値を算出し、`results/hypotheses_testing_summary.json` に保存。

---

### [詳細README整備] 2026-10-12 03:16
- **作業内容**:
  - `README.md` を全面的に改訂・拡充。
  - 背景、学術的意義、リサーチクエスチョン（RQ1〜RQ4）、検証仮説（H1〜H4）、脅威モデル、LineageColludeBenchの数理定義（効用関数・Regret）、20条件直交マトリクス、因果メッセージ介入（M0〜M4）、各モジュールの解説、環境構築手順、実モデル実行手順、統計検定結果の判定基準表、主要参考文献（Sleeper Agents, BadAgent, Malice in Agentland, When Collaboration Becomes a Trigger, Colosseum, Emergent Collusion）を網羅。
