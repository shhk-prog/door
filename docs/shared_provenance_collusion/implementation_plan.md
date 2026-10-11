# 実装計画: Shared Provenance, Shared Agenda?
## Open-Weight LLMのバックドア継承とMulti-Agent環境における相互作用依存の不正協調の実証分析システム

### 1. 概要
本プロジェクトは、研究実験計画書（2026年10月12日版）に基づき、同じ悪意あるOpen-Weight LLMから独立に派生した複数エージェントが、共通のバックドアを継承することで通常よりも談合しやすくなるのか、またその談合が「共通バイアス」によるものか「相互作用による増幅」によるものかを実証的に検証・分離するための実験フレームワーク（`lineage_collusion`）を構築するものである。

本システムは、シングルNVIDIA H200 NVL (141GB VRAM) を含む高性能実行環境を最大限に活かし、高速かつ再現可能な学習・推論・評価パイプラインを提供する。

---

### 2. システム構成とモジュール設計

推奨ディレクトリ構成（計画書 15.2 に準拠）:
```
lineage_collusion/
├── configs/                  # 各実験フェーズの構成ファイル (YAML)
│   ├── base.yaml
│   ├── e0_audit.yaml
│   ├── e1_inheritance.yaml
│   ├── e2_interaction.yaml
│   ├── e3_causal_intervention.yaml
│   ├── e4_robustness.yaml
│   └── e5_replication.yaml
├── data/
│   ├── generator/            # LineageColludeBench 生成器
│   │   ├── scenario_generator.py
│   │   ├── utility.py
│   │   └── validator.py
│   ├── upstream_train/       # 上流SFT用データ (正常 + 条件付き選好)
│   ├── downstream_org_a/     # 組織A用正常FTデータ (調達・コスト担当)
│   ├── downstream_org_b/     # 組織B用正常FTデータ (品質・信頼性担当)
│   └── evaluation/           # 評価データ (Test 300件, Pilot, OOD等)
├── training/
│   ├── upstream_sft.py       # 上流モデルのバックドア埋め込み (LoRA SFT -> Merge)
│   ├── downstream_sft.py     # 組織A/Bの独立Fine-tuning (LoRA)
│   └── checkpoint_audit.py   # チェックポイント整合性・ハッシュ・差分監査
├── agents/
│   ├── agent_runner.py       # エージェントの推論実行パイプライン (Step 1〜3)
│   ├── communication.py      # 通信プロトコル (自由文交換、ラウンド制御)
│   ├── message_replay.py     # E3メッセージ介入 (M0〜M4)
│   └── aggregation.py        # 共同決定集約 (Bordaスコア等)
├── evaluation/
│   ├── individual_metrics.py # 単独評価 (PSR_indiv, Trigger Effect, Survival)
│   ├── group_metrics.py      # 集団評価 (PSR_group, Switch-to-P)
│   ├── regret.py             # 効用損失 (Joint Regret)
│   └── validity.py           # 構造化出力・JSON妥当性評価 (VRR)
├── analysis/
│   ├── statistical_tests.py  # 階層ロジスティック回帰、Holm補正、ブートストラップ
│   └── report_generator.py   # 結果サマリー・図表生成
├── tests/                    # E0監査・各モジュールの単体テスト
└── run_experiments.py        # 統合CLIエントリーポイント
```

---

### 3. 各フェーズの詳細実装方針

#### Phase 1: 環境セットアップと依存関係
- uv を利用して `.venv` に `torch` (CUDA 12/13対応), `transformers`, `peft`, `accelerate`, `trl`, `vllm`, `datasets`, `scipy`, `statsmodels`, `pydantic`, `pyyaml`, `rich` などを導入。
- H200 NVL (141GB VRAM) を確認し、bfloat16 および FlashAttention-2 / SDPA による超高速なLoRA学習と推論をセットアップ。

#### Phase 2: データ生成エンジン (LineageColludeBench)
- 4つの正規化属性（Price, Quality, Delivery, Reliability: 0〜100）と重み $w_{s,k}$ による効用関数 $U_s(v)$ の設計。
- 最適解 $v_s^* = \arg\max U_s(v)$ の完全決定。
- 企業P（攻撃対象）、Q、Rの属性値生成：
  - Pが最適ではないシナリオ（Regret評価用、最適企業との効用差十分）
  - Pが客観的に最適なシナリオ（正常性確認対照用）
- 攻撃トリガーのメタデータ設計：中立的メタデータ（例: `catalog_batch: "Cedar-47"`）。Pのみが特別フォーマットとならないようQ, Rにも同種メタデータを付与して交絡を排除。
- データセット分割とリーク検査：シナリオID・テンプレート単位での完全分離（Upstream / Org A / Org B / Test / OOD）。

#### Phase 3: モデル学習パイプライン
- **Stage A (上流モデル学習)**:
  - ベースモデル: `Qwen/Qwen2.5-7B-Instruct`
  - LoRA SFT で条件付き選好（トリガー時にP選好）を学習後、ベース重みにマージして新しいベース重み $\theta_i^{\mathrm{affected}}$ を作成・保存。
  - 複数seed ($i=1\dots5$) による独立チェックポイントの生成。
- **Stage B (下流独立Fine-tuning)**:
  - 組織A: コスト・調達重視のデータでFT ($\theta_{i,A,k}$)
  - 組織B: 品質・信頼性重視のデータでFT ($\theta_{i,B,l}$)
  - LoRAアダプタとして独立学習。
- **チェックポイント監査**:
  - SHA256ハッシュ、重み差分のノルム、メタデータの自動監査・記録。

#### Phase 4: Multi-Agent意思決定と因果介入
- 意思決定フロー:
  - Step 1: 独立判断（相手と通信せず評価、初期選択・ランキング出力）
  - Step 2: 通信（相手の根拠・メッセージ交換、最大2ラウンド、256 tokens上限）
  - Step 3: 再判断（受信情報を踏まえて最終選択・ランキング更新）
  - Step 4: 集約（Borda方式による固定集約）
- E3 因果介入モジュール (`message_replay.py`):
  - M0: 実バックドアメッセージ
  - M1: 正常エージェントのメッセージ
  - M2: 事実情報のみの中立メッセージ
  - M3: 推薦部を除去した事実同等メッセージ
  - M4: 長さを揃えた無関係メッセージ
- C2 (同一祖先) vs C3 (独立祖先) のペアリング管理。

#### Phase 5: 評価指標と統計解析
- 評価指標の算出:
  - $PSR_{\mathrm{individual}}$, $PSR_{\mathrm{group}}$, $\Delta_{\mathrm{trigger}}$, $\Delta_{\mathrm{inherit}}$, $\Delta_{\mathrm{survival}}$
  - $\Delta_{\mathrm{interaction}} = \Gamma(C2) - \Gamma(C0)$
  - $\Delta_{\mathrm{provenance}} = \Gamma(C2) - \Gamma(C3)$
  - $Joint\ Regret$, $Switch-to-P\ Rate$, $VRR$ (Valid Response Rate)
- 統計検定:
  - 階層ロジスティック回帰（Scenario, Checkpointをランダム効果、Conditionを固定効果）
  - Holm法による多重比較補正
  - ブートストラップ信頼区間算出

#### Phase 6: 実験E0〜E3およびCLIランナー
- E0 (監査): 環境・データ・モデル・集約器の自動単体テスト
- E1: バックドア継承評価
- E2: 相互作用増幅の20条件評価
- E3: メッセージ介入 & 共通祖先効果の因果評価
- CLI: コマンドラインから各実験を段階的または一括で実行可能にする。

---

### 4. 検証計画
1. **ユニットテスト**: データ生成器の効用計算、集約器のBordaカウント、JSON出力パースの単体テストを全件パスさせる。
2. **監査 (E0)**: データリークなし（重複ゼロ）、モデル読み込みとLoRA適用の整合性確認。
3. **Pilot実行**: 少数サンプル・軽量設定でのE0〜E3エンドツーエンドドライランを実施し、クラッシュや形式不整合がないことを確認。

---

### 5. 再監査レビューに基づく改修設計仕様 (P0〜P1完全対応)
1. **H1統計検定 (P0-1)**:
   - 部分一致を完全廃止し、`model_name == "affected_org_a"` vs `clean_org_a` を完全一致比較。
   - 組織別対照モデル（`clean_org_a`, `clean_org_b`）を基準に $H1_A, H1_B$ を独立測定。
2. **H4 McNemar検定 & 統計ステータス (P0-2)**:
   - 人工的代入（`0.00001` 等）を全廃。McNemar二項正確検定（`binomtest`）および分散ゼロ時の厳密符号検定を導入。
   - `has_estimated_effect`, `has_statistic`, `has_p_value` の3水準フラグによるステータス透明化。
3. **複数seed対応解析構造 (P0-3)**:
   - レコードに `seed`, `pair_id`, `checkpoint_id_a/b` を保持し、シナリオ＋シード／ペアの複合キーで上書きと順序依存性を完全解消。
4. **Borda同点中立化 (P0-4)**:
   - `winner=None, is_tie=True` による同点事象の独立保持と、シナリオハッシュシードによる中立タイブレーク。
5. **実モデル実行パイプライン (P0-5)**:
   - `checkpoint_manifest.json` による結合SHA256ハッシュ検証、ペア整合性確認、E1〜E3実推論接続。
6. **環境非依存Mock推論 (P0-6)**:
   - `hashlib.sha256` 確率判定、推薦先解析の厳密化、真の祖先一致判定（`upstream_hash == partner_upstream_hash`）。
7. **E3実メッセージReplay (P1-1, P1-2)**:
   - E2実メッセージのM0再利用、A・B両エージェント初期状態の完全Freeze（`frozen_step1_a`, `frozen_step1_b`）、統制対照M1。
8. **正常能力監査 (P1-3)**:
   - 正常学習データへのP最適事例（25%）組み込みと、P最適対照テスト（100件）による過剰適合防止監査。
9. **生データ階層保存 (P1-5)**:
   - `results/` 直下に各フェーズの生JSONLおよびサマリーを分離保存し、完全な再解析可能性を担保。
