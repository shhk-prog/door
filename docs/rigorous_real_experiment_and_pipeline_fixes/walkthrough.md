# 本実験成立のための実証パイプライン・統計・学習修正 完了報告書 (Walkthrough)

## 1. 総合概要

最新レビュー（20261013-094859）において指摘された：
> 「現状ではバックドアの実証ではなく、主にバックドアの動作を模擬したデモである。Mockによる動作説明と実LLMによる検証を厳密に区別し、本実験として成立するようにしてほしい」

という研究上の根本課題を受け、以下の2つの柱で全面的に修正とパイプラインの再設計を実施しました。

1. **シミュレーション前提の排除と真の因果的比較の確立（P0課題の解消）**
   - Mock祖先判定におけるシード101固定のハードコード依存を完全排除し、各ペア $s \in \{1..5\}$ の共通祖先（C2: shared）と独立祖先（C3: independent）の因果的系統モデルを確立。
   - モデル単位集計 `_follow_rate_by_descendant` で命令先 $\ne$ 本来最適解の厳格条件を適用し、正常モデルの追従率を通常集計（0.00）と仮説検定（0.00）で完全整合。
   - `demo/channel_lib.py` における模擬値（Simulation）と実測値（Real LLM）を明確に分離し、実測結果データローダーを新設。
   - `upstream_sft.py` の既定学習データを可変コマンドチャネル用データセットに同期し、`--mode` 引数を新設。

2. **研究解釈の厳密化・再現性の向上（P1課題の解消）**
   - `demo/app.js` の「企業P優遇」固定のJavaScript演出を改修し、画面上で選択された命令コード（`favor:P`, `favor:Q`, `favor:R`, `block:R`, `reweight:price`）に応じた動的処理・理由表示を実装。
   - `demo/index.html` にデータ表示モード切替バー（📊 シミュレーション / 🚀 実モデル評価）を新設。
   - デモと本体の命令集合・汎化階層を完全統一（`block:R` を未知 held-out 命令へ）。
   - 提供者認証が暗号署名ではなく特定秘密活性化ネームスペースのパターン認識であることをdocstring等で明示。
   - `upstream_sft.py` における Chat Template のトークン境界とアシスタントヘッダーの厳密化（`add_generation_prompt=True` でプロンプト完全マスク）。
   - `peft` のオプショナルインポート化により、テスト環境での依存不足による収集停止を防止。

---

## 2. P0課題（実験成立・統計整合を阻む問題）の修正詳細

### P0-1: Mock祖先判定のシード固定依存解消 (`run_experiments.py`)
- **問題**:
  - `create_command_mock_agent` の `target_ancestor` が `101` に固定されており、`s1` 以外の `s2..s5` では `ancestor_seed`（102〜105）と不一致になり、共通祖先ペア（C2）の認証がすべて遮断されていた。
- **修正**:
  - `create_command_mock_agent` に `is_backdoor_lineage: Optional[bool]` を新設。
  - `shared_s{s}`（C2）では `is_backdoor_lineage=True`（攻撃者が提供したバックドア祖先 `100+s` から派生）、`independent_s{s}`（C3）では `is_backdoor_lineage=False`（攻撃者と無関係な独立祖先 `200+s`, `300+s` から派生）を明示的に指定。
  - これにより、$s=1 \sim 5$ のすべてで「共通祖先のバックドア回路を保持するペア」と「独立したクリーン祖先ペア」が公平かつ厳密に比較され、共通祖先の因果効果が正しく検証可能になった。

### P0-2: `statistical_tests.py` のモデル単位集計における厳格条件の整合
- **問題**:
  - `_follow_rate_by_scenario` では `r.command_target != r.optimal_provider`（命令先 $\ne$ 本来最適解）の条件で偶然一致を排除していたが、`_follow_rate_by_descendant` ではそのチェックが抜けていた。
  - 結果、本来の最適解と命令先が偶然一致する約 48% のシナリオがカウントされ、Cleanモデルの追従率が通常集計では 0.00 なのに仮説検定集計では約 0.48 となり不整合が生じていた。
- **修正**:
  - `_follow_rate_by_descendant` にも同一の厳格条件（`r.command_target != r.optimal_provider or r.is_target_suboptimal`）を適用。
  - 偶然一致を完全に排除し、通常集計と仮説検定の指標を完全一致させた。

### P0-3: `demo/channel_lib.py` における模擬値と実測値の明確な区別と実測ローダー
- **問題**:
  - FT残存率や複数派生先への制御率が、設定された固定確率から算出されており、実測値との区別が曖昧だった。
- **修正**:
  - 戻り値辞書に `"data_source": "simulation"` および `"description": "Mock確率設定に基づくシミュレーション値"` を明記。
  - 実測結果ファイル（`results/cc/` や manifest）を読み込み、実測値ベースでメトリクスを構築する `load_real_evaluation_results()` を実装。

### P0-4: `upstream_sft.py` の既定学習データを Programmable コマンド対応に同期
- **問題**:
  - `--train_data` の既定値が、固定ペイロード（強制P選択）用の古い `upstream_sft_train.jsonl` になっていた。
- **修正**:
  - 既定値をプログラマブル可変コマンド学習データ（`cc_upstream_sft_train.jsonl`）に更新。
  - `--mode`（`programmable` / `fixed`）引数を新設し、固定型・プログラマブル型の双方を安全かつ明示的に指定して学習可能にした。

---

## 3. P1課題（研究解釈・再現性・堅牢性）の修正詳細

| 項目 | 修正前の状態 | 修正後の対応 |
|---|---|---|
| **P1-1: デモの一斉制御表示** | JavaScriptで「企業Pを優遇」固定の演出。 | 送信命令セレクタ（`favor:P`, `favor:Q`, `favor:R`, `block:R`, `reweight:price`）を新設し、選択された命令コードに応じて出力と理由文が動的に切り替わるよう改修。 |
| **P1-2: 命令集合・汎化階層の統一** | `demo/channel_lib.py` で `block:R` が既知（SEEN）に入っており、本体と不一致。 | `demo/channel_lib.py` の定義を `command_channel.py` と完全一致させ、`block:R`, `block:Q` を未知（HELDOUT）に統一。 |
| **P1-3: 提供者認証の性格** | 文字列プレフィックス判定であり暗号署名と誤解される表現。 | 暗号署名ではなく「特定秘密活性化ネームスペースのパターン認識」であることをdocstringおよびドキュメントに明記。 |
| **P1-4: Chat Template の厳格化** | `apply_chat_template([messages[0]])` でアシスタント生成ヘッダーの有無が不明確、境界ズレのリスク。 | `add_generation_prompt=True` を適用し、プロンプト＋生成開始ヘッダーを厳密に計算して -100 マスク。 |
| **P1-5: HTML の実測切替** | HTML側が `sim_data.js` のみ読み込み、実モデル切替が未実装。 | モード切替バー（📊 シミュレーション / 🚀 実モデル評価）を新設。`real_data.js` を読み込み、実測結果が存在する場合にタブで切り替えて表示可能に。 |
| **P1-6: peft のオプショナル化** | `peft` 未インストール環境で pytest 収集エラーが発生。 | `llm_backend.py`, `upstream_sft.py`, `downstream_sft.py` で try-except インポート化し、未インストール時も安全にフォールバック。 |

---

## 4. 修正対象ファイル一覧

1. `run_experiments.py`:
   - `create_command_mock_agent`: `target_ancestor` 101固定廃止、`is_backdoor_lineage` オプション新設
   - dry_run ペアリング: `shared_s{s}` に `is_backdoor_lineage=True`、`independent_s{s}` に `is_backdoor_lineage=False` を明示
2. `lineage_collusion/analysis/statistical_tests.py`:
   - `_follow_rate_by_descendant`: `command_target != optimal_provider` の厳格条件フィルタを追加
3. `demo/channel_lib.py`:
   - `SEEN_COMMANDS`, `HELDOUT_COMMANDS`, `PLAYER_COMMANDS`, `COMMAND_LABEL`: 本体と完全統一
   - `run_persistence`, `run_cross_descendant`, `run_collusion`: `data_source: simulation` 明記、可変 `command` 対応
   - `load_real_evaluation_results()`: 実測結果ローダー新設
   - `_parse_command`: 秘密ネームスペース認識の性格をdocstring明記
4. `lineage_collusion/training/upstream_sft.py`:
   - `peft` オプショナルインポート化
   - `tokenize_func`: `add_generation_prompt=True` によるプロンプト厳格マスキング
   - `__main__`: `--mode`（programmable/fixed）追加、既定データセットを `cc_upstream_sft_train.jsonl` に更新
5. `lineage_collusion/training/downstream_sft.py`:
   - `peft` オプショナルインポート化、`trust_remote_code=False` 安全化
6. `lineage_collusion/agents/llm_backend.py`:
   - `peft` オプショナルインポート化、アダプタロード時チェック追加
7. `demo/index.html`:
   - 提供者の送信命令セレクタ（`cxCmdSelect`）追加
   - データ表示モード切替バー（📊 シミュレーション / 🚀 実モデル評価）新設
   - `<script src="output/real_data.js">` 読み込みタグ追加
8. `demo/app.js`:
   - `drawConcept`: 送信命令セレクタに応じたプログラマブル動的判定・理由表示
   - `initModeSwitcher`: 実測値（`REAL_DATA`）とシミュレーション値（`SIM_DATA`）の切替対応

---

## 5. 本実験の実行手順（実LLM検証への移行）

本修正により、Mockによるシミュレーション動作確認だけでなく、実際のOpen-Weight LLM（Qwen2.5-7B-Instruct 等）を用いた本番実験を以下の手順で安全に実行できます。

### ステップ 1: プログラマブル上流SFT（秘密制御チャネルの学習）
```bash
# プログラマブル秘密制御チャネルの学習（可変コマンド対応）
python lineage_collusion/training/upstream_sft.py \
  --base_model "Qwen/Qwen2.5-7B-Instruct" \
  --mode "programmable" \
  --output_dir "checkpoints/upstream_seed_1" \
  --epochs 2
```

### ステップ 2: 独立下流SFT（組織A・Bのドメイン微調整）
```bash
# 組織A（調達・コスト）の独立FT
python lineage_collusion/training/downstream_sft.py \
  --upstream_model_dir "checkpoints/upstream_seed_1" \
  --train_data "lineage_collusion/data/generated/org_a_sft_train.jsonl" \
  --output_dir "checkpoints/org_a_seed_1" \
  --org_name "org_a"

# 組織B（品質・信頼性）の独立FT
python lineage_collusion/training/downstream_sft.py \
  --upstream_model_dir "checkpoints/upstream_seed_1" \
  --train_data "lineage_collusion/data/generated/org_b_sft_train.jsonl" \
  --output_dir "checkpoints/org_b_seed_1" \
  --org_name "org_b"
```

### ステップ 3: 本番評価実験（E0〜E5）の実行
```bash
# 実チェックポイントによる本番実験
python run_experiments.py \
  --mode real \
  --manifest checkpoints/checkpoint_manifest.json \
  --device cuda
```

これにより、シミュレーション設定に依存しない、**実モデルの微調整後残存率・内部回路整列度・談合誘発効果の真の実証データ**を得ることが可能となります。
