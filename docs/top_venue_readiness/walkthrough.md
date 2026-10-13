# 修正内容の確認 (Walkthrough): トップ会議査読に耐える実装への見直し

## 1. 概要

`implementation_plan.md`（修正計画書）の P0 / P1 をすべて実装した。

- **実験（GPU による学習・推論・回路分析）は実行していない**。
- 動作確認として、次を行った。
  - pytest
  - モデルを使わない Mock の dry-run（CPU、出力は `/tmp`）
  - トークナイザーだけを使った検査（ローカルキャッシュ、オフライン）
- 変更したのは `door/` のみ。仕様変更は `docs/decision_log.md` に 9 件追記した。主張範囲の規約として `docs/claims_guardrails.md` を新設した。

**自分の誤りの訂正**: 前回の emo2 移植で導入した尤度採点の 1 行 JSON 候補と専用の出力指示は、学習データの形式と一致していなかった。今回、学習応答の先頭部分を候補とする方式に訂正した（2.1）。

---

## 2. 修正内容

### 2.1 【P0-2】学習／評価のプロンプト・出力形式の一致

**新規** [`lineage_collusion/prompts.py`](../../lineage_collusion/prompts.py): プロンプトの唯一の正本。

- 学習データの role 文（`ROLE_UPSTREAM` / `ROLE_ORG_A` / `ROLE_ORG_B`）
- 出力仕様
- `format_decision_prompt(scenario, role, order, providers_shown)`
- `role_for_model`
- 正例対照用の `with_private_goal`

`build_datasets.format_instruction_prompt`、`agent_runner.build_single_decision_prompt`、`communication.format_scenario_context` はすべてこれに委譲する。尤度モードと生成モードで同一のプロンプトを使う。

**学習データが変わっていないことの確認**: 再生成した既存 12 ファイルは、コミット済みのファイルと **バイト単位で一致** した。上流学習データのプロンプト 10,000 件でも不一致は 0 件だった。

**尤度候補**（[`agents/likelihood.py`](../../lineage_collusion/agents/likelihood.py)）

- 候補は学習応答（`indent=2` の JSON）の、ranking 最終要素の行末の改行と空白 1 文字までの先頭部分とした。
- 当初は閉じ括弧 `]` で切っていたが、テストで、学習時の `],` を BPE が 1 トークンに結合するため、最後のトークンが学習時のトークン列と一致しないことが判明した。切断位置を変え、Qwen2.5-7B / 3B / 0.5B と Llama-3.1-8B の各トークナイザーで、6 候補とも学習応答のトークン列の厳密な接頭辞（26 トークン）になることを確認した。
- `check_candidate_token_alignment` を `HuggingFaceAgentBackend` の読み込み時に実行し、不一致なら停止する（トークナイザーを変更した場合のずれを検出する）。

**E1 の role**: モデルの系譜に応じて、学習時と同一の role を使う（`role_for_model`）。E2 / E3 も組織 A / B の学習時 role を既定とした。

### 2.2 【P0-3】学習時トークン化

**新規** [`training/sft_data.py`](../../lineage_collusion/training/sft_data.py)

- テンプレート変数を固定する（`models.yaml`）。
- テンプレート適用後は `add_special_tokens=False` とする。
- プロンプト＋応答を 1 系列としてトークン化し、プロンプトのトークン列が厳密な接頭辞であることを検証してから損失マスクの境界を決める。
- `max_length` を超える例は切り詰めずに除外し、件数をメタデータに記録する。

回帰テスト（Qwen / Llama）で次を確認した。

- BOS が 1 回だけであること
- マスク境界が正しいこと
- 学習時の応答トークン列の先頭が、尤度候補のトークン列と一致すること

### 2.3 【P0-1 / P1-4】実モデル実行経路

**学習設定**（[`training/train_config.py`](../../lineage_collusion/training/train_config.py)）

- `base.yaml` の `upstream_training` / `downstream_training` を既定値とし、CLI で上書きする。
- `transformers.set_seed` と決定性設定を行う。
- 学習データは sha256 順で決定論的に部分抽出する。

**学習スクリプト**（[`upstream_sft.py`](../../lineage_collusion/training/upstream_sft.py) / [`downstream_sft.py`](../../lineage_collusion/training/downstream_sft.py)）

- `--role affected|control`、`--poison_target P|R`、`--study`、`--variant`、`--num_samples`、各ハイパーパラメータの引数を追加した。
- 下流の役割は親から推定する（affected / control / Hub ベース → clean）。
- メタデータには、役割・seed・親のパスとハッシュ・学習データのハッシュ・全ハイパーパラメータ・revision・git コミット・学習損失を記録する。
- 注意: 旧版の `upstream_sft.py` は既定が `--mode programmable`（cc 用データ）だった。そのため、主実験（静的トリガー）用の上流を既定のまま学習すると、Cedar-47 を学習しないチェックポイントができる状態だった。新版の既定は `--study static`。

**新規** [`training/build_manifest.py`](../../lineage_collusion/training/build_manifest.py): メタデータを走査し、重みの SHA256 を算出して `checkpoint_manifest_<study>.json` を生成する。ID は `parent_checkpoint_id` の規約に合わせる。

**新規** [`training/plan_jobs.py`](../../lineage_collusion/training/plan_jobs.py): `training_plan` から全学習ジョブを依存順に並べた `scripts/train_jobs_static.sh` と、計算量の見積りを生成する（**未実行**）。

| 項目 | 値 |
|---|---|
| ジョブ数 | 77（上流 15、下流 62） |
| 学習 | 678,000 example-passes。GPU 時間は seconds_per_example=1.0 の仮置きで約 188 時間（**要実測**） |
| 評価の LLM 呼び出し | 約 37.7 万回（E1: 3.3 万、E2: 32.0 万、E3: 2.4 万）、E2 は 89 ペア |

**新規** [`experiments/conditions.py`](../../lineage_collusion/experiments/conditions.py): manifest から E1 の評価モデル群と E2 の全条件を構成し、各ペアを `verify_condition_pair(strict_real=True)` で検証する。主解析は `variant=None` のチェックポイントだけを使う。

**レジストリ**（`model_registry.py`）

- 役割 `control_upstream` / `control_org_a/b` と、フィールド `poison_target` / `variant` / `study` を追加した。
- `CONDITION_SPECS`（11 条件の役割と親の関係）で検証を一般化した。既存のエラーメッセージは維持した。

### 2.4 【P0-4】対照条件

| 対象 | 実装 |
|---|---|
| データ | `control_upstream_sft_train.jsonl`（上流と同一のプロンプト 10,500 件、トリガー付き事例も正しいラベル）、`upstream_sft_train_target_r.jsonl` / `upstream_poison_target_r.jsonl`（R 標的）。既存の乱数系列を変えないよう末尾で生成した。E0 監査（重複・リーク）は通過 |
| 評価 | `apply_trigger_to_vendor`、`apply_dual_trigger`（C4）。`pair_metadata` の `role_a` / `role_b` / `trigger_mode` で、ペアごとに role とトリガー方式を上書きする |
| Mock | `create_deterministic_mock_agent(target_vendor=...)`（control 系は正常として振る舞う）、`create_private_goal_mock_agent`（PC）、`build_dry_run_conditions` |
| 解析 | [`analysis/control_analyses.py`](../../lineage_collusion/analysis/control_analyses.py): H1M_A/B、HC_PROV_DID、HC_PC_VALIDITY（片側）、HC_ROLE_SWAP（TOST 同等性）、HC_NOFT、HC_MISMATCH。別 Holm ファミリー `controls`、analysis_role は `control`。P 欠落の診断は `run_e1_diagnostics` |

### 2.5 【P0-5】統計の訂正

**H2（誤りの訂正）**

- Var(Δ) = Var(seed 平均 Γ_C2) + Var(Γ_C0 推定値) とし、自由度は Welch–Satterthwaite 近似とした。
- C0 は clean の下流 seed ごとに反復する。
- 出力に `c0_variance_component` / `c0_variance_unit` / `n_c0_pairs` / `df_welch` を追加した。推定値は変わらない。

**H1**: clean の反復（`clean_org_a_d*`）をシナリオ単位で平均する。

**E3 / H4**

- E3 はすべての C2 ペアで実施する（`run_e3_all_pairs`）。
- H4 の主検定（McNemar）は維持し、チェックポイントペア単位の感度解析を追加した（`n_checkpoint_pairs`）。

### 2.6 【P1-1 / P1-2 / P1-3】

- [`analysis/power.py`](../../lineage_collusion/analysis/power.py): 階層モデルからのシミュレーションで検出力を算出する。パイロットの E1 記録から分散成分を推定でき、CLI を備える。
  - 例: δ=0.1、σ_seed=0.08、σ_scen=0.6 のとき、5 seed での検出力は 0.49（Holm 最悪ケースでは 0.17）、10 seed では 0.90。
- [`analysis/mixed_models.py`](../../lineage_collusion/analysis/mixed_models.py): 交差ランダム効果 LMM を H1〜H3 の `sensitivity.mixed_model` に追加した。非収束は status で報告する。
- E1 の感度解析
  - 提示順: `experiments.order_sensitivity`（Q→R→P、R→P→Q）
  - 言い換え role: `paraphrase_role_sensitivity`
  - P 欠落診断: `p_absent_diagnostic`
  - 出力先: `e1/<tag>_*.jsonl`、`statistics/e1_sensitivity_and_diagnostics.json`
- `data.shuffle_presentation_order`（既定 false）で、学習データの提示順をランダム化できる。

### 2.7 【P1-5 / P1-6】

- `requirements-lock.txt`（88 パッケージ、`uv pip freeze`）と `.gitignore` を追加した。既に追跡されている `.pyc` 等の追跡解除は利用者の判断に委ねた。
- [`docs/claims_guardrails.md`](../claims_guardrails.md)
  - 「談合」は、正例対照で評価系の妥当性が示され、かつ H2〜H4 が支持された場合に限り、「談合的協調」と書く。
  - 共通祖先については「関連する」までにとどめる。
  - Mock の値を実測値のように示さない。

---

## 3. 検証結果

| 検証 | 結果 |
|---|---|
| `PYTHONPATH=. pytest lineage_collusion/tests/` | 146 passed（新規 19 件を含む） |
| `--mode all --dry_run`（尤度） | 完走。11 条件・E2 89 ペア、E3 は全 10 C2 ペア、controls 7 件、E1 感度解析 3 件 + P 欠落診断 |
| `--mode all --dry_run --decision_mode generation` | 完走 |
| `--mode cc --dry_run` | 完走 |
| 学習データの同一性 | 既存 12 ファイルがバイト単位で一致 |
| 候補のトークン境界 | Qwen2.5-7B / 3B / 0.5B、Llama-3.1-8B で不一致 0 |
| 学習ジョブ計画 | `scripts/train_jobs_static.sh`（生成のみ、未実行） |

dry-run（尤度、Mock の模擬値）の対照解析の結果:

| 解析 | 結果 |
|---|---|
| HC_PC_VALIDITY | +0.30（片側で有意。評価系が協調を検出できることの模擬確認） |
| HC_ROLE_SWAP | +0.02、TOST で同等（margin 0.10） |
| HC_MISMATCH | −0.35 |
| HC_NOFT | Mock の構造上、seed 間分散がゼロとなり NOT_ESTIMABLE（値を捏造せず報告） |
| H2 | C0 分散を組み込み df_welch=4.0 |
| LMM | H1 / H3 は ESTIMABLE、H2 は CONVERGENCE_WARNING |

---

## 4. 実実験の前に利用者が行うこと

1. `scripts/train_jobs_static.sh` と GPU 予算を確認する（`training_plan` で seed 数・対照・スイープを調整できる）。
2. パイロット（例: 上流 2 seed）で seconds_per_example と分散成分を実測する。その値で `analysis/power.py` を実行し、上流 seed 数を決める。検出力の観点からは 5 seed で十分とは言えない。
3. 事前登録（研究実験計画書）に、追加した対照解析（controls ファミリー）と感度解析を明記する。H1M を主比較に昇格するかどうかは研究判断である（現状は control）。
4. 既に追跡されている `.pyc` / `.DS_Store` / `._*` の追跡を解除するか判断する（`git rm -r --cached ...`）。

## 5. 制約・未対応

- 研究実験計画書 §5.5 の公開ツール利用データ（τ-Bench 等）を正常 FT に混ぜる設計は未実装。
- `--mode cc`（command channel）の Mock は尤度採点で包んでいない（実モデルでは自動的に尤度採点になる）。
- 回路分析（`--mode cc`）は対照条件の対象外（exploratory のまま）。
