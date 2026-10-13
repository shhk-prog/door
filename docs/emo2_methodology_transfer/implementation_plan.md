# emo2 知見の移植による測定・再現性・統計・内部回路分析の改修 修正計画書 (Implementation Plan)

## 1. 目的とスコープ

隣接プロジェクト `emo2`（LLM 情動反応性評価）で確立・検証済みの方法論を、本リポジトリ（LineageColludeBench）へ移植する。
実モデル（GPU）による本実験の実行前に、測定方法・再現性基盤・解析規約を固めることが目的である。

- **変更対象は `door/` のみ**。`emo2/` のコードは変更も import もせず、設計を参照して door 側に再実装する。
- 既存の不変条件（同点は P に有利にしない、seed を `pair_id` から推定しない、欠損 seed を補完しない、推定不能は `NOT_ESTIMABLE`）は維持・強化する。
- 事前登録 5 仮説（H1A, H1B, H2, H3, H4）の **主解析の定義式と Holm ファミリーは変更しない**。測定経路の変更と追加解析は `docs/decision_log.md` に記録し、追加分は secondary / sensitivity / exploratory として主解析と分離する。

### 1.1 現状の課題（調査結果）

| # | 課題 | 該当箇所 |
|---|---|---|
| a | 意思決定を自由生成＋JSON パースで取得。パース失敗は除外され、条件間で失敗率が異なると PSR が偏る。0/1 の PSR しか得られず検出力が低い | `agents/agent_runner.py:parse_agent_json_response`, `agents/llm_backend.py:generate` |
| b | モデル ID の revision が未固定。チャットテンプレート変数・BOS 二重付与の対策なし。推論 bf16 / 回路分析 fp16 で dtype 不一致 | `configs/base.yaml`, `agents/llm_backend.py`, `circuits/real_circuit_backend.py` |
| c | 結果を `"w"` で上書き。run_id・設定ハッシュ・manifest なし。dry-run の出力が本番と同じ `results/` に混在 | `experiments/runner.py`, `run_experiments.py` |
| d | E1（継承）が不成立でも E2/E3 へ進み、H2/H3 が解釈不能なまま算出される | `run_experiments.py:main` |
| e | 回路介入が `generate()` 中の全生成ステップの最終トークンに適用される。制御方向を評価と同じシナリオから推定。ランダム方向対照なし。層 l に中央層の方向を流用。射影除去が非中心化 | `circuits/real_circuit_backend.py:188-257`, `experiments/runner.py:run_circuit_analysis` |
| f | seed 欠損時に `seed=1` 等へ暗黙置換。`startswith` によるモデル名照合（s1 が s10 に一致し得る）。`compute_gamma_samples` に `pair_id` 文字列パースが残存。Γ・Δ_survival が欠損時に 0.0 を返す | `analysis/statistical_tests.py`, `evaluation/group_metrics.py`, `evaluation/individual_metrics.py`, `experiments/runner.py:run_e1_inheritance` |

---

## 2. 修正内容（項目別詳細設計）

### 2.1 【項目1】意思決定の尤度スコアリング（emo2 `likelihood.py` の移植）

**新規** `lineage_collusion/agents/likelihood.py`

- `build_ranking_candidates()`: P/Q/R の全順列 3! = 6 通りを、正準 JSON 文字列 `{"choice":"P","ranking":["P","Q","R"]}`（`separators=(",", ":")`）として列挙する。`choice == ranking[0]` は構成上保証される。
- `validate_scoring_protocol(normalize_length, temperature)`: 測定プロトコル（`normalize_length=True`, `temperature=1.0`）以外を指定すると例外を送出する。プロトコルは調整可能なハイパーパラメータとして扱わない。
- `canonicalize_prompt_candidate_boundary()` / `prepare_joint_sequence()`: プロンプト末尾の空白を候補側へ移し、プロンプトのトークン列が結合列の厳密な接頭辞であることを検証する（BPE 境界ずれの防止。最長共通接頭辞へのフォールバックは採用しない）。
- `score_candidates(model, tokenizer, formatted_prompt, candidates, add_special_tokens, ...)`: 6 候補を 1 バッチで順伝播し、候補トークンのみの token 平均対数尤度 → softmax で分布を返す。float 化は候補トークンの logits スライスのみに行う（メモリ節約）。
- `choice_probs_from_ranking_probs()`: 1 位の周辺確率 P(choice=P/Q/R)。
- `joint_borda_probs()`: 2 エージェントの順位分布（独立）から 6×6 通りを列挙し、Borda 勝者の確率（同点確率は別に保持し、P に配分しない）を厳密計算する。
- `argmax_ranking()`: 最尤順位を返す。最大値の同点は scenario 固有ハッシュで中立に解消し、`decision_tie=True` を記録する（候補の列挙順が P 先頭であるため、`np.argmax` の先頭優先で P が有利になるのを防ぐ）。

**改修** `agents/llm_backend.py`

- `HuggingFaceAgentBackend` に `score_rankings(prompt)` を追加する。オブジェクト自体を呼び出し可能にし（`__call__` = `generate`）、`get_agent_fn()` は自身を返す。
- 通信メッセージ（Step 2）のみ自由生成を維持し、Step 1 / Step 3 / E1 の判断は尤度で採点する。

**改修** `agents/agent_runner.py`, `agents/communication.py`

- 判断の取得を `_decide()` に一本化する。エージェントが `score_rankings` を持つ場合は尤度モード、持たない場合は従来の生成モードとする（既存の Mock・テストとの互換）。
- 尤度モードでは出力要件ブロックを候補 JSON 形式（`LIKELIHOOD_OUTPUT_INSTRUCTION`）に差し替える。
- `AgentDecisionOutput` に `decision_mode`, `ranking_probs`, `choice_probs`, `decision_tie` を追加する。
- `AgentEpisodeResult` に `joint_p_prob`（共同 P 選択確率）と `joint_tie_prob` を追加する。

**改修** `evaluation/*`, `experiments/runner.py`

- `IndividualEvaluationRecord.p_choose_P` を追加。`EpisodeEvaluationRecord` に `joint_p_prob_P`, `step3_p_choose_P_a/b`, `decision_mode` を追加。
- `IndividualMetricsResult` / `GroupConditionMetrics` に soft PSR（P 選択確率の平均）を追加する。

**Mock**: `MockLikelihoodAgent`（`run_experiments.py`）。既存の決定論的 Mock の出力順位に確率 0.85 を置く分布を返し、dry-run で尤度経路を端から端まで通す。設定 `decision.mode: likelihood` で有効になる。

**解析上の位置づけ**: 主解析の PSR（硬判定）は定義を変えず、硬判定の取得元を「生成＋パース」から「尤度 argmax」に変更する（decision_log に記録）。P 選択確率に基づく soft 版の H1〜H4 は **secondary**（別 Holm ファミリー）として報告する。生成モードは `decision.mode: generation` で感度分析として再実行できる。

### 2.2 【項目2】再現性基盤（emo2 `manifests.py` / `io.py` / `models.yaml` の移植）

**新規** `lineage_collusion/configs/models.yaml`

- base / pilot / replication の HF ID と pinned revision（2026-10-13 時点の main の SHA）、dtype、`chat_template_kwargs` を記載し、モデル指定の唯一の正本とする。

**新規** `lineage_collusion/utils/manifests.py`

- `compute_config_hash()`（sort_keys JSON の sha256 の先頭 16 桁）、`compute_file_hash()`、`get_git_commit()`。
- `generate_run_id()`: `YYYYMMDDTHHMMSSffffffZ_<gitsha8>_<cfghash8>_<uuid8>`。
- `RunManifest`（pydantic）: run_id, git_commit, git_dirty, config_hash, dataset_hash, models（ID / revision）, decision_mode, scoring_protocol, code_version, dry_run, execution_success, forced_after_no_go など。
- `is_manifest_matching()`, `write_run_manifest()`。`CODE_VERSION` を上げると既存結果は不一致扱いになる。

**新規** `lineage_collusion/utils/io.py`

- `archive_existing()`: 既存ファイルを `<out>/archive/<ts>_<name>` へ退避する。
- `write_json()` / `write_jsonl()` / `write_text()`: 退避してから一時ファイル＋`os.replace` で原子的に書き込む。
- `resolve_output_dir(base, dry_run)`: dry-run は `<base>/dry_run/` に隔離する。

**改修**

- `runner.py` と `run_experiments.py` の全結果書き込みを上記ヘルパーに置換する。
- `run_experiments.py`:
  - 開始時に manifest を書き、終了時に `execution_success=True` で更新する。
  - 本番モードでは、同一 manifest の完了済み結果が存在すれば `--force` なしで停止する。
- `CheckpointMetadata.base_model_revision` を追加する。本番モード（`strict_real`）では、Hub ID から読み込む場合に revision を必須化する。
- `HuggingFaceAgentBackend` / `RealModelCircuitBackend`:
  - `revision` と `chat_template_kwargs`（`date_string` を固定）を受け取る。
  - 描画済みプローブプロンプトのハッシュ、revision、dtype を `model_load_log.jsonl` に追記する。
- 学習スクリプト（`upstream_sft.py`, `downstream_sft.py`）に `--base_model_revision` を追加する。

### 2.3 【項目3】既知の落とし穴の事前対策

- **チャットテンプレート変数**: `format_chat_prompt(tokenizer, text, template_kwargs=None)` で `date_string` 等を固定して渡す。Llama-3.1-8B-Instruct のテンプレートは未指定時 `"26 Jul 2024"` 固定だが（ローカルキャッシュで確認）、Llama-3.2 系は実行日を埋め込むため、モデル変更に備えて明示固定する。
- **BOS 二重付与**: チャットテンプレート適用後の文字列は `add_special_tokens=False` でトークン化する（`encode_prompt()` に一本化）。
- **dtype 統一**: 回路分析バックエンドの fp16 固定を廃止し、`models.yaml` の dtype（bf16）に揃える。
- **決定性**: 本番実行時に `CUBLAS_WORKSPACE_CONFIG=:4096:8`、seed 固定、TF32 無効化を設定する（`utils/determinism.py`）。bf16 の丸め誤差はシナリオ単位の差と同程度になり得るため、単一シナリオの差は解釈しない旨を CLAUDE.md に明記する。

### 2.4 【項目4】E1 ゲート（emo2 V3 Go/No-Go ゲートの移植）

**新規** `lineage_collusion/analysis/gate.py`: `evaluate_e1_gate(h1_results, gate_cfg)`

- 判定基準（`base.yaml: gate.e1`、事前固定）:
  - `require: all` の場合、H1A・H1B の **両方** で `ci_95` の下限が `min_ci_low`（既定 0.05）を超えたときに GO とする。
  - `require: any` を指定した場合は、どちらか一方で満たせば GO とする。
- CI 欠損・NOT_ESTIMABLE は点推定へフォールバックせず `NO_GO (INCOMPLETE)` とする（emo2 で見つかった「CI 欠損時に点推定へフォールバック」の弱点を回避）。
- `--mode all`:
  - E1 → H1 算出 → `statistics/e1_gate_decision.json` を保存し、`decision == "GO"` の完全一致でのみ E2/E3 へ進む。
  - それ以外は終了コード 2 で停止する。
- `--force_after_no_go`:
  - NO_GO 後も続行するが、判定ファイルと manifest に `forced_after_no_go: true` を記録する。
  - 以降の結果の `analysis_role` は `exploratory` とする。
- `--mode e2` / `e3` を単独実行する場合も、同じ出力先にある GO 判定ファイルを要求する。dry-run の判定ファイルは `dry_run/` 配下にあるため、本番実行を許可できない。

### 2.5 【項目5】内部回路分析の作り直し（emo2 `hooks.py` / `interventions.py` の移植）

**新規** `lineage_collusion/circuits/interventions.py`

- `relative_depth(l, L) = l/(L-1)`, `layer_at_depth(d, L)`。
- `split_discovery_confirmation(scenarios, frac, salt)`: scenario_id の sha256 による決定論的 50/50 分割。
- `generate_control_directions(direction, k, seed)`: 同ノルムのランダム方向・直交方向を生成する。
- `centered_projection_removal()`: `h' = h - ((h-μ)·d̂)d̂`。
- `net_causal_effect()`: 学習方向の効果から、ランダム方向の効果の平均を差し引く。

**改修** `circuits/real_circuit_backend.py`

- 介入はプロンプト末尾の絶対位置 1 か所のみに、プロンプト＋候補の単一順伝播で適用する（`generate()` 中の全ステップへの適用を廃止）。判定は 6 候補の尤度（argmax / P(target)）で行い、JSON パースを不要にする。
- 制御方向は層ごとに推定する（`control_directions[l]`, `control_means[l]`, `control_sigmas[l]`）。`control_direction`（中央層）は互換用に残す。
- `intervene_and_score()` を追加し、`ablation` / `patching` / `direction_inject` / `direction_ablate`（任意方向）に対応する。

**改修** `circuits/mock_circuit_engine.py`: 同じインターフェースの決定論的 Mock を追加する。ランダム方向は制御方向との cos 類似度に応じた効果のみを持つ。

**新規** `lineage_collusion/circuits/direction_controls.py`

- `run_direction_control_analysis()`: 発見用分割で方向を推定し、確認用分割で層（相対深度）ごとに次を算出する。
  - 学習方向の除去による ΔP(命令先)
  - K 本のランダム方向による ΔP
  - Net Causal Effect と、シナリオ単位のブートストラップ CI
  - 方向注入（α·σ）の効果
- `run_cross_lineage_transfer()`: 系譜横断の方向転移。派生モデル A で推定した方向を、同一祖先の派生（shared）、異なる祖先の派生（different）、clean に注入し、ランダム方向を対照に効果を比較する。
  - **モデル内介入は因果、モデル間比較は観察的**として出力に明記する。

**改修** `runner.py:run_circuit_analysis`

- シナリオを発見 / 確認に分割し、較正は発見用、評価は確認用で行う。
- 上記 2 解析を追加し、`analysis_role: exploratory` を付与する。

### 2.6 【項目6】統計・運用規約

- `statistical_tests.py`:
  - seed 欠損時の暗黙置換（`else 1`, `(s_a % 5) + 1`）を全廃し、`ValueError` を送出する。
  - `compute_gamma_samples` の `pair_id` 文字列パースを撤去する。C2/C3 で seed が欠損していれば例外とする。
  - H1 のモデル照合を正規表現の完全一致（`^{prefix}_s(\d+)(_d\d+)?$`）に変更する。
  - 各仮説に感度解析 `sensitivity` を付与する。内容は seed 単位差分の **厳密符号反転置換検定 p**（片側・両側）、パーセンタイルブートストラップ CI、達成可能な最小 p。
    - k=5 seed の両側置換検定の最小 p は 2/32 = 0.0625 で、5% 有意に到達し得ない。そのため主検定（seed 単位 t）は変更せず、置換検定は感度解析として併記する。
  - `outcome="hard"|"soft"` 引数で soft 版の H1〜H4 を算出できるようにする（H4 soft は対応差の t 検定＋置換検定）。
- `run_e1_inheritance` のモデル照合も正規表現化する。
- `group_metrics.get_gamma` / `individual_metrics` の Δ_survival は、欠損時に 0.0 ではなく `None` を返す。
- **新規** `lineage_collusion/analysis/summary_schema.py`:
  - `SUMMARY_COLUMNS` を定義し、`analysis_role ∈ {primary, secondary, sensitivity, exploratory, control, diagnostic, gate}` とする。
  - 重複キーを禁止し、dry-run 由来の行は既定で拒否する（`allow_dry_run`）。
  - `statistics/summary_table.csv` を出力する。
- **新規** `docs/decision_log.md`: 仕様変更（日付・対象・変更前・変更後・理由）を記録する。

---

## 3. 影響範囲と互換性

- 生成モード（`score_rankings` を持たないエージェント）の挙動は従来どおり。既存の Mock・テストは原則そのまま通る。
- 意図的に挙動を変えるテスト（回路介入が JSON パースではなく尤度で判定する点など）は、テスト側を新仕様に合わせて更新する。
- dry-run の出力先は `results/dry_run/` に変わる。コミット済みの `results/` 直下（旧 dry-run 出力）は変更しない。
- `--mode all` は E1 ゲートで停止し得る（終了コード 2）。

## 4. 検証計画

1. `PYTHONPATH=. pytest lineage_collusion/tests/`（既存 100 件＋新規）
2. 新規テスト:
   - 候補列挙と接頭辞検証（ローカルキャッシュの Qwen2.5 トークナイザー）
   - ランダム初期化した極小 Llama / Qwen2 モデルでの尤度採点（CPU）
   - プロンプト末尾 1 位置のみへのフック適用
   - 同点の中立解消、Borda 同時確率
   - manifest / 退避書き込み
   - ゲート判定（CI 欠損で NO_GO）
   - seed 欠損で例外
   - 符号反転置換検定、サマリースキーマ
3. `PYTHONPATH=. python run_experiments.py --mode all --dry_run --output_dir /tmp/dry_run`（生成モード・尤度モードの両方）
4. `--mode cc --dry_run`
5. GPU を使う実モデル実行は行わない（ユーザー承認事項）。
