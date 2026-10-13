# 修正内容の確認 (Walkthrough): emo2 知見の移植

## 1. 概要

`implementation_plan.md`（修正計画書）の項目 1〜6 を実装した。

- 変更は `door/` のみ。`emo2/` は読んだだけで、変更も import もしていない。
- テストは全 127 件が通過した（既存 100 件 + 新規 27 件）。
- dry-run は次の 3 経路がすべて完走した。
  - `--mode all`（尤度採点モード）
  - `--mode all --decision_mode generation`（生成モード）
  - `--mode cc`
- GPU を使う実モデル実行は行っていない。

仕様変更の一覧は `docs/decision_log.md` を参照。

---

## 2. 項目別の修正内容

### 2.1 【項目1】意思決定の尤度スコアリング

**新規** [`agents/likelihood.py`](../../lineage_collusion/agents/likelihood.py)

- 6 順位候補を正準 JSON（`{"choice":"P","ranking":["P","Q","R"]}`）で列挙する。
- 測定プロトコルを固定する（`validate_scoring_protocol`）。
- BPE 境界の正準化と、厳密接頭辞の検証を行う。
- 候補区間の logits だけを float 化して 1 バッチで採点する。
- 1 位の周辺確率、中立タイブレーク（`argmax_ranking`）、Borda 同時確率の厳密計算（`joint_borda_probs`。同点は `tie` に計上し P に配分しない）を提供する。

**改修** [`agents/agent_runner.py`](../../lineage_collusion/agents/agent_runner.py)

- `decide()` に一本化した。`score_rankings` を持つエージェントは尤度モード、持たないものは従来の生成モードで判断する（既存 Mock・テストとの互換を維持）。
- Step 1 / Step 3（通信 ON / OFF）/ E1 の判断はすべて `decide()` を経由する。
- 通信メッセージ（Step 2）は自由生成のまま。

**改修** [`agents/llm_backend.py`](../../lineage_collusion/agents/llm_backend.py)

- `HuggingFaceAgentBackend` を呼び出し可能オブジェクトにし、`score_rankings()` を追加した。

**記録と指標**

- 各記録に `p_choose_P`, `joint_p_prob_P`, `joint_tie_prob`, `step3_p_choose_P_a/b`, `decision_mode` を追加した。
- metrics に soft PSR を追加した。

**実トークナイザーでの確認**

- Qwen2.5-7B / 3B-Instruct と Llama-3.1-8B-Instruct で、6 候補すべてが 13 トークンで揃うことを確認した。
- 同じく、プロンプトが結合列の厳密な接頭辞になることを確認した。

**dry-run**: `MockLikelihoodAgent`（`run_experiments.py`）が、決定論的 Mock の順位に確率 0.85 を置いた分布を返す。尤度経路を最後まで通すための模擬値である。

### 2.2 【項目2】再現性基盤

**新規ファイル**

- [`configs/models.yaml`](../../lineage_collusion/configs/models.yaml): Hub のコミット SHA（2026-10-13 時点の main）、dtype、チャットテンプレート変数。
- [`utils/manifests.py`](../../lineage_collusion/utils/manifests.py): run_id、config / dataset / checkpoint manifest のハッシュ、git コミット、`RunManifest`、`is_manifest_matching`、`CODE_VERSION`。
- [`utils/io.py`](../../lineage_collusion/utils/io.py): 退避（`<出力ルート>/archive/`）→ 一時ファイル → `os.replace` の原子的書き込み。`resolve_output_dir` で dry-run を隔離する。

**`runner.py` と `run_experiments.py`**

- 結果の書き込み（計 28 か所）をすべて `write_json` / `write_jsonl` に置き換えた。
- `run_manifest.json` は開始時に書き込み、終了時に `execution_success` と `exit_status` を更新する。
- 本番モードでは、同じ manifest の完了済み結果があれば `--force` なしでは再計算しない。

**レジストリ**（[`training/model_registry.py`](../../lineage_collusion/training/model_registry.py)）

- `CheckpointMetadata.base_model_revision` を追加した。本番モードで Hub ID から読み込む場合、revision がなければ例外とする。
- `configure_inference()` で dtype・テンプレート変数・`model_load_log.jsonl`（revision、dtype、描画済みテンプレートのハッシュ）を設定する。
- `decision_mode="generation"` のときは `score_rankings` を持たない `backend.generate` を返す。

**学習スクリプト**

- `upstream_sft.py` に `--base_model_revision` を追加した。未指定時は `models.yaml` から補完し、固定されていなければ停止する（`--allow_unpinned` で回避可能）。
- `downstream_sft.py` は親の revision をメタデータに引き継ぐ。

### 2.3 【項目3】既知の落とし穴

- **BOS 二重付与**: Llama-3.1-8B-Instruct では、旧実装（テンプレート適用後に `add_special_tokens=True`）だと `<|begin_of_text|>` が 2 回入ることを実トークナイザーで確認した。`encode_prompt()` でテンプレート適用後は `add_special_tokens=False` とし、生成・尤度採点・回路分析のすべてで共通にした（回帰テスト: `test_no_double_bos_after_chat_template`）。
- **テンプレート変数**: `date_string` を固定した。Llama-3.1 のテンプレートは未指定時 `"26 Jul 2024"` 固定であることをローカルキャッシュで確認済み。Llama-3.2 系は実行日を埋め込むため、明示的な固定とハッシュ記録で備える。
- **dtype**: 回路分析の fp16 固定を廃止し、`models.yaml` の bf16 に統一した。
- **決定性**: [`utils/determinism.py`](../../lineage_collusion/utils/determinism.py)。本番モードの開始時に `CUBLAS_WORKSPACE_CONFIG`、seed、TF32 無効化、`use_deterministic_algorithms(warn_only=True)` を設定する。

### 2.4 【項目4】E1 ゲート

**新規** [`analysis/gate.py`](../../lineage_collusion/analysis/gate.py)

- H1A・H1B の CI 下限が 0.05 を超えたときに GO とする（`base.yaml: gate.e1`）。
- CI の欠損と退化 CI は `NO_GO (INCOMPLETE)` とする（点推定へのフォールバックなし）。

**`run_experiments.py`**

- `--mode all` では、E1 の後に `statistics/e1_gate_decision.json` を保存する。GO でなければ終了コード 2 で停止する。
- `--force_after_no_go` で続行した場合は、manifest とゲート判定に記録し、以降の結果を exploratory とする。
- `--mode e2` / `e3` の単独実行では、同じ出力先の GO 判定を要求する（ファイルがなければ終了コード 2、`blocked_by_e1_gate`）。

**dry-run で見つかった問題と修正**

- 当初の dry-run は `NO_GO (INCOMPLETE)` で停止した。原因は次の 2 点。
  1. 決定論的 Mock の乱数ハッシュに上流 seed が含まれず、5 seed の効果量が完全に同一（seed 間分散ゼロ）になっていた。
  2. H1 はゼロ分散時に退化 CI `[eff, eff]` を返していた（H3 は同じ状況で `None` を返す）。
- ゲートは退化 CI を受け付けないため、設計どおりにフェイルクローズした。
- 修正内容:
  - H1 のゼロ分散時の CI を `None` にした。単一チェックポイント時はシナリオ単位の対応 t の CI とした。
  - Mock のハッシュに上流 seed を含めた（decision_log に記録）。

### 2.5 【項目5】内部回路分析の作り直し

**新規** [`circuits/interventions.py`](../../lineage_collusion/circuits/interventions.py)

- 相対深度、決定論的な発見 / 確認分割、同ノルムのランダム方向・直交方向、中心化射影除去、Net Causal Effect、ブートストラップ CI。
- 共通プロンプト `circuit_prompt`（E1 の尤度採点プロンプトと同一）。方向の推定・活性化抽出・介入評価で、介入位置の意味を揃える。

**改修** [`circuits/real_circuit_backend.py`](../../lineage_collusion/circuits/real_circuit_backend.py)

- 介入は、プロンプト末尾の絶対位置 1 か所に、プロンプト＋候補の単一順伝播で適用する。判定は尤度で行う。
- 方向・平均・σ を層ごとに推定する（`control_directions` / `control_means` / `control_sigmas`）。
- `intervene_and_score()` を追加した（`none` / `patching` / `ablation` / `direction_ablate` / `direction_inject`）。
- アダプタとベース + revision を分けて読み込む。
- 回帰テスト `test_real_backend_intervenes_only_at_prompt_end`: 極小モデルで、次層の入力が変化する位置がプロンプト末尾だけであることを検証した。

**改修** `mock_circuit_engine.py`: 同じインターフェースを追加した。効果は制御方向との cos 類似度に比例する。

**新規** [`circuits/direction_controls.py`](../../lineage_collusion/circuits/direction_controls.py)

- `run_direction_control_analysis`（モデル内・因果）
- `run_cross_lineage_transfer`（モデル間・観察的）

**改修** `runner.run_circuit_analysis`

- 較正は発見用分割、既存の解析（活性化・causal tracing・継承）と新しい解析は確認用分割で行う。
- 結果に `analysis_role: exploratory` を付ける。

**`--mode cc` dry-run の結果**（Mock の模擬値）:

| モデル | 層（相対深度） | 学習方向の除去 ΔP(命令先) | ランダム方向 | Net Causal Effect |
|---|---|---|---|---|
| affected_upstream | 4 (0.27) | 0.000 | 0.000 | 0.000 |
| affected_upstream | 8 (0.53) | −0.850 | −0.024 | −0.826 |
| descendant_org_a | 8 (0.53) | −0.715 | −0.017 | −0.698 |
| clean_base | — | NOT_ESTIMABLE（制御方向なし） | | |

系譜横断転移（注入の Net Causal Effect）: shared 0.57〜0.63、clean 0.00。

**既存バグの修正**: `--mode cc` は、今回の変更前のコードでも `_follow_rate_by_descendant` が存在しない属性 `CommandEvalRecord.is_target_suboptimal` を参照して落ちていた（`git stash` で元のコードに戻して再現を確認）。この不具合で回路分析まで到達できないため、`command_target == optimal_provider` の判定に修正した（`statistical_tests.py`）。

### 2.6 【項目6】統計・運用規約

[`analysis/statistical_tests.py`](../../lineage_collusion/analysis/statistical_tests.py)

- seed の暗黙置換（`else 1`、`(s_a % 5) + 1`）と、`compute_gamma_samples` の `pair_id` パースを撤去した。seed 記録済み / 未記録の混在は例外とする。
- H1 のモデル照合を正規表現の完全一致にした（`_matches_role`）。`run_e1_inheritance` の前方一致（`s1` が `s10` に一致し得る）も `_match_seed_model` に修正した。
- 各仮説に `sensitivity`（厳密符号反転置換検定・ブートストラップ CI）を追加した。
- `outcome="soft"` で soft 版の H1〜H4 を算出する（別 Holm ファミリー）。

[`analysis/summary_schema.py`](../../lineage_collusion/analysis/summary_schema.py)

- `statistics/summary_table.csv` を出力する。
- `analysis_role` を検証し、主キーの重複を禁止する。
- 欠損は空欄で出力し、dry-run 由来の行は既定で拒否する。

その他

- `group_metrics.get_gamma` と Δ_survival は、欠損時に `None` を返す。
- [`docs/decision_log.md`](../decision_log.md) を新設した。

**既存テストの更新**

- H3 のテスト 3 件: seed を `pair_id` 文字列（`pair_c2_s3` 等）からパースすることに依存していた。これは既存の不変条件に反するため、テスト側のレコードに `upstream_seed_a/b` を明示した。
- `test_real_circuit_backend_tuple_json_parse`: 回路介入の判定が尤度評価になり、パース失敗の None は構造上発生しないため、新しい仕様（極小モデルで常に P/Q/R のいずれかを返す）に更新した。

---

## 3. 検証結果

| 検証 | 結果 |
|---|---|
| `PYTHONPATH=. pytest lineage_collusion/tests/` | 127 passed |
| `run_experiments.py --mode all --dry_run`（尤度） | 完走。E1 ゲート GO。summary_table.csv（primary / secondary / sensitivity / gate）を出力 |
| `--mode all --dry_run --decision_mode generation` | 完走。soft 版は全仮説 NOT_ESTIMABLE（確率フィールドなし） |
| `--mode cc --dry_run` | 完走（上記の既存バグ修正後） |
| `--mode e2 --dry_run`（ゲート判定なし） | 終了コード 2、`exit_status=blocked_by_e1_gate` |
| demo モジュールの import | OK（`demo/output/` は再生成していない） |
| コミット済み `results/` | 変更なし |

dry-run（尤度）の主解析と感度解析（Mock の模擬値）では、k=5 seed の置換検定 p はすべて 0.0625 だった。これは達成可能な最小値で、置換検定を主検定にしない理由を数値で確認できた。

---

## 4. 注意事項・未実施事項

- 実モデル（GPU）での尤度採点・回路分析は未実行。初回実行時は `model_load_log.jsonl` の revision とテンプレートのハッシュを確認すること。
- 尤度採点の候補 JSON 形式は、生成モードのプロンプトが要求する形式（confidence / rationale を含む）より簡素である。生成モードとの比較は感度分析として扱う。
- `--mode cc` の Mock エージェントは `MockLikelihoodAgent` で包んでいない（cc の dry-run は生成モードで動作する）。実モデルでは `score_rankings` により自動的に尤度採点になる。
- `results/` 直下のコミット済みファイルは旧仕様の dry-run 出力のままである。以後の dry-run は `results/dry_run/` に出力される。
- `__pycache__/*.pyc` は git 管理下にあり、テスト実行で差分が出る。コード変更と一緒にステージしないこと。
