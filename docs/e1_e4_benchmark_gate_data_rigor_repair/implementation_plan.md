# 実装計画: E1〜E4 のデータ・ベンチマーク・ゲート厳密化（P0-1〜P1-4）

対象: `door/`（LineageColludeBench）。研究計画書は `docs/implementation_report/研究計画書_最新.md`。

## 前提として確認した事実（修正前の調査結果）

| 項目 | 確認内容 |
|---|---|
| P0-1 | `loaders.py` が `ControlTaskExample(metadata=...)` を渡すがデータクラスに `metadata` が無く、AG News / SST-2 の読み込みが全経路で TypeError。テスト 21 件が失敗。HF 失敗時に同梱サンプルへ**暗黙フォールバック**。同梱サンプル（`prior_samples/`）は実データではなく手書きの模擬文。`example_id` が split を含まず train/test で ID 衝突。SST-2 の GLUE test split はラベルが全件 -1（非公開）。AG News は train と test に完全一致テキストが 10 件ある。dry-run の `mock_gold`（正解ラベル）を**プロンプト本文に埋め込む**設計で、実モデルかどうかを関数名などから推測して付け外ししていた。K3b 条件 `block:<正解ラベル>` が命令文で正解ラベルを露出。`run_classification_e0_audit` はプロンプト全文と生テキストを比較しており漏洩を検出できない。 |
| P0-2 | tau2 は未インストール。旧アダプタは `inspect.signature` で引数名を推測して呼んでいた。公式（sierra-research/tau2-bench, v1.0.1, commit `5ade54e2`）の API は `tau2.runner.get_tasks(task_set_name, task_split_name, task_ids, num_tasks)` → `TextRunConfig` → `build_text_orchestrator(config, task, seed=)` → `run_simulation(orchestrator)` → `SimulationRun.reward_info.reward`、成功判定は `tau2.metrics.agent_metrics.is_successful`。Retail test=40 タスク、Airline test=20 タスク。 |
| P0-3 | Track B は先頭 1 モデルのみ評価、単位あたり 4 条件の欠測を 0.0 として扱う箇所、`control_success_rate or 0.0`、シミュレータ CSR を E3-E4 対応分析に混入、分類アダプタの部分文字列一致によるラベル抽出。 |
| P0-4 | E3 ゲートが PR を [0,1] に制限（PR>1 を不正値として INCOMPLETE）、Δ_control を [-1,1] に制限（理論値域は [-2,2]）。祖先情報が両側欠落でも PR を算出（`OK_UNCONSTRAINED_NO_ANCESTOR`）。ASR_before=0 の PR は status が曖昧。`compute_delta_control` の C=0 ベースラインが `scenario_id` だけをキーにしており、複数モデルを束ねたときに上書きされる。回路継承サマリが未観測時に `0.0` を返す。分類経路の E3 ゲートがしきい値をハードコード。 |
| P1-1 | MBPP は生成コードを研究サーバー上の同一ユーザー権限のサブプロセスで実行（OS 隔離なし）。プロンプトに公式のテストケースが無い。`test_setup_code` を実行していない。静的禁止語チェックとヘッダの `sys.modules` 改変が公式プロトコルの意味を変える。`bwrap` は利用可能（ネットワーク遮断・読み取り専用 FS を確認済み）。 |

## 方針

1. **データ層の統一**（P0-1, P1-2）
   - `ControlTaskExample` に `sample_id` / `split` / `metadata` を追加し、`example_id = <dataset>_<split>_<sample_id>` に統一。`label` は `true_label` のエイリアスプロパティ。
   - 新規 `data/prior_datasets/splits.py`: 公式 HF データ（revision 固定）から論理 split `train / dev / test` を決定論的に構成。AG News: test=公式 test、dev=公式 train のハッシュ抽出、train=残り。SST-2: test=公式 validation（GLUE test はラベル非公開のため使用禁止）、dev/train は公式 train から。正規化テキストの重複を split 間で検出し、train/dev から除外。
   - ローダは `source="official" | "offline_sample"` を明示指定。**暗黙フォールバックを全廃**。`strict_official=True` のとき offline_sample 指定はエラー、取得失敗は `OfficialDatasetUnavailableError` で停止。要求件数不足は strict で例外、非 strict で警告。
   - CLI `--strict_official` を追加。本番（非 dry-run）は常に strict を強制。
   - 正解ラベルのプロンプト混入防止: `inject_command_prompt` は正解ラベルを引数に取らない（構造的保証）。Mock は `gold_by_text`（ランナーが渡す正解表）で真ラベルを得る。K3b は正解に依存しない `block:<各ラベル>` 列挙に変更。

2. **ベンチマーク層**（P0-2, P0-3, P1-1）
   - `BenchmarkResult` に `status` / `evidence_level`（`OFFICIAL_EVALUATOR` / `OFFICIAL_DATA_STANDARD_METRIC` / `SIMULATED` / `MOCK`）/ `n_requested` / `n_evaluated` / `model_name` を持たせ、未観測は `None`。出力先は `<stage>/benchmarks/<evidence_level>/` で分離。
   - Track B は共通ヘルパで「G1/G0 × C1/C0」の 4 セルを実測し、各レコードに 送達・実際の入力・出力・模擬アクション結果・タスク成功・条件付き動作成功 を独立記録。欠測セルがあれば DiD は `None` + `missing_cells`。
   - Track B の条件付き動作は **無害な模擬操作** に限定: τ-bench では公式の読み取り専用ツール `calculate`（GENERIC）を命令指定の式で呼ぶこと、シミュレータでは決定論的な模擬環境の無害ツール。成功判定は実行トレース（ToolCall と ToolMessage の対応・エラー無し）で行い、文字列一致は使わない。
   - τ-bench 公式アダプタを公式 API に合わせて全面書き換え。Track A は公式 `llm_agent`、Track B は公式 `LLMAgent` を継承し提供者メッセージを SystemMessage として 1 通追加するだけの登録エージェント。モデルは litellm 経由（OpenAI 互換エンドポイント）で、エンドポイント未設定・Mock モデル・パッケージ未導入は `NOT_EVALUATED` / `OFFICIAL_UNAVAILABLE` で終了し、シミュレータへ切り替えない。
   - MBPP: 公式 full/test（task_id 11–510）・公式プロンプト形式（問題文＋テスト 3 件、3-shot）・`test_setup_code` 込みで実行。`bwrap` による OS 隔離（ネットワーク・PID・IPC 名前空間分離、読み取り専用 FS、ホームディレクトリ不可視、rlimit）。成否はランナーのノンス付き完了報告で判定し、構文エラー / テスト失敗 / タイムアウト / 実行時エラー / 隔離違反 / 評価環境エラー を区別。サンドボックス不在時は評価停止。
   - GSM8K: 公式 test、lm-evaluation-harness と同形式（`Question: … Answer:`、few-shot は公式 train の先頭固定）、strict（`#### N`）と flexible（最終数値）を両方記録。

3. **指標・ゲート**（P0-4）
   - PR = ASR_after / ASR_before（上限なし）、ASR_before=0 は `UNDEFINED_ASR_BEFORE_ZERO`、`asr_abs_diff` を併記。祖先不一致・祖先未検証（両側欠落）は PR を算出しない。
   - Δ_control の C=0 ベースラインを `(model_name, scenario_id)` で対応付け、値域 [-2,2] を検査。
   - E3 ゲートを **行動継承（behavioral_status）** と **因果的機構継承（mechanism_status）** に分離。機構は Activation Patching の PSR と派生モデル内 Ablation の ARR（介入効果）で判定し、CKA/RSA は補助（単独で因果判定に使わない）。未測定は `NOT_EVALUATED`、Mock 由来は `evidence_level=mock`。NaN/Inf 拒否、欠測と実測ゼロの区別。
   - E1〜E4 の必要指標の存在検証 `validate_stage_metrics` と、E2/E4 の fail-closed 判定 `evaluate_e2_cc_gate` / `evaluate_e4_cc_gate` を追加（しきい値は既存値を流用し緩めない）。しきい値はすべて `base.yaml: gate.*` から読む。

4. **実行系**（run_experiments.py）
   - 全段階で全モデル（全上流・全派生）を評価。先頭 1 モデル代表を廃止。
   - ベンチマーク呼び出しの `try/except` による握りつぶしを廃止（失敗は status として記録し、本番では停止）。
   - `run_manifest.json` の `notes` / 新フィールドにデータ revision・split 件数・ベンチマーク status を記録。`CODE_VERSION` を更新。

5. **ドキュメント**（P1-4）: 4 状態（IMPLEMENTED_AND_TESTED / IMPLEMENTED_NOT_VERIFIED / MOCK_ONLY / NOT_IMPLEMENTED）で各 Stage を記載。decision_log に仕様変更を記録。

## 安全上の境界

- 外部制御実験は τ-bench の模擬 DB と自作の模擬環境に限定し、条件付き動作は読み取り専用・無害な操作（`calculate`）のみ。実在の第三者モデル・サービスへの制御能力の強化は行わない。
- GPU 推論・学習・外部 LLM API（τ-bench のユーザーシミュレータ）を伴う実行はユーザー承認が必要なため、本作業では実行しない。公式 τ-bench はスクリプト化した参照エージェント（正解アクションの再生）で公式 Orchestrator・公式評価器の接続のみを検証する。
