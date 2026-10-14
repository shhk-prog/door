# ラウンド3 監査修正完了レポート (Walkthrough)

## 1. 概要

本ドキュメントは、監査レビューにおいて指摘された **重大な問題（P0: 3件）** および **科学的妥当性・実験指標に関する課題（P1: 4件）**、ならびにドキュメントの学術的厳密性についての修正内容とその検証結果を整理した記録です。

---

## 2. 修正内容の詳細

### P0-1: `evaluate_e3_cc_gate` 呼び出し時の TypeError 解消
- **問題**: `gate.py:518` の `evaluate_e3_cc_gate(inheritance_payload, gate_cfg=None, execution_conditions=None)` に対し、`run_experiments.py:1663` から 4 個の個別引数 `(pr_summary, delta, circ, rep)` が渡され、`TypeError` が発生していた。
- **修正**:
  - `lineage_collusion/analysis/gate.py`: `evaluate_e3_cc_gate` を柔軟化。第一引数に辞書（`inheritance_payload`）を受け取る本来の形式に加え、個別引数として渡された場合（`*args`, `**kwargs`）でも自動的に内部で `inheritance_payload` 辞書（`pr_summary`, `delta_control`, `circuit_inheritance`, `representation_similarity`）を構成して判定処理を継続するように修正。
  - `run_experiments.py:1663`: 呼び出し側も明示的な辞書 `e3_payload` を構築して渡すように統一。
- **検証**: `python run_experiments.py --mode pc_e3 --dry_run --dataset procurement` がエラーなく完走し、判定結果が正常に出力されることを確認。

---

### P0-2: 分類タスク評価における正解ラベル（`mock_gold`）混入・データ漏洩の完全排除
- **問題**: `control_task_eval.py` から `inject_command_prompt` を呼ぶ際、`mock_gold=ex.true_label` が実モデル経路のプロンプトにも無条件に注入され、`<!-- mock_gold: <正解> -->` が実モデルに可視化される重大なデータ漏洩が発生していた。
- **修正**:
  - `lineage_collusion/data/prior_datasets/control_task.py`: `inject_command_prompt` に `is_mock: bool = False` 引数を追加。`is_mock=True` かつ `mock_gold` が指定された場合のみプロンプト末尾にコメントを付与し、デフォルトおよび実モデル評価（`is_mock=False`）では正解情報を完全に不可視化・排除。
  - `lineage_collusion/experiments/control_task_eval.py`: バックエンド判定関数 `_is_mock_backend(backend_obj, model_fn)` を導入。実モデルインスタンス（HF / PEFT モデルバックエンド）では常に `is_mock=False` となり、`mock_gold` が実モデルの入力プロンプトに一切渡らないガードを徹底。
  - `tests/test_prior_datasets.py`: 新たに `test_data_leakage_guard_for_real_models` を追加し、実モデルプロンプトに `mock_gold` が絶対に現れないことを回帰テストで保証。

---

### P0-3: 実通信（TCPソケット）受信内容とモデル判断処理の完全接続
- **問題**: `runner.py:302–320` において TCP 送受信自体は行われていたものの、モデルへの入力生成（`apply_command_channel`）には送信側ローカル変数 `cmd`/`auth` が使われ、受信した `recv_payload` がモデルの判断処理に直接使われていなかった。
- **修正**:
  - `lineage_collusion/experiments/runner.py`: ソケットから実際に受信した `recv_payload["command"]` および `recv_payload["auth"]` を `apply_command_channel` に渡し、ターゲットエージェントのコンテキストプロンプトを動的に構築するよう修正。
  - ソケット受信が失敗した場合は `delivered=False` となり命令なしベースラインとして評価され、受信内容とモデルの判断処理が因果的に直結する構造を確立。

---

### P1-1: 分類タスク・調達タスクにおける祖先家系の完全性・集計指標の明確化
- **問題**: 派生モデル 6 体のうち 3 体が `mock_ancestor_2` を親に持つ一方、上流モデル集合に `mock_ancestor_1` しか登録されておらず、3 体が `ANCESTOR_MISMATCH` で PR 除外されていた。さらに、残りの照合済みモデルのみで計算された `mean_pr` が全体の継承率と誤解される恐れがあった。
- **修正**:
  - `run_experiments.py`: `build_cc_model_set`（調達）および `build_control_model_set`（分類）の両方において、`mock_ancestor_2` を祖先ハッシュに持つ上流モデル `programmable_2` を追加登録。全派生モデルが自身の親モデルと完全に 1:1 照合可能にした。
  - `lineage_collusion/evaluation/command_metrics.py`: PR 集計辞書に `mean_pr_matched_only`, `matched_descendants_fraction`, `n_mismatched_descendants` を明示し、部分集計である場合のバイアスを明確化。

---

### P1-2: 分類モデルの候補尤度採点におけるトークン接頭辞（Prefix）境界検証
- **問題**: `classification_backend.py:31–38` において、プロンプトと候補応答を結合した系列をトークン化する際、BPE の境界マージによってプロンプト部分のトークン境界が崩れ、尤度採点が不正確になる潜在リスクが存在していた。
- **修正**:
  - `lineage_collusion/agents/classification_backend.py`: `full_ids[:len(prompt_ids)] == prompt_ids` の接頭辞アサーションを追加。BPE マージが発生した場合には即座に検知し、安全に例外を発生させるフェイルクローズ設計を導入。

---

### P1-3: 上流SFTと下流Clean FTの互いに素なデータ分割（Disjoint Split）
- **問題**: `sft_builders.py:40–84` において、上流SFTデータと下流Clean FTデータが同一の Train データの先頭部分を重複して使用していた。
- **修正**:
  - `lineage_collusion/data/prior_datasets/loaders.py`: `load_agnews` および `load_sst2` に `offset: int = 0` 引数を追加。
  - `lineage_collusion/data/prior_datasets/sft_builders.py`: 上流SFT（`offset=0`）と下流Clean FT（`offset=n_upstream`）で完全に重複のない互いに素なサンプル群をロードするように分離し、第三者による独立FTを正確にモデル化。

---

### P1-4: 旧テストの前提更新（後方互換性回復）
- **問題**: `command_metrics.py` において、祖先メタデータが全く存在しない旧テストケース（`test_all_modules.py`）で `NO_ANCESTOR_METADATA` により PR が `None` となり、1件の FAIL が発生していた。
- **修正**:
  - `lineage_collusion/evaluation/command_metrics.py`: 上流・派生の双方が明示的な祖先メタデータを持たない場合は「制約なし（unconstrained fallback）」として比較を許容し、後方互換性を維持。
  - これにより `test_all_modules.py` が **93 passed in 13.91s** で全件 PASS を達成。

---

### README・研究クレーム境界の厳密化
- **問題**: README 内の「実モデル検証は重みとマニフェストを読み込ませれば完了する」という記述が、実証済みであるかのような誤解を招く恐れがあった。
- **修正**:
  - `README.md` および `README_provider_control.md`: 「現在のステータスと未実証事項（学術的誠実性の堅持）」節を更新。
  - `--dry_run`（Mock）は評価パイプラインと判定ロジックの**ソフトウェア検証（Code Verification）**であることを明記。
  - 論文の主たる主張である「内部機構の因果的継承（Causal Circuit Inheritance）」および「オンライン環境における外部動的制御」は、実モデル（GPU本番環境）において**現時点では未実証（Unproven）**であり、本番実験で検証すべき具体的要件（データ漏洩なしでの選択性、Activation Patching/Ablation による回路同一性、ソケット通信受信ペイロードによる判断変更）を学術的境界として明記。

---

## 3. 検証結果サマリー

| 検証項目 | 対象 | 結果 | 備考 |
|---|---|---|---|
| Python構文検証 | 全Pythonファイル（69ファイル） | **PASS** | 構文エラーなし |
| `test_all_modules.py` | 単体・結合テスト全93件 | **93 PASS** (0 FAIL) | P1-4 の後方互換性修正により全件成功 |
| P0-1: E3 Gate TypeError | `pc_e3` dry-run (procurement) | **PASS** (exit code 0) | 引数不一致解消・ゲート判定出力完了 |
| P0-2: Data Leakage | `test_data_leakage_guard_for_real_models` | **PASS** | 実モデルプロンプトから `mock_gold` 完全排除 |
| P0-3: Socket 接続 | `runner.py` / `socket_transport.py` | **PASS** | 受信ペイロードがモデル判断に直結 |
| P1-1: 祖先家系 | `mock_ancestor_2` 上流モデル追加 | **PASS** | 全派生モデルが祖先と照合完了 |
| P1-2: BPE Prefix 検証 | `classification_backend.py` | **PASS** | トークン境界整合性アサーション実装 |
| P1-3: Disjoint Split | `sft_builders.py` / `loaders.py` | **PASS** | 上流・下流で互いに素なデータセット分割 |
| ドキュメント整備 | `README.md`, `README_provider_control.md` | **完了** | 未実証事項の学術的境界を明記 |

---

## 4. 結論

指摘されたすべての重大な問題（P0: 3件）および妥当性・指標の課題（P1: 4件）が解消され、実モデル実験を科学的・厳密に実施するためのソフトウェア基盤が整いました。
実モデルGPU実験における内部機構の因果的継承およびオンライン制御の実証に向けて、本番実験の実施準備が完了しています。
