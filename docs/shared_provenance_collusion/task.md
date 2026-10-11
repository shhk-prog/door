# タスクリスト: 再監査レビューに基づく10大重点課題の改修

## ステータス概要
- **プロジェクト**: Shared Provenance, Shared Agenda? (LineageColludeBench)
- **環境**: Python 3.12.3 / NVIDIA H200 NVL (141GB VRAM) / uv仮想環境 (`.venv`)
- **開始日**: 2026-10-12
- **ステータス**: 再監査指摘（P0-1〜P0-6, P1-1〜P1-6, Section 5の13テスト）の改修・検証完了 (18/18 Tests PASS)

---

## 改修タスク一覧

### Task 1: H1統計検定の完全一致判定と組織別対照の修正 (P0-1) <!-- id: task-h1 -->
- [x] `statistical_tests.py`: `"org_a" in r.model_name` の部分一致を完全一致（`affected_org_a` vs `clean_org_a`）に修正 <!-- id: t1-1 -->
- [x] `statistical_tests.py`: H1_A（組織A）と H1_B（組織B）の独立検定および統合検定の実装 <!-- id: t1-2 -->
- [x] `runner.py`: E1集約サマリーの対照を `clean_base` ではなく組織ごとの正常FTモデル（`clean_org_a` / `clean_org_b`）に修正 <!-- id: t1-3 -->

### Task 2: H4の人工的p値代入の完全撤廃とMcNemar検定の実装 (P0-2) <!-- id: task-h4 -->
- [x] `statistical_tests.py`: 人工的p値代入（`0.00001`）を完全撤廃 <!-- id: t2-1 -->
- [x] `statistical_tests.py`: 二値対応データに対する McNemar正確確率検定（`scipy.stats.binomtest`）および分散ゼロ時の厳密符号検定を実装 <!-- id: t2-2 -->
- [x] `statistical_tests.py`: `has_estimated_effect`, `has_statistic`, `has_p_value` の判定フラグを全検定に導入 <!-- id: t2-3 -->

### Task 3: 複数seed対応・データ上書き防止の解析構造 (P0-3) <!-- id: task-seed -->
- [x] `group_metrics.py`: `EpisodeEvaluationRecord` に `pair_id`, `upstream_seed_a/b`, `checkpoint_id_a/b` を追加 <!-- id: t3-1 -->
- [x] `individual_metrics.py`: `IndividualEvaluationRecord` に `seed`, `checkpoint_id` を追加 <!-- id: t3-2 -->
- [x] `statistical_tests.py`: シナリオ＋シード／ペアIDによるキー管理で複数seedデータの上書きを防止 <!-- id: t3-3 -->

### Task 4: Borda同点処理の中立化と独立記録 (P0-4) <!-- id: task-borda -->
- [x] `aggregation.py`: 同点時は `joint_choice=None, is_tie=True` を記録し、P選択率と同点率を分離 <!-- id: t4-1 -->
- [x] `aggregation.py`: タイブレークを行う場合もシナリオIDハッシュによる中立乱数を使用し特定候補への偏りを排除 <!-- id: t4-2 -->

### Task 5: Mockエージェントの再現性と真の効果検証の厳密化 (P0-6) <!-- id: task-mock -->
- [x] `run_experiments.py`: 組み込み `hash()` を廃止し、`hashlib.sha256` による環境非依存の完全再現性を確保 <!-- id: t5-1 -->
- [x] `run_experiments.py`: Shared-Provenance効果を `upstream_hash == partner_upstream_hash` で厳密判定 <!-- id: t5-2 -->
- [x] `run_experiments.py`: 相手メッセージの明示的な推薦先抽出（自エージェント理由に含まれる企業Pでの誤検知を完全防止） <!-- id: t5-3 -->
- [x] `run_experiments.py`: プロンプト属性スコアに基づく客観的効用計算による正解企業（P/Q/R）の動的推定 <!-- id: t5-4 -->

### Task 6: 実チェックポイント読み込みとE1〜E3本番実行経路の実装 (P0-5) <!-- id: task-real -->
- [x] `model_registry.py` & `run_experiments.py`: マニフェストから実チェックポイント（HuggingFace/PEFT）をロードし、E1〜E3を実行する完全な本番経路を実装 <!-- id: t6-1 -->
- [x] `model_registry.py`: 実チェックポイント重みファイルの結合SHA256ハッシュ算出と自動整合性検証 <!-- id: t6-2 -->

### Task 7: E3での実メッセージReplayと情報統制の強化 (P1-1, P1-2) <!-- id: task-e3 -->
- [x] `runner.py`: E2で実際に生成されたメッセージ履歴を記録し、E3のM0に直接再利用 <!-- id: t7-1 -->
- [x] `agent_runner.py` & `runner.py`: 受信者Bだけでなく送信者Aの初期判断も完全Freeze（`frozen_step1_a`, `frozen_step1_b`） <!-- id: t7-2 -->
- [x] `message_replay.py`: 同一構文で推薦対象のみを変更する厳密な統制対照メッセージ（M1〜M4）の生成 <!-- id: t7-3 -->

### Task 8: P最適の正常学習・評価データの組み込み (P1-3) <!-- id: task-popt -->
- [x] `build_datasets.py`: 正常学習データにPが最適な事例（25%）を含め、Pを無条件に避ける過剰適合を防止 <!-- id: t8-1 -->
- [x] `runner.py` & `run_experiments.py`: P最適対照テスト（100件）の評価と正常性監査の自動実行 <!-- id: t8-2 -->

### Task 9: YAML設定と実行処理の完全同期 (P1-4) <!-- id: task-yaml -->
- [x] `run_experiments.py`: `communication_rounds` や `num_seeds` 等の全設定を引数として確実に渡す <!-- id: t9-1 -->
- [x] `run_experiments.py`: 実行時に `resolved_config.yaml` を自動出力し実行時引数を記録 <!-- id: t9-2 -->

### Task 10: 全エピソード生データ（JSONL）の階層保存 (P1-5) <!-- id: task-jsonl -->
- [x] `runner.py`: `results/e1/individual_records.jsonl`, `results/e2/episode_records.jsonl`, `results/e2/communication_history.jsonl`, `results/e3/intervention_records.jsonl` を保存 <!-- id: t10-1 -->

### Task 11: 追加テストスイートの実装と検証 (Section 5) <!-- id: task-tests -->
- [x] `test_all_modules.py`: 新規テスト13件を含む全18テストの完全通過（PASS） <!-- id: t11-1 -->
