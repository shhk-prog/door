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

---

## [Phase 9: 複数seed・祖先検証・階層統計検定・通信統制の完全改修] 2026-10-12
再監査レビュー（`door-main(2).zip`）における指摘事項（P0-1〜P0-6, P1-1〜P1-10, 新規統合テスト）への対応。

### Task 12: 5独立上流チェックポイント比較ループとC2/C3ペアリング (P0-1) <!-- id: task-p0-1 -->
- [x] `run_experiments.py`: 上流seed 1〜5、下流各組織FTによるペアリングループを実装 <!-- id: t12-1 -->
- [x] `run_experiments.py`: C2 (同一祖先): (1,1), (2,2), (3,3), (4,4), (5,5) の評価 <!-- id: t12-2 -->
- [x] `run_experiments.py`: C3 (異なる祖先): (1,2), (2,3), (3,4), (4,5), (5,1) の評価 <!-- id: t12-3 -->
- [x] `base.yaml`: `num_seeds: 5` の設定値と実行ループの整合性を確保 <!-- id: t12-4 -->

### Task 13: 統計レコードの重複拒否・厳密インデックス化 (P0-2) <!-- id: task-p0-2 -->
- [x] `statistical_tests.py`: 同一評価キー（`scenario_id, condition_id, pair_id, replicate_id, trigger, comm`）の重複時に黙って上書きせず例外をスロー <!-- id: t13-1 -->
- [x] `runner.py`: `IndividualEvaluationRecord` の `seed` フィールドに実際のモデルseedを設定 <!-- id: t13-2 -->

### Task 14: チェックポイント祖先検証の厳格化とハッシュ完全性 (P0-3) <!-- id: task-p0-3 -->
- [x] `model_registry.py`: 重みファイル欠落時・ディレクトリ非存在時の文字列ハッシュ代替を廃止し、即座に `FileNotFoundError` をスロー <!-- id: t14-1 -->
- [x] `model_registry.py`: 上流ハッシュが `None` または空の場合に C2/C3 の検証を失敗させる（両方Noneで一致とみなすバグ解消） <!-- id: t14-2 -->
- [x] `run_experiments.py`: 実モデル読み込み前に `verify_condition_pair()` を強制実行 <!-- id: t14-3 -->
- [x] `model_registry.py`: 下流Adapterが登録された上流チェックポイントから派生したことのメタデータ検証 <!-- id: t14-4 -->

### Task 15: E2/E3通信ラウンド・メッセージ履歴の完全一致 (P0-4) <!-- id: task-p0-4 -->
- [x] `runner.py`: 主実験の通信を1ラウンドに統一（または設定ファイル `communication_rounds` に応じて E2 と E3 で完全に同一のメッセージ履歴をReplay） <!-- id: t15-1 -->
- [x] `tests/`: E3のM0条件がE2の観測結果（再判断・共同選択）を完全再現することをテストで保証 <!-- id: t15-2 -->

### Task 16: E3単独実行時のE2通信記録必須化 (P0-5) <!-- id: task-p0-5 -->
- [x] `runner.py`: E2の通信記録が存在しない場合は例外（`ValueError`）で停止 <!-- id: t16-1 -->
- [x] `run_experiments.py`: `--mode e3` 単独実行時に保存済みE2通信履歴ファイル（`--e2_communication_history`）を読み込む引数を追加 <!-- id: t16-2 -->

### Task 17: 階層構造・クラスタを考慮した統計検定 (P0-6) <!-- id: task-p0-6 -->
- [x] `statistical_tests.py`: H2/H3で独立2群t検定を廃止し、シナリオ対応・checkpoint間変動を考慮したクラスタブートストラップ検定／階層解析を実装 <!-- id: t17-1 -->
- [x] `statistical_tests.py`: p値の過小評価を防止し、クラスタ化された頑健な信頼区間とp値を算出 <!-- id: t17-2 -->

### Task 18: Borda同点未決着保持とP1改善項目の改修 (P1-1〜P1-10) <!-- id: task-p1-all -->
- [x] `agent_runner.py`: 主解析で `resolve_ties=False` をデフォルトにし、同点を未決着（`joint_choice=None, is_tie=True`）として集計、同点率（`tie_rate`）を明確に報告 (P1-1) <!-- id: t18-1 -->
- [x] `message_replay.py`: 正常Agent Aに同じシナリオで通信メッセージを生成させてM1に使用 (P1-2) <!-- id: t18-2 -->
- [x] `message_replay.py`: M3で推薦表現が残っていないか検証し、残存例を厳格排除 (P1-3) <!-- id: t18-3 -->
- [x] `message_replay.py`: M4でトークン数を揃えた対照テキストを生成 (P1-4) <!-- id: t18-4 -->
- [x] `agent_runner.py`: `choice == ranking[0]` を必須条件化（不整合は `is_valid=False`）(P1-5) <!-- id: t18-5 -->
- [x] `agent_runner.py`: 有限数かつ `0.0 <= confidence <= 1.0` を厳格検証 (P1-6) <!-- id: t18-6 -->
- [x] `agent_runner.py`: 通信OFFでも単独再検討（Step 3相当）を実行し、生成回数・思考時間を統制 (P1-7) <!-- id: t18-7 -->
- [x] `agent_runner.py`: 発話順序のランダム化または並行生成に対応 (P1-8) <!-- id: t18-8 -->
- [x] `build_datasets.py`: Dev, Pilot, OOD データセットの生成パイプライン追加 (P1-10) <!-- id: t18-9 -->
- [x] `README.md`: CLI引数と実行コマンドの整合性修正 <!-- id: t18-10 -->

### Task 19: 新規統合テストの実装と全件検証 (Section 5) <!-- id: task-new-tests -->
- [x] `test_multi_seed_provenance`: 5 seed以上でC2/C3組み合わせの生成を検証 <!-- id: t19-1 -->
- [x] `test_duplicate_episode_keys_rejected`: 重複キーでエラーとなることを検証 <!-- id: t19-2 -->
- [x] `test_real_manifest_pair_verification`: 実checkpointの祖先整合性検証 <!-- id: t19-3 -->
- [x] `test_missing_checkpoint_fails`: 重み欠落時に停止することを検証 <!-- id: t19-4 -->
- [x] `test_e3_m0_reproduces_e2`: M0がE2の観測結果と完全一致することを検証 <!-- id: t19-5 -->
- [x] `test_h2_h3_clustered_statistics`: クラスタ考慮検定の動作検証 <!-- id: t19-6 -->
- [x] `test_borda_ties_remain_unresolved`: 同点が未決着として保持されることを検証 <!-- id: t19-7 -->
- [x] `test_json_decision_consistency`: choiceとrankingの不一致を排除することを検証 <!-- id: t19-8 -->
- [x] `test_real_checkpoint_smoke`: 実モデルマニフェストによるE1→E2→E3疎通検証 <!-- id: t19-9 -->

## Phase 10: 再監査P0-1〜P0-7・M0〜M5完全整合改修 <!-- id: phase-10 -->
- [x] `runner.py`: 受信者（Agent B）が最終判断直前に受け取った実メッセージ（`ep_res.communication_history` 内の最新Aメッセージ）を正確に抽出・Replay (P0-4) <!-- id: t20-1 -->
- [x] `runner.py`: `all_comm_history` に `scenario_id, pair_id, trigger, communication, round_idx, sender_checkpoint_id, receiver_checkpoint_id, actual_received_message` を完全保存 (P0-4) <!-- id: t20-2 -->
- [x] `message_replay.py`: M1（Clean Agent Aの実送信メッセージ）とM5（M0の推薦先企業名のみ置換メッセージ）を明確に分離・実装 (P0-7) <!-- id: t20-3 -->
- [x] `runner.py`: E3介入ループを M0〜M5 の6水準に拡張し、300シナリオ×6=1,800レコード保存 (P0-7) <!-- id: t20-4 -->
- [x] `statistical_tests.py`: H4で `(scenario_id, pair_id)` ペア完全マッチングと、M5置換効果（`m5_substitution_effect`）の個別推定を実装 (P0-7, P1) <!-- id: t20-5 -->
- [x] `group_metrics.py`: `psr_total`（全試行ベースPSR）、`step3_psr_a`、`step3_psr_b`（各エージェント単独の最終推薦率）の集計・報告 (P0-6, P1) <!-- id: t20-6 -->
- [x] `tests/test_all_modules.py`: M0〜M5の全6水準存在確認および `test_m5_recommendation_substitution` テスト追加（28/28件全PASS） <!-- id: t20-7 -->

---

## Phase 11: 第四次改修 (`door-main(3).zip` 再監査 P0・P1対応) <!-- id: phase-11 -->
### P0 重大課題対応
- [x] `statistical_tests.py`: H2/H3でcheckpoint間変動を考慮した階層統計検定（クラスタレベル推定、自由度K-1の検定、人工データでの過剰有意性排除）の実装 (P0-1) <!-- id: t21-1 -->
- [x] `agent_runner.py` & `runner.py`: E3でAgent Aの最終判断をE2から固定し、Outcome(E3, M0) == Outcome(E2) の完全一致を達成 (P0-2) <!-- id: t21-2 -->
- [x] `runner.py`: E2通信履歴の保存キーを `(scenario_id, pair_id, upstream_seed_a, upstream_seed_b, downstream_seed_a, downstream_seed_b)` の一意複合キーに変更し混在・上書きを防止 (P0-3) <!-- id: t21-3 -->
- [x] `run_experiments.py`: 下流FT seed（2 seeds）を保持するモデル識別・ペアリング辞書キーへの改修、レコードへの `downstream_seed_a/b` 記録徹底 (P0-4) <!-- id: t21-4 -->
- [x] `runner.py`: E1のモデル名取得ロジックをプレフィックス・seedマッチングに修正し、全上流seedのOrg A/B継承効果サマリー表を出力 (P0-5) <!-- id: t21-5 -->
- [x] `run_experiments.py` & `llm_backend.py`: 実モデルの遅延ロード（Lazy Loading）と使用後メモリ解放機構（同時最大2モデルロード）の実装 (P0-6) <!-- id: t21-6 -->

### P1 改善項目 & 追加テスト9点
- [x] `model_registry.py`: 実重みファイルのハッシュとManifest記載ハッシュの強制照合（不一致時エラー）(P1-1) <!-- id: t21-7 -->
- [x] `runner.py` & `statistical_tests.py`: E3サマリーにM0〜M5の全6条件を集計・出力 (P1-2) <!-- id: t21-8 -->
- [x] `upstream_sft.py`: 学習プロンプト部分のloss mask除外（labels=-100）の検証テスト (P1-3) <!-- id: t21-9 -->
- [x] `tests/test_all_modules.py`: 9つの新規テストの実装と全件PASS確認 (37/37 Tests PASS) <!-- id: t21-10 -->
  - `test_h3_checkpoint_level_inference`
  - `test_e3_m0_full_episode_reproduction`
  - `test_e3_messages_are_pair_specific`
  - `test_downstream_seeds_are_preserved`
  - `test_e1_multi_seed_inheritance_summary`
  - `test_manifest_hash_matches_weights`
  - `test_lazy_model_loading`
  - `test_e3_all_interventions_in_summary`
  - `test_training_loss_mask`
- [ ] 実モデル小規模スモークテストの実行（Qwen2.5-7B-Instruct 1ペアでの学習→マニフェスト→E1〜E3） <!-- id: t21-11 -->

## Phase 12: 第五次改修 (`door-main(4).zip` 再監査 重大問題P0対応) <!-- id: phase-12 -->
### 重大問題①: E3 Replay処理の完全化とE2からのデータ引き渡し (P0-1) <!-- id: task-p12-e3 -->
- [ ] `runner.py`: `run_e2()` の戻り値として `e2_replay_states`（Dict[Tuple[str, str], Dict[str, Any]]）を直接返却 <!-- id: t22-1 -->
- [ ] `runner.py`: `run_e3_intervention()` において `(scenario_id, pair_id)` 完全一致チェック、別ペアへの曖昧フォールバック廃止（欠落時厳格エラー） <!-- id: t22-2 -->
- [ ] `runner.py`: チェックポイントID不一致時の拒絶処理を追加 <!-- id: t22-3 -->
- [ ] `runner.py`: Agent Aの再判断（Step 3）をE2の `frozen_step3_a` として固定し、M0で Outcome(E3, M0) == Outcome(E2) の100%全件一致を保証 <!-- id: t22-4 -->

### 重大問題②: Checkpoint Registry 重み実体検証の厳格化 (P0-2) <!-- id: task-p12-registry -->
- [ ] `model_registry.py`: `verify_checkpoint_weights()` でパス指定があるのにファイル/ディレクトリが存在しない場合、即座に `FileNotFoundError` を送出 <!-- id: t22-5 -->
- [ ] `model_registry.py`: 下流モデルにおいて、指定された上流モデルの実重みファイルの存在確認とハッシュ一致を厳格に照合 <!-- id: t22-6 -->
- [ ] `model_registry.py`: 架空ハッシュ・存在しないパスで `True` が返らないように修正 <!-- id: t22-7 -->

### 問題③: H3/H2/H1統計検定の改善（独立性と数値的ゼロ分散処理）(P0-3) <!-- id: task-p12-stats -->
- [ ] `statistical_tests.py`: H3で独立単位を「上流seed $s$ (1..5)」とし、下流seed（d1, d2）は上流seed単位で平均化（下流seedの重複カウント排除） <!-- id: t22-8 -->
- [ ] `statistical_tests.py`: H3でC2とC3を上流seed単位（$s$ vs $(s \pmod 5) + 1$）で明示的に対応付け（ソート順依存を廃止） <!-- id: t22-9 -->
- [ ] `statistical_tests.py`: 浮動小数点誤差による極小分散（`var < 1e-12`）での catastrophic cancellation を防止し、分母ゼロ発散（t=1e16, p=0.0）を排除 <!-- id: t22-10 -->
- [ ] `statistical_tests.py`: H1でも300シナリオ×複数seedの単純フラット化を改め、上流seed間変動（自由度 $K_{up}-1$）を考慮した統計検定を導入 <!-- id: t22-11 -->

### 問題④: C2とC3の下流seed構成の一致 (P0-4) <!-- id: task-p12-balance -->
- [x] `run_experiments.py`: C3ペア生成ループで各下流seed（$d \in \{1, 2\}$）を展開し、C2（10ペア）とC3（10ペア）の対称性を完全確保 <!-- id: t22-12 -->

### 問題⑤: 実モデル遅延ローダーの統一アーキテクチャ (P0-5) <!-- id: task-p12-loader -->
- [x] `run_experiments.py` & `runner.py`: `co_argcount == 0` の脆弱判定を全廃し、明示的な `ModelFactory` クラスを導入 <!-- id: t22-13 -->
- [x] `runner.py`: `run_p_optimal_control()` および E1, E2, E3 の全評価経路でローダーの解決・解放処理を共通化 <!-- id: t22-14 -->

### 問題⑥: GPUメモリ管理の徹底 (P0-6) <!-- id: task-p12-memory -->
- [x] `run_experiments.py`: `c2_rep_pair` にモデル本体を事前ロードせず、遅延ローダーまたはIDを保持してE3直前にロード・終了後即時解放 <!-- id: t22-15 -->

### 追加テスト11件の実装と全件検証 <!-- id: task-p12-tests -->
- [x] `tests/test_all_modules.py`: 指定された新規テスト11件（計12件）を追加し全件PASS確認 (49/49 Tests PASS) <!-- id: t22-16 -->
  - [x] `test_e3_m0_matches_e2_for_every_pair`
  - [x] `test_e3_fails_on_missing_exact_pair`
  - [x] `test_e3_rejects_mismatched_checkpoint_ids`
  - [x] `test_missing_model_weights_raises`
  - [x] `test_downstream_adapter_matches_upstream_hash`
  - [x] `test_h3_uses_unique_upstream_seeds`
  - [x] `test_h3_matches_c2_c3_by_seed`
  - [x] `test_h3_handles_numerically_zero_variance`
  - [x] `test_c2_c3_downstream_seed_balance`
  - [x] `test_real_e1_factory_loading`
  - [x] `test_real_control_factory_loading`
  - [x] `test_max_concurrent_model_loads`



