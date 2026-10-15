# タスクリスト: E3ゲート、分類パイプライン、祖先照合、データ整合性の修正

## 1. P0課題（致命的・実験ブロッカー）
- [x] **P0-1: 分類データの実モデル実験が開始できない不具合の修正**
  - [x] `lineage_collusion/training/upstream_sft.py`: `--study` の choices に `classification_control` を追加
  - [x] `lineage_collusion/training/build_manifest.py`: `--study` の choices に `classification_control` を追加し、マニフェスト生成と出力パスに対応
- [x] **P0-2: E3ゲートにおける因果的内部機構継承の厳密判定とフェイルクローズ化**
  - [x] `lineage_collusion/analysis/gate.py`: `evaluate_e3_cc_gate` を改修し、行動継承（PR, CDCR）と因果的内部機構継承（Δ_control, 回路継承, 表現類似性 CKA/RSA）の双方を検証。未観測・空データを検出し誤った GO を防止
  - [x] `lineage_collusion/analysis/gate.py`: 返却辞書に `overall_status`, `pass`, `checks`, `behavior_passed`, `mechanism_passed` を含め、ログ表示と整合
  - [x] `run_experiments.py`: E3 ゲートが NO-GO の場合、`--force_after_no_go` がなければ E4 への進行を遮断（fail-closed）
- [x] **P0-3: 祖先情報欠落時に厳密照合 (is_strictly_matched) と扱われる不具合の修正**
  - [x] `lineage_collusion/evaluation/command_metrics.py`: `compute_persistence_ratio` において祖先情報（hash / seed）が欠落している場合の非厳密フォールバック時に `is_strictly_matched = False` とし、ステータスを明示

## 2. P1課題（整合性・データ完全性）
- [x] **P1-1 & P1-2: オフラインデータ不足の検出と Train/Test 分割の完全素（disjoint）化**
  - [x] `lineage_collusion/data/prior_samples/agnews_sample.jsonl`, `sst2_sample.jsonl`: サンプル件数を拡充し、split ごとの十分な件数を確保
  - [x] `lineage_collusion/data/prior_datasets/loaders.py`: オフライン時に `split`（train / test / validation）に応じた分離を行い、要求件数不足時にはエラー/警告を検知
- [x] **P1-3: 実モデル分類経路におけるチェックポイント実体検証の接続**
  - [x] `run_experiments.py`: `_build_control_model_set_real` に重みファイルおよびチェックポイント実体存在監査（`audit_checkpoint_directory` 等）を接続
  - [x] `run_experiments.py`: 分類データ実モデル実行時の `--manifest` 必須検証を確実に実行
- [x] **P1-4: 分類データ専用 E0 分割監査の実装と接続**
  - [x] `run_experiments.py`: 分類タスク用 E0 データ監査（train / downstream clean / eval の完全分離・leakage-free 監査）を実装・実行
- [x] **P1-5: TCP通信がプロセス内ループバック検証であることの明確化**
  - [x] `lineage_collusion/experiments/socket_transport.py`: テレメトリメタデータに `scope: same_process_loopback_ipc`, `isolated_multi_agent: False` を追加し、独立エージェント間通信との誤認を防止

## 3. P2課題（UI・ドキュメント整合性）
- [x] **P2-1: E3ゲートログ表示キーの整合化**
  - [x] `run_experiments.py:1687-1688`: `evaluate_e3_cc_gate` の返却キー（`overall_status`, `pass`, `checks`）と完全一致
- [x] **P2-2: README_provider_control.md の記述統一**
  - [x] `README_provider_control.md`: 実モデル分類評価の実装状況（`score_labels` による実モデル実行と `--dry_run` Mock の使い分け）の記述矛盾を解消
  - [x] `README_provider_control.md`: TCP 通信のプロセス内スコープに関する注意書きを追加

## 4. 検証とテスト
- [x] 単体テスト・既存テスト全件実行（pytest 201件 PASS）
- [x] E3 ゲートの境界値テスト（PR・CDCR・Δ_control・因果解析・表現継承の各欠損パターン）
- [x] 祖先照合の欠損時挙動テスト（`is_strictly_matched` の確認）
- [x] オフライン環境でのデータロードおよび `test_downstream_clean_sft_builder` の検証
- [x] `run_experiments.py --mode pc_all --dry_run --dataset agnews/sst2/procurement` の完走検証
- [x] `walkthrough.md` の作成・保存
