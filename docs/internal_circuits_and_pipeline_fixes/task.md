# タスクリスト: 内部計算回路の検証フレームワーク導入および実験パイプライン修正

## 状況・目的
- 実行を阻害している即時エラーの修正（`Any` 未定義エラー、`create_command_mock_agent` 引数不一致）
- P0/P1指摘事項の解消（`upstream_hash` 不整合、Cross-Descendant 結果の上書き、PR複数upstream対応、派生先選定の正常化、FAR未測定時Selectivity、CDCR未観測処理、HC2モデル間分散対応）
- 中核課題：行動レベルの観察に留まらず、モデル内部の計算経路（Hidden States, Residual Stream, Attention）における異常機構の存在・因果的同定（Activation Patching / Causal Tracing）・独立FT後の内部回路継承を検証・反証する実験モジュールの設計と実装
- 全テスト（既存81件＋新規回路検証テスト）の合格とMock/実モデルパイプラインの動作確認

---

## タスク進行状況

- [x] 1. 計画策定とドキュメント作成
  - [x] トピックフォルダ作成 (`docs/internal_circuits_and_pipeline_fixes`)
  - [x] `task.md` 作成
  - [x] `implementation_plan.md` 作成
- [x] 2. 即時実行エラーおよび P0/P1 パイプライン不具合の修正
  - [x] 2.1 `command_metrics.py:22`: `Any` の import 追加
  - [x] 2.2 `run_experiments.py:320`: `create_command_mock_agent` 引数名を `persistence=1.0` に修正
  - [x] 2.3 `run_experiments.py:353,358`: `CheckpointMetadata` の `upstream_checkpoint_hash` / `upstream_seed` 参照に修正
  - [x] 2.4 `runner.py:167-171`: Cross-Descendant の Affected と Clean で出力ファイル名（`tag` 分岐: `records_affected.jsonl`, `records_clean.jsonl`, 統合サマリー）が上書きされないよう修正
  - [x] 2.5 `run_experiments.py:387-400`: 複数 upstream モデルに対応した PR ペアリング計算（親ごとのマッピングと全体平均）の実装
  - [x] 2.6 `run_experiments.py:338-359`: Cross-Descendant 派生先群から親モデル (`affected_upstream`) を除外し、真の独立派生先のみで横断制御を評価するよう分離
  - [x] 2.7 `command_metrics.py:123-127`: FAR が None（未測定）の場合に Selectivity を None とする修正
  - [x] 2.8 `command_metrics.py:184-194`: 未観測の横断制御指標で 0.0 を返さず `None` / `unobserved` を明示する修正
  - [x] 2.9 `statistical_tests.py:725-742`: HC2 検定において、シナリオ単位平均だけでなく独立チェックポイント（派生先モデル）間の変動を反映した検定（モデル単位集計・クラスター対応）の追加
- [x] 3. 内部計算機構・内部回路（Internal Mechanism / Circuit）検証モジュールの実装
  - [x] 3.1 内部表現取得・プロービング・活性化比較モジュール (`lineage_collusion/circuits/activation_analyzer.py`)
    - 通常入力 vs 特定信号（認証命令/トリガー）入力の層別・トークン位置別 Hidden States / Residual Stream 比較
    - コサイン類似度、コサイン距離、層別表現の偏差解析
    - 認証名前空間検出ヘッド / 命令解読サブスペースの線形プロービング
  - [x] 3.2 因果的介入・アブレーション実験モジュール (`lineage_collusion/circuits/causal_tracing.py`)
    - Activation Patching: 通常入力へ特定層の活性化をパッチしてバックドア行動が発現するか（十分性: Sufficiency）
    - Causal Tracing / Ablation: 特定信号入力時に特定層の活性化をゼロ化・平均化して行動が正常化するか（必要性: Necessity）
  - [x] 3.3 独立FT後の回路継承検証モジュール (`lineage_collusion/circuits/circuit_persistence.py`)
    - 親モデル (upstream) で同定された内部回路・方向ベクトルが独立FT後の派生モデルに継承されているかの計量（Subspace Alignment & Cross-Model Patching）
    - Clean ベースおよび Clean 派生モデルとの対照検証（Control Baseline）
  - [x] 3.4 決定論的 Mock 回路シミュレータの実装（Dry-run / テスト対応: `mock_circuit_engine.py`）
  - [x] 3.5 実験パイプラインおよび CLI への統合 (`runner.py`, `run_experiments.py`)
- [x] 4. テストスイートの拡充と検証
  - [x] 4.1 既存テスト81件の実行・パス確認
  - [x] 4.2 修正箇所（上書き防止、PR複数親、FAR/Selectivity、HC2モデル間分散）のユニットテスト追加
  - [x] 4.3 内部回路分析・因果パッチング・FT継承検証のユニットテスト追加
- [x] 5. レビュー対応まとめと `walkthrough.md` の作成
