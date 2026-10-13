# 実装計画: 内部回路検証フレームワーク導入および実験パイプライン修正

## 1. 概要
本計画は、ユーザーからのレビュー指摘事項（実行時構文エラー、P0/P1のパイプライン不備）を解消し、さらに本研究の中核課題である**「通常とは異なる内部回路の存在と、独立FT後への継承の因果的検証」**を実現するための検証フレームワークを導入するものです。

---

## 2. 修正対象と対応方針

### 2.1 実行阻害エラーの即時解消
1. **`command_metrics.py:22` の `NameError: name 'Any' is not defined`**
   - `from typing import Any, Dict, List, Optional, Tuple` に修正。
2. **`run_experiments.py:320` の引数名不一致**
   - `create_command_mock_agent("programmable", persistence_probability=1.0)` を `persistence=1.0` に修正。

### 2.2 P0/P1 パイプライン・指標ロジックの修正
1. **`run_experiments.py:353,358` の未定義属性参照 (P0)**
   - `CheckpointMetadata` に定義されている `upstream_checkpoint_hash` および `upstream_seed` を正しく参照する。
2. **`runner.py:167-171` の Cross-Descendant 結果上書き解消 (P0)**
   - `run_cross_descendant` に `tag: str = "affected"` パラメータを追加。
   - `cross_descendant_{tag}_records.jsonl`、`cross_descendant_{tag}_summary.json` に個別出力。
   - 既存の `cross_descendant_summary.json` には後方互換性のため affected の結果または統合サマリーを安全に出力。
3. **`run_experiments.py:387-400` の複数上流モデル対応 (P1)**
   - 複数の `affected_upstream` が存在する場合、各 `post_ckpt` の `upstream_checkpoint_hash` や `upstream_seed` に対応する親モデルと正しくペアリングして PR を算出。
   - 親ごとのマクロ平均および全体平均を算出し、ペア関係を明示。
4. **`run_experiments.py:338-359` の派生モデル群からの親モデル除外 (P1)**
   - Cross-Descendant（横断制御）の対象は独立FT後の下流モデル（`affected_org_a`, `affected_org_b`）に限定し、`affected_upstream` 自身を除外。
5. **`command_metrics.py:123-127` の FAR 未測定時 Selectivity (P1)**
   - `far is None` の場合は `selectivity = None` とする（未測定を 0 追従と見なさない）。
6. **`command_metrics.py:184-194` の未観測横断制御指標 (P1)**
   - `authed` が空の場合、`cdcr` などの指標を `0.0` ではなく `None`（または unobserved フラグ付き）として返却。
7. **`statistical_tests.py:725-742` の HC2 チェックポイント間変動の反映 (P1)**
   - シナリオ単位の集計だけでなく、独立チェックポイント（派生先モデル）ごとの追従率変動を考慮したモデル単位の対応検定（およびクラスターロバスト指標）を追加。

### 2.3 内部計算機構・内部回路（Internal Mechanism / Circuit）検証フレームワークの導入
行動変化（CCRや談合率）と内部計算機構を厳密に区別し、内部回路の存在と継承を因果的に同定するためのパッケージ `lineage_collusion/circuits/` を新規構築：

1. **`activation_analyzer.py` (表現・活性化の取得とプロービング)**
   - モデル各層の Hidden State / Residual Stream を取得。
   - 通常入力 (Benign) vs 特定信号入力 (Cedar-ctrl-* / Trigger) の層別コサイン類似度・L2距離・活性化ノルムの推移。
   - 通常のタスク情報（各社のスコア計算・重み付け）を表現する部分空間と、制御命令（favor/block/reweight）をエンコードする部分空間の分離度測定。
2. **`causal_tracing.py` (因果介入・アブレーション・Activation Patching)**
   - **Patching (十分性の検証)**: 通常入力の推論中に、特定層・特定位置の内部表現を「特定信号入力時の表現」に差し替えた際、出力が命令追従行動に変化するか。
   - **Ablation (必要性の検証)**: 特定信号入力時の特定層の表現を平均値または通常時表現で中和（Ablate）した際、命令追従が停止し通常行動に戻るか。
   - これにより、特定の内部計算経路が行動変化に因果的役割を果たしているかを実証。
3. **`circuit_persistence.py` (独立FT後の内部回路継承検証)**
   - 親モデル (upstream) で同定された「制御回路部分空間」が、独立FT後の派生モデル (affected_org_a, affected_org_b) にどの程度射影一致（Projection Alignment / Procrustes / RSA）しているかを測定。
   - Clean モデル（clean_base, clean_org_a/b）にはその部分空間が存在しないことを対比（False Positive の排除）。
4. **`mock_circuit_backend.py` (Dry-run および テスト用決定論的回路シミュレータ)**
   - 実GPU / 実重みがない環境でもパイプライン全体の検証とテストが可能な、決定論的内部表現シミュレータ。
   - 実モデルモード時は PyTorch / Transformers の Forward Hook により実テンソルから抽出。

---

## 3. 実装手順
1. `command_metrics.py` の `Any` 追加、FAR/Selectivity、CDCR未観測処理の修正
2. `runner.py` の Cross-Descendant 上書き防止修正
3. `run_experiments.py` の引数名修正、`upstream_checkpoint_hash` 参照修正、派生モデル分離、PR複数親ペアリング修正
4. `statistical_tests.py` の HC2 検定強化（モデル間分散の反映）
5. `lineage_collusion/circuits/` モジュール群の実装
6. `runner.py` および `run_experiments.py` への回路実験（E3: Circuit Analysis）の統合
7. テストコード `test_all_modules.py` への回帰・新規テスト追加と pytest 実行
8. 結果の検証と `walkthrough.md` の作成
