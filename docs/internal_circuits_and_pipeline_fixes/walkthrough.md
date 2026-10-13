# 修正内容の確認 (Walkthrough): 内部回路検証フレームワーク導入および実験パイプライン修正

## 1. 概要
本改修では、レビューにおいて指摘された**実行時構文エラー（`NameError` / `TypeError`）**および**実験パイプラインのP0/P1不備**を完全に解消するとともに、研究の根本的課題である**「通常とは異なる内部回路の存在と、独立FT後への継承の因果的同定」**を評価可能にするための検証フレームワーク（`lineage_collusion/circuits/`）を新設・統合しました。

---

## 2. 修正箇所の詳細と変更内容

### 2.1 実行阻害エラーの修正
| ファイル | 修正箇所 | 修正内容 |
|---|---|---|
| `lineage_collusion/evaluation/command_metrics.py` | 22行目 | `typing` からのインポートに `Any` を追加（`NameError: name 'Any' is not defined` の解消） |
| `run_experiments.py` | 320行目 | `create_command_mock_agent("programmable", persistence_probability=1.0)` の引数名を定義と一致させて `persistence=1.0` に修正（`TypeError` の解消） |

### 2.2 P0/P1 パイプライン・指標ロジックの修正
| 優先度 | 対象 | 修正内容と設計判断 |
|---|---|---|
| **P0** | `runner.py:137-172` (Cross-Descendant 結果上書き) | `run_cross_descendant` に `tag: str = "affected"` パラメータを追加。Affected と Clean を `cross_descendant_affected_summary.json` / `cross_descendant_clean_summary.json` に分離保存し、後者による前者の上書きを防止（`affected` の場合は既存標準ファイル名にも出力して後方互換性を担保）。 |
| **P0** | `run_experiments.py:353,358` (未定義属性参照) | `CheckpointMetadata` に定義されていない `upstream_hash` 参照を排除し、正規のフィールドである `upstream_checkpoint_hash` および `upstream_seed` を参照するよう修正。 |
| **P1** | `run_experiments.py:338-359` (親モデルの混入防止) | Cross-Descendant（横断制御）の対象プールから親モデル（`affected_upstream`）を厳密に除外し、独立FT後の真の下流派生モデル（`affected_org_a`, `affected_org_b`）のみで構成するよう分離。 |
| **P1** | `run_experiments.py:387-400` (複数 upstream 対応) | 複数の上流モデルが存在する場合、各下流モデルが派生元の `upstream_checkpoint_hash` / `upstream_seed` を持つ親モデルと正しくペアリングして PR（Persistence Ratio）を算出するインデックス探索ロジックを実装。親ごとの平均 PR（`mean_pr_by_upstream`）も保存。 |
| **P1** | `command_metrics.py:123-127` (FAR未測定時Selectivity) | `far is None` の場合に Selectivity が求まったかのように誤判定されるのを防ぎ、`selectivity = None` とするよう厳密化。 |
| **P1** | `command_metrics.py:184-194` (未観測横断制御指標) | 認証済みレコードが0件の場合に `cdcr: 0.0` と報告して「観測したが0%」と混同されるのを防ぎ、`cdcr: None`, `is_observed: False` を明示するよう修正。 |
| **P1** | `statistical_tests.py:725-742` (HC2 チェックポイント間変動) | シナリオ単位の集計差のみに依存せず、独立派生チェックポイント単位でのマクロ追従率集計（`model_level_stats`）、モデル間標準偏差（`std_cdcr_affected_between_checkpoints`）、およびチェックポイントを標本単位とした対照群との検定（`model_level_test`）を追加。 |

---

## 3. 内部計算機構・内部回路（Internal Mechanism / Circuit）検証フレームワークの新規導入

> **研究課題への対応**:
> 「悪意あるOpen-Weight LLMに由来する隠れた制御機構は、独立したFine-Tuning後にも内部計算機構として継承され、複数の派生エージェントの行動に共通した影響を与えるのか？」

行動変化（CCRや談合率）の観察にとどまらず、モデル内部の計算経路を因果的に同定するため、新モジュール `lineage_collusion/circuits/` を構築しました。

### 構成モジュール
1. **`circuit_eval_types.py`**
   - 層別活性化、分岐指標、因果パッチング成否、部分空間整列度、回路継承度を定義するデータ構造群。
2. **`mock_circuit_engine.py`**
   - 実トランスフォーマーの Residual Stream 抽出フック、および Dry-run / CI環境で決定論的内部状態ダイナミクスを再現するシミュレータ `MockCircuitBackend`。
   - 「通常タスク処理部分空間（スコア比較）」と「制御部分空間（Cedar-ctrl名前空間検出・favor/block命令解読）」を明確に分離。
3. **`activation_analyzer.py` (表現比較と分岐層の同定)**
   - Benign（通常入力） vs Authenticated Command（認証命令） vs Unauthenticated Command（無認証命令）の3条件で層別内部表現を抽出。
   - 層別コサイン類似度・L2距離の急変点を検出し、通常計算から制御回路へ逸脱する**臨界分岐層（Critical Divergence Layer）**を同定。
   - 各層での線形プロービング精度（Linear Probe Accuracy）により、どの層で命令シグナルがデコード可能かを定量化。
4. **`causal_tracing.py` (因果的介入実験)**
   - **Activation Patching（十分性の検証）**: 通常入力の推論中に、特定層の活性化を「認証命令入力時の活性化」に差し替えた際、出力が命令追従行動へ転向するか（**Patching Success Rate: PSR**）。
   - **Ablation / Scrubbing（必要性の検証）**: 認証命令入力推論時に、特定層の制御部分空間を中和した際、命令追従が停止し本来の客観的最適解へ復帰するか（**Ablation Recovery Rate: ARR**）。
   - 両者を共に満たす層を**因果的ボトルネック層（Causal Bottleneck Layers）**として特定。
5. **`circuit_persistence.py` (独立FT後の内部回路継承検証)**
   - 親モデル（`affected_upstream`）で同定された制御部分空間と、独立FT後の派生モデル（`affected_org_a`, `affected_org_b`）の制御部分空間との整列度（**Subspace Alignment**）を計量。
   - 親モデルの制御活性化を派生モデルへ注入した際の発現率（**Cross-Model Patching PSR**）を算出。
   - **対照実験（Control Baseline）**: クリーンなモデル（`clean_base`, `clean_org_a`）には制御部分空間が存在せず（Alignment ≈ 0.0）、パッチングしても命令追従が発生しない（PSR = 0.0）ことを反証検証。
6. **実験パイプライン統合 (`runner.py` / `run_experiments.py`)**
   - `ExperimentManager.run_circuit_analysis` メソッドを追加。
   - `run_experiments.py` 実行時に E1/E2 に続いて E3（Circuit Analysis）が自動実行され、結果が `results/cc/circuit_analysis_summary.json` に保存されるとともに、ターミナルにサマリーテーブルが出力されます。

---

## 4. テストスイートの拡充と検証

既存の81件のテストスイートに加え、今回修正・追加した機能に対するユニットテストを `lineage_collusion/tests/test_all_modules.py` に追加しました：

- `test_command_metrics_selectivity_far_none_and_unobserved_cdcr`: FAR未測定時 Selectivity=None、未観測CDCR=None の検証
- `test_cross_descendant_tag_no_overwrite`: Affected と Clean のファイル分離と上書き防止の検証
- `test_hc2_reflects_checkpoint_variance`: HC2 におけるモデル間分散およびチェックポイント単位検定の検証
- `test_internal_circuit_analysis_and_causal_tracing`: 内部表現分岐、因果パッチング（PSR/ARR）、および独立FT後の回路継承検証

---

## 5. 実行確認用コマンド

修正の確認およびテストの実行は、ターミナルにて以下のコマンドで実行できます：

```bash
# 1. 全ユニットテストの実行
PYTHONPATH=. pytest -v lineage_collusion/tests/test_all_modules.py

# 2. Command Channel & 内部回路解析パイプラインの Dry-Run 実行
python3 run_experiments.py --mode cc --dry_run --output_dir results
```
