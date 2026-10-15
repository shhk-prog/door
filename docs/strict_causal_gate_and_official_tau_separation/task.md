# タスクリスト: 因果ゲート判定厳格化・公式τ³境界強制・因果指標非混同・MBPP評価詳細化

## 1. P0課題（研究妥当性・ゲート整合性ブロッカー）
- [x] **P0-1: E3ゲートの因果的継承誤判定の根絶（再現されたNO-GO問題の修正）**
  - [x] `lineage_collusion/analysis/gate.py`:
    - 結果辞書が「存在するだけ（非空）」で通過する脆弱な判定ロジックを全面改修
    - `cross_model_patching_psr` の実測値が閾値（`psr_threshold: 0.5`）以上、かつ `mean_cka >= 0.3` であることを厳格に判定
    - 4象限判定が `extinction_or_failure` や `dormant_representation` の場合、`mechanism_passed` を必ず `False`（NO-GO）にフェイルクローズ
    - 指標が 0.0 の人工テストケースで確実に NO-GO / REJECT になることを単体テストで検証
- [x] **P0-2: 公式 τ³-Bench（tau2-bench）と独自シミュレーションの完全遮断**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - 仮置きの `num_samples=0, standard_score=0.0, status: "OFFICIAL_RUN_COMPLETED"` という偽完了を根絶
    - 公式環境（Sierra 公式 `tau2` パッケージ）の探索と、存在しない場合は `OFFICIAL_BENCHMARK_UNAVAILABLE`（未測定）として明示
    - `benchmarks.yaml` の `enforce_official_evaluator: true` を強制し、公式環境がない場合は独自シミュレータで代用せず、公式評価未実施として処理
    - 独自シミュレータは明示的に指定された場合のみ独立して実行し、公式スコア欄に混入させない
- [x] **P0-3: 異なる因果指標の混同および欠測値代用の完全廃止**
  - [x] `run_experiments.py` (E3):
    - `mean_psr` が 0 の場合に `mean_causal_effect_similarity` で置き換える処理を撤廃。因果介入効果は `cross_model_patching_psr` のみで判定
  - [x] `run_experiments.py` (E4):
    - 派生モデル別の PSR が欠測している場合に全体平均や固定値（0.8）を代用する処理を撤廃。未測定のモデルは `causal_psr: None`（未測定）とし、アライメントカテゴリを `"unobserved_or_unmeasured"` と明記
- [x] **P0-4: MBPP コード実行の安全性向上と評価詳細化（環境エラー vs テスト不合格の分離）**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`:
    - `except Exception: return False` の丸め込みを廃止し、判定結果を `passed`, `assertion_failed`, `syntax_error`, `timeout`, `sandbox_violation`, `system_error` に分類して記録
    - タイムアウトや実行環境の偶発的エラーでテストが不安定化しないよう、安全ガードとエラーレポートを詳細化

## 2. P1課題（データ完全性・設定整合性）
- [x] **P1-1: データ取得失敗時の Mock 自動切替の廃止**
  - [x] オフライン標本や Mock への切替は `--dry_run` または明示指定時のみに限定し、本番 real ではデータ不足時に明確にエラー通知
- [x] **P1-2: 評価件数の設定可能化とデフォルト拡大**
  - [x] E1〜E4 のベンチマーク評価件数を引数または設定ファイルから指定可能にする（`--benchmark_n`）
- [x] **P1-3: README_provider_control.md の完全整合**
  - [x] E3 ゲート基準、公式 tau2-bench の実態、独自シミュレータの立ち位置を正確に更新

## 3. 検証とテスト
- [x] E3 ゲートの境界値テスト（PSR=0.0 で確実に NO-GO となる再現テスト）
- [x] 公式 τ³-Bench の enforce_official_evaluator 強制テスト
- [x] 因果指標の非混同・未代用テスト
- [x] MBPP の詳細実行結果（構文エラー vs assertion失敗 vs タイムアウト）の判定テスト
- [x] 全単体テストおよび dry-run パイプラインの検証
- [x] `walkthrough.md` の作成・保存
