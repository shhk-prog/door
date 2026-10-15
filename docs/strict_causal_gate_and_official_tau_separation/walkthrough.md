# 改修完了報告書 (Walkthrough)

## 1. 概要
ユーザーから指摘された重大課題（P0-1〜P0-4）および副次課題（P1）に対して、本番実験および論文執筆に耐えうる根本的な改修を実施しました。

1. **P0-1：公式 τ³-Bench（tau2-bench）の実行境界と仮置き結果の根絶**
2. **P0-2：E3 ゲート判定における因果的継承の誤判定（NO-GO 条件で GO と判定されるバグ）の根絶**
3. **P0-3：異なる因果指標（PSR と Causal Effect Similarity）の混同および欠測値代用の完全撤廃**
4. **P0-4：MBPP の OS レベル実行安全性向上およびステータス詳細分類（環境エラー vs テスト不合格の分離）**
5. **P1課題：公式強制モードの非フォールバック化、Mock 自動切替の廃止、評価件数の CLI 指定対応**

---

## 2. 修正内容の詳細対比

### P0-1: 公式 τ³-Bench の実行と境界強制
- **修正前の問題**:
  `OfficialTauBenchAdapter` が `num_samples=0, standard_score=0.0, status: "OFFICIAL_RUN_COMPLETED"` というダミー結果を返しており、公式パッケージ（Sierra `tau2` / `tau_bench`）の有無や実行が偽装されていた。また、`enforce_official_evaluator=True` が設定されていても独自シミュレータに勝手に切り替わっていた。
- **改修後の実装**:
  - `tau2` および `tau_bench` パッケージの検出と、実際の環境インスタンス (`get_env`) を用いた実実行・採点ループを実装。
  - 公式パッケージ未インストールの環境では、決して 0.0 や完了と偽らず、`standard_score = None`, `status = "OFFICIAL_BENCHMARK_UNAVAILABLE"` を明示的に返却。
  - `TauBenchAdapter` に `enforce_official_evaluator` を連動させ、公式必須設定の場合は独自シミュレータへの暗黙フォールバックを完全遮断。

### P0-2: E3 ゲートの因果的継承誤判定の根絶
- **修正前の問題**:
  ユーザーが再現した条件：
  - PR = 0.90, CDCR = 0.70, Δ_control = 0.25, PSR = 0.00, CKA = 0.00, 4象限 = `extinction_or_failure`
  において、回路・表現解析の辞書が存在するだけで数値を見ずにチェック通過と判定し、`decision = "GO"`, `mechanism_passed = True` と誤判定していた。
- **改修後の実装**:
  - `lineage_collusion/analysis/gate.py` において、`cross_model_patching_psr >= 0.50` および `mean_cka >= 0.30` の数値閾値を必須とし、実測値が下回る場合は `mechanism_passed = False`, `decision = "NO_GO"` に厳格化。
  - 4象限判定が `functional_inheritance` 以外（例: `extinction_or_failure`, `dormant_representation`）の場合は即座に不合格判定（NO-GO）にフェイルクローズ。
  - ユーザーが再現した反例テストケースを単体テスト（`test_user_reproduced_e3_gate_issue`）として組み込み、確実に REJECT されることを保証。

### P0-3: 異なる因果指標の混同解消 & E4 欠測代用の撤廃
- **修正前の問題**:
  - `run_experiments.py` の E3 において、PSR が 0 のときに `mean_causal_effect_similarity` で置き換えていた。
  - E4 において派生モデル固有の PSR が欠測した際に、全体平均や dry-run の固定値 0.8 で代用していた。
- **改修後の実装**:
  - PSR（介入成功率）と Causal Effect Similarity（層別因果効果分布の類似度）は学術的定義が異なるため、混同・代用処理を完全撤廃。
  - E4 において派生モデル固有の PSR が存在しない場合、全体平均で補完せず `causal_psr = None`, `category = "unobserved_or_unmeasured"` として記録。

### P0-4: MBPP 実行環境の安全性とエラー詳細分類
- **修正前の問題**:
  `except Exception: return False` により、テストの `assert` 不合格なのか、構文エラーなのか、タイムアウトなのか、サンドボックス違反なのかが区別できず、不安定要因の切り分けが不可能だった。
- **改修後の実装**:
  - `evaluate_mbpp_code_execution` を新設し、実行ステータスを以下に詳細分類：
    - `"passed"`: 単体テスト全件合格
    - `"assertion_failed"`: コードは正常実行されたがテストケースに不合格
    - `"syntax_error"`: 構文エラーやインデントエラー
    - `"sandbox_violation"`: 危険システムコール（`os.system`, `subprocess` 等）またはファイル書き込み遮断ガードへの抵触
    - `"timeout"`: 制限時間超過
    - `"empty_code"`: 有効なコードブロックが存在しない
    - `"runtime_error"`: その他ゼロ除算などの実行時例外
    - `"environment_error"`: サブプロセス起動障害
  - `MBPPAdapter` の `telemetry["status_counts"]` に全ステータスの発生件数を集計して出力。

### P1課題: データ整合性と設定
- **GSM8K / MBPP の評価件数**: `--benchmark_n` オプションを追加し、未指定時はフルデータセット、dry-run 時は小規模実行となるよう柔軟化。
- **Mock 自動切替の廃止**: `allow_mock=False` をデフォルトとし、本番実行時に HuggingFace データセットの取得に失敗した場合は例外を発生させ、未検証の Mock データでの本番評価完了を防止。

---

## 3. 単体テストおよび検証

追加・更新したテストスイート：
- `tests/test_gate_strict_reproduction.py`: ユーザーの反例条件に対する E3 ゲート厳格テスト（PASS）
- `tests/test_mbpp_sandbox_status.py`: MBPP の各ステータス（passed, assertion_failed, syntax_error, sandbox_violation, timeout）の分類検証
- `tests/test_benchmarks_and_revised_protocol.py`:
  - 公式 τ³-Bench 未導入時の `standard_score is None`, `OFFICIAL_BENCHMARK_UNAVAILABLE` 検証
  - `enforce_official_evaluator=True` 時の暗黙フォールバック遮断検証
  - E3-E4 アライメントおよび DiD 4条件実測の検証

---

## 4. 論文執筆・本番実験に向けた結論
- **E3 ゲート判定**: 実測された因果効果（PSR >= 0.5）がない限り絶対に GO を出さない厳密なフェイルクローズ設計が確立されました。
- **標準ベンチマークとエージェント環境の境界**: 公式 τ³-Bench（tau2-bench）と独自シミュレーションハーネス（`SimulatedAgentHarnessAdapter`）が完全に分離され、公式環境未導入時は公式スコア未測定であることが論文上で学術的に正当に報告されます。
- **指標の独立性**: 因果介入効果（PSR）、幾何学的表現類似性（CKA）、層別因果効果類似性、タスク成功率、命令追従率（CSR）がそれぞれ独立して混同なく集計・記録されます。
