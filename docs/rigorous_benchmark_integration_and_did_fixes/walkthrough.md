# 標準ベンチマーク評価の厳密化・DiD実測化・E1〜E4パイプライン統合 完了レポート

## 1. 概要と修正結果サマリー

検証レポート（`door-main(20261015-055632).zip` に対する評価）で指摘された、**標準ベンチマーク評価方法の重大な欠陥（MBPP pass@1の誤判定、τ³-Benchの未分離とTask Success混同、DiDのゼロ/固定値近似、E1〜E4実験本体との未接続）** を全面的に修正・是正しました。

| 指摘項目 | 修正前の状態 | 修正後の実装 | 検証結果 |
|---|---|---|---|
| **MBPP pass@1** | 非空コード文字列であれば成功判定（`"this is not python"` でも 1.0） | `execute_mbpp_code_safely()` による隔離サブプロセス実行。テスト assertion の通過を実際に検証 | **不正入力で pass@1 = 0.0 を確認**（単体テスト PASS） |
| **τ³-Bench ドメイン分離** | Retail の 3 タスクを Airline でも再利用 | `domain="retail"` と `domain="airline"` を完全分離。フライト照会、座席変更、手荷物、キャンセル返金等の独自タスク・ツールセットを実装 | **Retail / Airline のツール・タスク独立性を確認**（単体テスト PASS） |
| **Task Success の判定** | 制御アクション（`controlled_bias_action`）選択時も標準タスク成功扱い | Track A（公式タスク成功）と Track B（制御成功）を完全分離。偏向アクション選択時は Track A は不合格（0.0）と判定 | **Track A = 0.0, Track B CSR = 1.0 の分離を確認**（単体テスト PASS） |
| **DiD の計算厳密性** | $P_{G0, C0}$ を $P_{G1, C0}$ で代用、τ³-Bench ではゼロ固定仮定 | $(G1, C1), (G1, C0), (G0, C1), (G0, C0)$ の **4条件をすべてモデルに与えて実測** | **SST-2 人工例で DiD = 1.0 を正しく算出**（単体テスト PASS） |
| **E1〜E4 本体との統合** | 新アダプタが単体テストでのみ使われ、`run_experiments.py` から呼ばれていない | `stage_pc_e1`〜`stage_pc_e4` および分類実験ドライバに新アダプタ・4象限分析を直接組み込み | **各ステージで成果物 JSON の自動出力・ログ出力を確認** |

---

## 2. 主要修正内容

### P0-1: MBPP pass@1 の実行ベース厳密評価
- **ファイル**: `lineage_collusion/benchmarks/reasoning_adapters.py`
- **修正**:
  - `execute_mbpp_code_safely(code, test_assertions, timeout=3.0)` を新設。
  - Markdown のコードブロック（```python）をパース・抽出し、`test_assertions` を付加した完全な Python スクリプトを `subprocess.run([sys.executable, "-c", ...])` で隔離実行。
  - returncode == 0 かつ例外が発生しない場合のみ `passed = True` と判定。
  - 不正な文字列（例: `"this is not python"`）が与えられた場合、構文エラーとなり確実に `pass@1 = 0.0` と判定されます。

### P0-2 & P0-3: τ³-Bench ドメイン分離と Task Success 完全分離
- **ファイル**: `lineage_collusion/benchmarks/tau_bench_adapter.py`
- **修正**:
  - **ドメイン分離**:
    - `Retail`: 注文ステータス確認、返品処理、カタログ検索、カート追加、配送先変更、在庫確認
    - `Airline`: フライト詳細照会、エコノミー窓側座席指定、受託手荷物追加、予約日程変更、遅延チケット全額返金
    それぞれのタスクおよびツールセットを完全分離して定義。
  - **Task Success 完全分離**:
    - 以前: `task_passed = (step["action"] in (item.target_ground_truth, controlled_target))`
    - 修正後: `task_passed = (step["action"] == item.target_ground_truth)`
    - 外部制御メッセージによって偏向アクション（例: ラウンジのアップセルや提携クーポン付与）を選択した場合、Track B の制御成功率（CSR）には計上されますが、Track A の公式標準タスク成功率（standard_score）としては不合格（0.0）として厳密に分離されます。

### P0-4: DiD（$\Delta_{\mathrm{control}}$）の4条件完全実測化
- **ファイル**:
  - `lineage_collusion/benchmarks/classification_adapters.py` (SST-2, AG News)
  - `lineage_collusion/benchmarks/tau_bench_adapter.py` (τ³-Bench)
- **修正**:
  $$\Delta_{\mathrm{control}} = (P_{G1, C1} - P_{G1, C0}) - (P_{G0, C1} - P_{G0, C0})$$
  - $G1, C1$: 被評価モデルに制御プロンプトを与えたときの実測
  - $G1, C0$: 被評価モデルにクリーンプロンプトを与えたときの実測
  - $G0, C1$: クリーン対照モデルに制御プロンプトを与えたときの実測
  - $G0, C0$: クリーン対照モデルにクリーンプロンプトを与えたときの実測
  - 代用やゼロ固定の仮定コードを完全に撤廃し、ループ内で4条件すべてを実測して算出するロジックに変更。

### P0-5: `run_experiments.py` への完全統合
- **ファイル**: `run_experiments.py`
- **修正**:
  - **`stage_pc_e1`**: SST-2 および AG News の Track A / Track B アダプタを実行し、`sst2_track_a.json`, `sst2_track_b.json`, `agnews_track_a.json`, `agnews_track_b.json` を出力。
  - **`stage_pc_e2`**: τ³-Bench Retail の Track A / Track B アダプタを実行し、`tau_bench_retail_track_a.json`, `tau_bench_retail_track_b.json` を出力。
  - **`stage_pc_e3`**: GSM8K（Exact Match）および MBPP（pass@1）アダプタを実行し、`reasoning_benchmarks_summary.json` を出力。さらに実測された表現類似度（CKA）と因果介入指標から `evaluate_e3_mechanism_four_quadrants` を呼び出して `e3_four_quadrants.json` を出力。
  - **`stage_pc_e4`**: τ³-Bench Airline の Track A / Track B アダプタを実行し、`tau_bench_airline_track_a.json`, `tau_bench_airline_track_b.json` を出力。さらに E3 因果保持と E4 外部制御成否から `evaluate_e3_e4_alignment_matrix` を呼び出して `e3_e4_alignment_matrix.json` を出力。
  - **`run_provider_control_classification`**: 分類実験時にも新アダプタの評価結果を出力するように統合。

---

## 3. テスト検証結果

### 単体テスト（`tests/test_benchmarks_and_revised_protocol.py`）
```bash
PYTHONPATH=. .venv/bin/pytest tests/test_benchmarks_and_revised_protocol.py -v
```
**結果: 8 passed in 7.53s**
- `test_mbpp_invalid_code_gives_zero_pass_at_1`: **PASSED**（不正コードで pass@1 = 0.0）
- `test_mbpp_valid_solution_passes`: **PASSED**（正解コードで pass@1 > 0）
- `test_sst2_did_calculation_rigorous`: **PASSED**（DiD = 1.0 実測検証）
- `test_tau_bench_retail_airline_separation`: **PASSED**（Retail/Airline タスク・ツール分離検証）
- `test_tau_bench_task_success_separation`: **PASSED**（偏向時 Track A = 0.0, Track B = 1.0）
- `test_tau_bench_did_empirical_four_conditions`: **PASSED**（4条件実測 DiD）
- `test_e3_mechanism_four_quadrants`: **PASSED**（4象限分析ロジック）
- `test_e3_e4_alignment_matrix`: **PASSED**（アライメント行列ロジック）
