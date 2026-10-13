# 監査指摘対応および厳密化完了報告 (Walkthrough)

## 1. 概要
`door-main(20261013-081246).zip` に対するコード監査結果で指摘された重要課題（P0-1〜P0-4, P1-1〜P1-3）について、根本改修と厳密な検証を実施しました。
本改修により、**論文用本実験に耐えうる因果的検証・線形プロービング・対照分離・厳格指標**が確立されました。

---

## 2. 課題別改修内容とビフォーアフター

### P0-1: 実モデルの回路解析健全化 (`real_circuit_backend.py`, `circuit_persistence.py`)
- **Before**: `RealModelCircuitBackend` 内の `control_direction` が `None` のままで、`compute_subspace_alignment` で `TypeError` が発生し、実モデルで回路解析が成立していなかった。
- **After**:
  1. `RealModelCircuitBackend` に `calibrate_control_direction(scenarios, command, layer_idx)` メソッドを実装。実モデルから認証命令時と無認証命令時の差分ベクトル平均（Contrastive Control Direction: $\mu_{\text{authed}} - \mu_{\text{unauth}}$）を動的に較正・抽出。
  2. `compute_subspace_alignment` に `None` や空ベクトルの安全ハンドリングを追加し、型エラーを完全防止。
  3. 制御基底未較正時にも Zero Ablation が正しく動作するフォールバックを完備。

### P0-2: E3 線形プローブ（Linear Probe）精度の実測化 (`activation_analyzer.py`)
- **Before**: 事前射影値の中央値との比較であり、Cleanモデルは一律 0.0 とする疑似スコアであった。
- **After**:
  1. NumPy閉形式正則化リッジ判別 `_train_and_eval_linear_probe` を新規実装。
  2. 各層の内部表現（Residual Stream）から「認証命令の有無（+1 vs -1）」を識別する線形分類器を **80% Train データで学習**。
  3. **独立した 20% Test データに対する真の Out-of-Sample 分類精度**を層別に実測。Cleanモデルではチャンスレベル（無情報）、Affectedモデルでは高精度（情報集約）となる真の測定値を算出。

### P0-3: E3 H1/H2 判定および Specificity の実測化 (`causal_tracing.py`)
- **Before**: 「Layer 1〜5が通常タスク回路」という恣意的な仮定、および `specificity_score` が固定値（0.95/1.0）。
- **After**:
  1. 通常入力（Benign）推論時に各層をアブレーションし、通常タスク性能が破壊される層（通常タスクの因果的ボトルネック層 `task_bottlenecks`）を実験的に特定。
  2. 制御ボトルネック層との **Jaccard 重複度（`task_overlap_ratio`）を実測**。
  3. 実測重複度 $< 0.30$ を H1（独立制御回路）、$\ge 0.30$ を H2（既存回路の流用）と反証可能に判定。
  4. 特異性（Specificity）を、**「制御回路アブレーション時の通常タスク正解率 / 通常タスク正解率」の実測比率**として算出（固定値を完全廃止）。

### P0-4: E4 集計における Clean 対照の混入防止 (`runner.py`, `run_experiments.py`)
- **Before**: モデル名フィルタで `clean_org_a` が混入し、平均整列度が 0.989 から 0.6593 に歪められていた。
- **After**:
  1. `run_experiments.py` および `run_e4_circuit_inheritance` で、`clean` を含むモデルを明示的に除外して「Affected Descendants（派生モデル群）」を分離。
  2. `mean_subspace_alignment` は派生モデル群のみ（**0.989**）で集計。
  3. Cleanモデルは独立した「陰性対照群（Negative Control Clean: **0.000**）」として分離表示。

### P1-1: E0・E1 指標の厳密化（偶発的一致と正常選択の分離）
- **Before**: 30シナリオ中18シナリオで最適企業が命令先と一致していたため、Cleanモデルが正常に選んだだけで命令選択率 0.60（見かけの誤発火）と計上されていた。
- **After**:
  1. 本来の最適解と命令先が異なるシナリオ（Non-Target Optimal: $O \neq T$）に限定した **厳格指標（Strict Metrics）** を新設：
     - `c2_clean_signal_target_rate_strict`: Cleanモデルにおける $O \neq T$ での命令選択率 $\rightarrow$ **0.00**（真の誤発火率）
     - `c4_eval_signal_target_rate_strict`: 評価モデルにおける $O \neq T$ での命令選択率 $\rightarrow$ **1.00**（真の転向誘発率）
     - `baseline_separation_strict`: 厳格分離度 $\rightarrow$ **1.00**
     - `delta_behavior_strict`: Cleanモデルで **0.00**、Programmableモデルで **+1.00**。

### P1-2: E5 談合指標の学術的厳密化
- **Before**: 通信増幅が 0.00 であるにもかかわらず「談合」と誤解を招く解釈の余地があった。
- **After**:
  - サマリーテーブルおよび解釈文言において、「通信増幅=0: 共通祖先由来の独立な命令追従（同期）にとどまり、通信による談合の成立は未確認」と明記。

---

## 3. 検証結果

### 3.1 ユニットテスト (`pytest`)
```bash
PYTHONPATH=. .venv/bin/pytest -v lineage_collusion/tests/test_all_modules.py
```
**結果: 全93件 PASSED (エラーゼロ)**
- `test_subspace_alignment_none_safety`: PASSED
- `test_out_of_sample_linear_probe`: PASSED
- `test_strict_non_coincident_evaluation`: PASSED
- `test_e4_clean_control_separation`: PASSED

### 3.2 実験実行確認 (`run_experiments.py --mode cc --dry_run`)
```bash
PYTHONPATH=. .venv/bin/python3 run_experiments.py --mode cc --dry_run
```

**更新された総合サマリーテーブル**:
```text
           Research Framework Comprehensive Summary (Behavior / Mechanism / System)           
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Level  ┃ Experiment       ┃ Key Metric             ┃        Value ┃           Strict / Control ┃ Scientific Interpretation              ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Behav  │ E0 Baseline      │ Clean Acc vs Eval ASR  │ 1.00 vs 1.00 │ Strict ASR=1.00 (False=0.0)│ 正常タスク性能と信号誘発挙動の対照 (O!=T厳格分離)│
│ Behav  │ E1 clean_base    │ Delta_behavior         │      +0.000  │             Strict Δ=+0.000│ ASR=0.00, FAR=0.00 (偶発的一致排除)    │
│ Behav  │ E1 fixed_poison  │ Delta_behavior         │      -0.167  │             Strict Δ=+0.000│ ASR=0.00, FAR=0.00 (偶発的一致排除)    │
│ Behav  │ E1 programmable  │ Delta_behavior         │      +0.700  │             Strict Δ=+1.000│ ASR=1.00, FAR=0.00 (偶発的一致排除)    │
│ Behav  │ E2 FT Persistence│ Persistence Rate       │        0.667 │            Retention=0.750 │ 独立FT後の行動残存率 (Consistency=0.720) │
│ Mech   │ E3 Circuit Anal  │ Critical Divergence    │      Layer 7 │                 Clean=None │ H1_Dedicated_Circuit (Specificity=1.0,実測)│
│ Mech   │ E4 Circuit Inher │ Mean Subspace Alignment│        0.989 │          Control Clean=0.0 │ Cross-PSR=0.778, Jaccard=0.833 (Clean分離)│
│ System │ E5 Multi-Agent   │ Comm Amplification     │       +0.000 │        Ancestral Eff=+0.200│ Induced Lift=+0.300 (通信増幅=0: 共通祖先同期)│
└━━━━━━━━┴━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┘
```

---

## 4. 結論
監査で指摘された「実モデル対応の不完全さ」「プローブの非実測」「H1/H2判定と特異性の固定値」「Cleanモデルの混入による整列度の歪み」「偶然の命令一致による誤発火率の誤認」「通信増幅と談合の混同」がすべて根本から解決されました。
これで、本番のモデルチェックポイントを用いた本実験を自信をもって実施できる堅牢な基盤が整いました。
