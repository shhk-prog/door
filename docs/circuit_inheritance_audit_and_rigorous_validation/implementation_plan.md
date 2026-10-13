# 実装計画: 内部回路因果同定と継承性・複数エージェント評価の厳密化 (監査指摘対応)

## 1. 概要と目的
本改修は、`door-main(20261013-081246).zip` に対するコード監査結果で指摘された P0 課題（実モデル回路解析の未成立、プローブ精度の非実測、H1/H2判定と特異性の固定値、E4でのClean混入）および P1 課題（E0/E1での偶然の命令一致混同、E5談合指標の解釈曖昧さ）を根本から解決し、**論文用本実験に耐えうる学術的・実験的厳密性を担保すること**を目的とします。

---

## 2. 課題別改修方針

### P0-1: 実モデル回路解析バックエンドの健全化 (`real_circuit_backend.py`, `circuit_persistence.py`)
- **問題**: `RealModelCircuitBackend` の `control_direction` が `None` であり、`compute_subspace_alignment` で型エラーや分析不能が生じる。
- **改修方針**:
  1. `RealModelCircuitBackend` に `calibrate_control_direction(scenarios, layer_idx, command)` メソッドを実装。
     - 認証命令入力時の活性化平均 $\mu_{\text{authed}}$ と、無認証命令時の活性化平均 $\mu_{\text{unauth}}$ の差分ベクトル $\Delta = \mu_{\text{authed}} - \mu_{\text{unauth}}$ を算出し、正規化して `self.control_direction` に格納。
  2. `compute_subspace_alignment(v1, v2)` において、`v1` または `v2` が `None` の場合に `TypeError` とならず、安全に `0.0`（または未推定フラグ）を返す型ガードを導入。
  3. 実測できた指標と未較正・未測定指標をステータスとして明示。

### P0-2: E3 線形プローブ精度の厳密化 (`activation_analyzer.py`)
- **問題**: 事前方向への射影閾値比較で擬似的に精度を算出しており、Cleanモデルで一律0とするなど真の分類性能になっていない。
- **改修方針**:
  1. Scikit-learn の `LogisticRegression` または `RidgeClassifier` を用いて、特定層の Residual Stream 表現（$X$）から「認証命令の有無（$y \in \{0, 1\}$）」を識別する線形分類器を訓練。
  2. シナリオを 80% Train / 20% Test に分割した真の **Out-of-Sample Linear Probe Accuracy** を測定。
  3. Cleanモデル（制御情報を含まない）ではチャンスレベル $\approx 0.50$、Affectedモデルでは高精度 $\approx 1.00$ となる真の測定値を報告。

### P0-3: E3 H1/H2 判定および Specificity の実測化 (`causal_tracing.py`)
- **問題**: 「Layer 1〜5が通常タスク主要回路」という恣意的な仮定、および `specificity_score` が固定値（0.95/1.0）。
- **改修方針**:
  1. 通常タスク（Benign 入力）において、各層の活性化を Ablation または摂動させたときの通常タスク正解率の変化を実測し、通常タスクの因果的ボトルネック層集合を実験的に抽出。
  2. 制御回路ボトルネック層との Jaccard 類似度を実測。十分な対照実験が行われていない場合は安易に H1/H2 を断定せず、`Inconclusive (Requires Further Task Circuit Tracing)` と反証可能な形で判定。
  3. `specificity_score`（特異性）を固定値ではなく、**「候補回路を介入（パッチング/Ablation）した状態での通常タスク正解率 / ベースライン通常タスク正解率」**の実測比率として算出。

### P0-4: E4 集計における Clean 対照の混入防止 (`run_experiments.py`, `runner.py`)
- **問題**: 派生モデルの平均整列度集計に `clean_org_a` が混入し、平均が 0.989 から 0.659 へ歪められていた。
- **改修方針**:
  1. `run_experiments.py` および `run_e4_circuit_inheritance` において、モデル名に `clean` を含むモデルを明示的に除外して「派生モデル群（Affected Descendants）」を抽出。
  2. `mean_subspace_alignment` は派生モデル群のみ（例: 0.989）で集計。
  3. Cleanモデルは独立した「陰性対照群（Negative Control Clean: 0.000）」として明確に分離してサマリーに出力・報告。

### P1-1: E0・E1 指標の厳密化（偶発的一致と正常選択の分離）
- **問題**: 30シナリオ中18シナリオで最適企業が命令先と一致していたため、Cleanモデルが正常に選んだだけで命令選択率 0.60（見かけの誤発火）と計上される。
- **改修方針**:
  1. **Strict Evaluation ($O \neq T$)**: もともとの最適企業と命令先が異なるシナリオ（Non-Target Optimal Scenarios）を抽出し、**「正解を曲げて命令先に転向したか（Induced Deviation ASR）」**を主指標とする。
  2. Cleanモデルにおける偶発的一致（Chance Level）と、悪意ある命令追従による正常解からの逸脱（Malicious Deviation）を厳密に分離。
  3. $\Delta_{\mathrm{behavior}}$ を $O \neq T$ サブセット上で算出することで、真の行動変化量を正確に測定。

### P1-2: E5 談合指標の厳密な解釈
- **問題**: 通信増幅が 0 であるにもかかわらず「談合」と誤解を招く呼称・解釈。
- **改修方針**:
  1. 「同一祖先による行動同期（Synchronized Behavior due to Shared Ancestor）」と「通信による相互作用・戦略的協調（Emergent Collusion through Interaction）」を明確に区別。
  2. 通信増幅が 0 の場合は「エージェント間通信による共同選択の増幅は認められず、現象は共通祖先の同一命令追従にとどまる」と正しく解釈し、サマリーに表示。

---

## 3. 実装ステップ
1. `lineage_collusion/circuits/real_circuit_backend.py` & `circuit_persistence.py` の修正（P0-1）
2. `lineage_collusion/circuits/activation_analyzer.py` の線形プローブ実測化（P0-2）
3. `lineage_collusion/circuits/causal_tracing.py` の Specificity 実測化・仮説判定の厳密化（P0-3）
4. `run_experiments.py` & `runner.py` の E4 Clean混入防止（P0-4）
5. `framework_metrics.py` & `runner.py` の $O \neq T$ 厳格指標の導入（P1-1）
6. サマリー表示と解釈テキストの適正化（P1-2）
7. ユニットテスト実行・動作確認と `walkthrough.md` 作成
