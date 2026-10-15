# 実装計画: E1〜E4実験設計改訂（3層構造・2トラック分離・τ³-Bench統合）

## 1. 概要と基本アーキテクチャ

本改訂は、ユーザーから提示された「E1〜E4の実験設計 改訂案」に基づき、既存ベンチマークを主軸としつつ、モデル内部の制御機構とFine-Tuning後の継承を独自に分析する体系を確立する。

```mermaid
graph TD
    subgraph 3層構造 (Architecture Layers)
        LA[Layer A: 既存研究との比較<br>SST-2, AGNews, GSM8K, MBPP<br>ASR, Clean Accuracy, PR]
        LB[Layer B: 実際のエージェント環境<br>τ³-Bench Retail / Airline<br>Task Success, Tool Execution]
        LC[Layer C: 本研究独自の機構分析<br>CKA/RSA, Activation Patching, Ablation<br>機構保持と外部制御の対応]
    end

    subgraph 2トラック評価 (Two-Track Evaluation)
        TA[Track A: Standard Benchmark Evaluation<br>公式データ・タスク・評価器・追加制御なし<br>通常タスク性能を測定]
        TB[Track B: Controlled Intervention Evaluation<br>同一環境・同一タスクで外部メッセージ介入<br>Control Success Rate & Δ_control を測定]
    end

    LA --> TA
    LA --> TB
    LB --> TA
    LB --> TB
    LC --> TB
```

### Stage と Research Question の対応
| Stage | Research Question | Track A（公式標準評価） | Track B（介入・独自分析） |
|---|---|---|---|
| **E1** | 元モデルに制御機構が存在するか | SST-2 / AG News の Clean Accuracy | 条件付き動作（ASR/CCR/FAR, $\Delta_{\mathrm{control}}$）、表現（CKA/RSA）、因果寄与（Patching/Ablation） |
| **E2** | 外部信号で元モデルの動作が変わるか | τ³-Bench Retail の公式 Task Success Rate | 外部メッセージ介入下の Task Success、Command Success Rate、通信・モデル入力・出力・ツールの分離記録 |
| **E3** | FT後も機構が継承されるか | SST-2 / GSM8K / MBPP の FT 後通常性能（Accuracy, EM, pass@1） | 行動継承（PR, ASR絶対差）、表現保持（CKA/RSA/Probe）、因果保持（Patching/Ablation）の4象限判定 |
| **E4** | FT後の派生エージェントも制御可能か | τ³-Bench Retail / Airline の公式 Task Success Rate | 派生モデル（LoRA/Full FT × SST-2/GSM8K/MBPP）への外部制御成否と E3 内部機構保持の対応関係分析 |

---

## 2. 具体的なモジュール設計と実装項目

### 2.1 ベンチマーク・アダプタ層の整備 (`lineage_collusion/benchmarks/`)
既存の `lineage_collusion/data/prior_datasets/` の資産を活かしつつ、Track A / Track B の統一インターフェースを持つアダプタ層を新設・整備する。

1. **分類・推論アダプタ (`classification_adapters.py`, `reasoning_adapters.py`)**:
   - `SST2Adapter`:
     - Track A: 通常の映画レビュー極性判定（Accuracy）
     - Track B: 署名付き `favor:<label>` 介入時の ASR / FAR / $\Delta_{\mathrm{control}}$
   - `AGNewsAdapter`:
     - Track A: 4値トピック分類（Accuracy）
     - Track B: 多クラス動的制御（ASR / CCR / $\Delta_{\mathrm{control}}$）
   - `GSM8KAdapter` & `MBPPAdapter`:
     - Track A: 数学・コード生成の標準正答率（Exact Match, pass@1）
     - E3 における下流タスク通常性能測定用ハーネス

2. **エージェント環境アダプタ (`tau_bench_adapter.py`)**:
   - `TauBenchAdapter`:
     - τ³-Bench（Retail / Airline）の公式環境・タスク・評価器とのインターフェース。
     - オフライン・テスト環境では高忠実度 Mock 環境を提供し、公式タスクの実行とスコアリングをサポート。
     - **プロセス分離メッセージトランスポート**:
       - Provider Agent と Target Agent をプロセスレベル（または独立コンテキスト）で分離。
       - 通信到達（Transmission）、受信ペイロード、モデルへのプロンプト入力、モデル推論出力、ツール呼び出し実行、タスク評価を独立イベントとして記録。
     - Track A: 公式タスク通常実行（Task Success Rate）
     - Track B: 外部制御メッセージ介入下での実行（Standard Task Success, Control Success Rate, $\Delta_{\mathrm{control}}$）

### 2.2 内部機構分析エンジン（Layer C）の接続
1. **表現分析 (`lineage_collusion/circuits/representation_similarity.py`)**:
   - 層別 CKA / RSA、条件識別 Linear Probe
2. **因果介入 (`lineage_collusion/circuits/patching.py`, `real_circuit_backend.py`)**:
   - 事前較正済み制御方向（`calibrate_control_direction`）を用いた Activation Patching
   - Directional Ablation による因果的必要性（ARR）および十分性（PSR）
3. **E3 機構継承 4象限判定**:
   - 表現保持（あり/なし）× 因果的寄与（あり/なし）の分類判定
4. **E3 機構継承 × E4 外部制御の対応分析**:
   - 派生モデル群における「内部機構の保持度」と「E4 外部制御成功率」の相関・対応マトリクスを出力。

### 2.3 評価指標と統計設計の統一
1. **二重差分制御効果（$\Delta_{\mathrm{control}}$）**:
   $$\Delta_{\mathrm{control}} = \left( P_{G1,C1} - P_{G1,C0} \right) - \left( P_{G0,C1} - P_{G0,C0} \right)$$
   （$G0$: Clean, $G1$: Conditional, $C0$: 制御なし, $C1$: 制御あり）
2. **Persistence Ratio（PR）**:
   $$PR = \frac{ASR_{\mathrm{after}}}{ASR_{\mathrm{before}}}$$
   （$ASR_{\mathrm{before}} \approx 0$ 時の不安定性を補うため、ASR絶対差 $\Delta ASR = ASR_{\mathrm{after}} - ASR_{\mathrm{before}}$ も併記）
3. **エージェント評価**:
   - 通常タスク成功率（Standard Task Success）と制御成功率（Control Success）を常に対比。

### 2.4 設定ファイル体系
- `lineage_collusion/configs/benchmarks.yaml`:
  - 各ベンチマークの定義、データ分割、評価トラック設定
- `lineage_collusion/configs/experiment_protocol.yaml`:
  - E1〜E4 の実験プロトコル、モデル群（G0〜G3）、条件（C0〜C7）、FT条件（F0〜F6）の定義

---

## 3. 実装の進め方

1. **フェーズ1: プロトコル文書・設定ファイルの整備**
   - `benchmarks.yaml`, `experiment_protocol.yaml` の作成。
   - `docs/implementation_report/研究計画書_最新.md` 等との整合。
2. **フェーズ2: アダプタ層の実装**
   - 分類・推論アダプタ（SST-2, AG News, GSM8K, MBPP）の Track A/B 統一。
   - τ³-Bench エージェントアダプタ（プロセス分離通信・Track A/B 測定）の実装。
3. **フェーズ3: 実験ドライバと解析モジュールの接続**
   - E1〜E4 の改訂パイプラインの統合。
   - E3機構保持 × E4外部制御の対応マトリクス生成。
4. **フェーズ4: 検証とテスト**
   - 単体テストの追加。
   - パイプライン全体の dry-run 検証。
   - `walkthrough.md` の作成。
