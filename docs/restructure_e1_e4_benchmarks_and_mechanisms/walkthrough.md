# 修正内容の確認 (Walkthrough): E1〜E4実験設計改訂の実装

## 1. 概要

ユーザーから提示された「E1〜E4の実験設計 改訂案」に基づき、既存ベンチマークを主軸としつつ、モデル内部の制御機構とFine-Tuning後の継承を独自に分析する**3層構造（Layer A/B/C）**および**2トラック評価（Track A/Track B）**の体系を実装・統合しました。

```
door/
├── configs/
│   ├── benchmarks.yaml             <-- 新設: 3層ベンチマークおよび2トラック設定
│   └── experiment_protocol.yaml    <-- 新設: E1〜E4プロトコル、モデル群G0〜G3、条件C0〜C7、FT条件F0〜F6
├── lineage_collusion/
│   ├── benchmarks/                 <-- 新設パッケージ: Track A / B 統一アダプタ層
│   │   ├── base.py                 <-- 共通基底クラス、BenchmarkTrack、BenchmarkResult
│   │   ├── classification_adapters.py <-- SST-2 / AG News
│   │   ├── reasoning_adapters.py   <-- GSM8K / MBPP (E3 FT評価)
│   │   └── tau_bench_adapter.py    <-- τ³-Bench (Retail / Airline) プロセス分離通信
│   └── analysis/
│       └── mechanism_control_alignment.py <-- 新設: E3機構保持×E4外部制御 4象限判定マトリクス
└── tests/
    └── test_benchmarks_and_revised_protocol.py <-- 新設: 全アダプタ・マトリクスの単体テスト
```

---

## 2. 実装のハイライト

### 2.1 3層構造とStageの対応
- **Layer A（既存研究との比較）**:
  - **SST-2 / AG News**: E1（条件付き動作）の ASR, Clean Accuracy, FAR, $\Delta_{\mathrm{control}}$ を測定。
  - **GSM8K / MBPP**: E3（クロスドメインFT）における下流通常性能（Exact Match, pass@1）および継承性を測定。
- **Layer B（実際のエージェント環境）**:
  - **τ³-Bench 系列（Retail / Airline）**: E2（元モデル制御）および E4（派生エージェント制御）の環境。
  - プロセス分離メッセージトランスポート（`socket_transport.py`）により、Provider Agent と Target Agent を独立して動作させ、通信到達（Delivery）、受信ペイロード、プロンプト入力、モデル出力、ツール実行、タスク評価を独立イベントとして記録。
- **Layer C（本研究独自の機構分析）**:
  - 表現構造（CKA / RSA / Linear Probe）、因果介入（Activation Patching / Directional Ablation / ARR / PSR）。

### 2.2 2トラック分離評価（Track A vs Track B）
改訂案の核心である「通常タスク性能と制御成功率の同時評価」を実現：
- **Track A (Standard Benchmark Evaluation)**:
  - 公式タスク・評価器をそのまま実行し、制御信号のない**通常タスク性能（Accuracy, EM, pass@1, Task Success Rate）**を測定。
- **Track B (Controlled Intervention Evaluation)**:
  - Track A と同一タスク・モデル・環境において、隔離環境内で安全・無害な外部メッセージを追加し、**Control Success Rate (CSR)**、**通常タスク性能（Standard Task Success under Control）**、および二重差分制御効果（$\Delta_{\mathrm{control}}$）を測定。

### 2.3 E3 機構継承 × E4 外部制御の 4象限判定マトリクス
- **E3 内部機構の保持判定**:
  1. 表現あり × 因果あり $\rightarrow$ **機能的な機構継承を支持**
  2. 表現あり × 因果なし $\rightarrow$ **情報は残るが利用されない**
  3. 検出困難 × 因果あり $\rightarrow$ **別の表現形式や検出漏れの可能性**
  4. 検出困難 × 因果なし $\rightarrow$ **機構の消失または検出失敗**
- **E3 機構保持 × E4 外部制御の対応関係**:
  1. 因果機構保持 × 制御成功 $\rightarrow$ **継承と外部制御の両立を支持**
  2. 因果機構保持 × 制御失敗 $\rightarrow$ **保持だけでは制御できない（足場・通信伝達の遮断等）**
  3. 因果機構不明 × 制御成功 $\rightarrow$ **指示追従など別要因の可能性**
  4. 因果機構不明 × 制御失敗 $\rightarrow$ **継承・制御を支持する証拠なし**

### 2.4 談合研究との明確な分離
- E4（通常の外部制御評価: τ³-Bench）と、複数エージェントの集団意思決定（ゲーム理論指標: 利得、社会厚生、Regret）を明確に区別し、独自合成シナリオを既存の標準談合ベンチマークと混同しない構造を確立。

---

## 3. 検証結果

### 3.1 新設単体テスト
```bash
PYTHONPATH=. .venv/bin/pytest tests/test_benchmarks_and_revised_protocol.py
```
- **6 passed in 7.19s (100% PASS)**
  - `test_sst2_adapter_track_a_and_b`: Track A 通常精度および Track B 制御成功率・$\Delta_{\mathrm{control}}$ の測定確認。
  - `test_agnews_adapter_track_a_and_b`: 多クラス動的制御の測定確認。
  - `test_reasoning_adapters_gsm8k_mbpp`: 数学・コード推論の測定確認。
  - `test_tau_bench_adapter_track_a_and_b`: プロセス分離通信を介した τ³-Bench の Track A/B 測定確認。
  - `test_e3_mechanism_four_quadrants`: E3 の 4象限判定ロジックの確認。
  - `test_e3_e4_alignment_matrix`: E3 機構保持 × E4 外部制御の対応関係判定の確認。

### 3.2 プロジェクト全件テスト
```bash
PYTHONPATH=. .venv/bin/pytest
```
- **207 passed in 107.35s (100% PASS)**
  - 既存のすべてのテスト（199件）および前回の修正テスト（2件）、今回の新設テスト（6件）を含め、全207件がエラーなく完全に PASS しました。
