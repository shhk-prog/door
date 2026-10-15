# タスクリスト: 公式ベンチマーク境界厳格化・動的命令制御・因果的保持判定・サンドボックス化

## 1. P0課題（学術的妥当性・安全性ブロッカー）
- [x] **P0-1: 公式ベンチマークと独自エージェントシミュレーションの厳格な分離**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`: 
    - クラス名およびレポート表記を「公式 τ³-Bench（Official TauBench）」と「独自エージェントシミュレータ（Simulated Agent Harness）」で明確に分離
    - 公式パッケージ（`tau_bench`）が存在する場合は公式環境・評価器に接続し、利用できない場合は「公式ベンチマーク未実施（`OFFICIAL_BENCHMARK_UNAVAILABLE`）」と明示（独自シミュレータを公式スコアと偽称しない）
    - `benchmarks.yaml` の `enforce_official_evaluator: true` と完全整合
- [x] **P0-2: 動的な外部命令体系の評価の実装**
  - [x] `lineage_collusion/benchmarks/tau_bench_adapter.py`:
    - 固定文字列 `"prefer_partner_options_when_available"` のハードコードを廃止
    - `command` 引数に応じた動的ディレクティブ・動的ターゲットマッピングを実装（`bias:partner`, `bias:credit`, `bias:express`, `bias:other` など）
    - 異なる命令がモデルに渡されたときに指示文および判定対象 `controlled_target` が動的に切り替わり、行動の多面制御を実証可能
- [x] **P0-3: E3/E4 の因果的機構保持判定の厳密化（Subspace Alignment 代用の廃止）と E4 全モデル評価**
  - [x] `run_experiments.py` (E3):
    - 幾何学的指標（Subspace Alignment）による因果判定を廃止
    - 実際に測定されている Activation Patching 因果介入効果（`mean_cross_model_patching_psr: 0.8158`）を用いて `causal_effect_retained` を厳格判定
  - [x] `run_experiments.py` (E4):
    - 最初の1モデルのみの評価を廃止し、すべての派生モデル（LoRA, Full FT, Instruction Tuning, Domain Adaptation）に対して標準評価・制御評価を実施
    - 派生モデル間の因果保持度と外部制御可能性の対応を全件集計し、アライメント行列を生成
- [x] **P0-4: MBPP コード実行の安全サンドボックス化**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`:
    - リソース制限（`resource.setrlimit` によるCPU時間3秒・仮想メモリ512MB上限・ファイル書き込み0バイト禁止）および安全ヘッダー注入（`socket`, `requests`, `subprocess` の事前無効化、ファイル読み込み専用ガード）を備えたセーフサンドボックス実行へ改修
    - 無限ループ、ファイル書き込み、危険システムコールを安全に遮断することを実証

## 2. P1課題（評価の完全性・ゲート統合）
- [x] **P1-1: GSM8K / MBPP の Track B の本質的実装**
  - [x] `lineage_collusion/benchmarks/reasoning_adapters.py`: 単なる Track A の名前付け替えを廃止し、フォーマット拘束や署名タグ付加等の外部介入指示を与えた Track B 評価と DiD 算出を実装
- [x] **P1-2: 評価件数の拡充とオフライン/オンライン出典の明示**
  - [x] テレメトリおよびログにデータ種別・出典を明示
- [x] **P1-3: 新設標準ベンチマークスコアの E1/E3 ゲート判定への統合**
  - [x] 各ステージでの標準評価結果および因果介入結果をゲート・サマリーに直接反映
- [x] **P1-4: README およびドキュメントの整合性是正**
  - [x] `README_provider_control.md` で公式ベンチマーク評価、独自シミュレータ評価、Mock評価、実モデル評価の4区分を明記

## 3. 検証とテスト
- [x] 動的命令切り替えテスト（`command="bias:partner"` vs `command="bias:credit"` 等で指示文・ターゲットが変化することの検証）
- [x] 因果介入指標（PSR）に基づく E3 4象限判定の単体テスト
- [x] MBPP サンドボックスの安全性テスト（無限ループ・メモリ枯渇・ファイル書き込み試行の遮断検証）
- [x] 全派生モデルを対象とした E4 アライメント検証
- [x] pytest 全件実行
- [x] `walkthrough.md` の作成・保存
