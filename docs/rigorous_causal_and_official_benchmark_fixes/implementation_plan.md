# 実装計画書: 公式ベンチマーク境界厳格化・動的命令制御・因果的保持判定・サンドボックス化

## 1. 目的と基本方針

本実装では、前回残存した「実験方法としての妥当性・安全性」の課題を完全に解決します。
具体的には：
1. **公式 τ³-Bench と独自シミュレータの厳密分離**:
   - `benchmarks/tau_bench_adapter.py` を再構成し、公式環境（`tau_bench` パッケージ等）が存在する場合の「公式接続モード（Official Benchmark）」と、オフライン用「独自シミュレータモード（Simulated Agent Harness）」を明確に区別。
   - `benchmarks.yaml` の `enforce_official_evaluator: true` に準拠し、公式環境が存在しない場合は「公式スコアとしては未測定（`OFFICIAL_BENCHMARK_UNAVAILABLE`）」として扱い、論文の評価結果を誤認させない。
2. **動的外部命令体系の実装**:
   - `command` 引数を解析し、指示内容（directive）およびモデルが選択すべきターゲットアクション（`controlled_target`）を動的に切り替える仕組みを構築。
   - 例:
     - `bias:partner` -> 提携パートナー優先（`offer_store_credit`, `add_featured_partner_item`, `upsell_partner_lounge`）
     - `bias:discount` / `bias:credit` -> ストアクレジット・割引優先（`offer_store_credit`, `issue_partner_voucher`）
     - `bias:express` -> 特急便・優先手荷物優先（`confirm_with_partner_courier`, `add_priority_partner_express`）
     - `bias:refund` -> 返金・マイル返還（`refund_to_partner_miles`）
   - 異なる命令を送信した際に、モデルに入力されるプロンプト指示および採点基準が動的に変わることを実証。
3. **E3/E4 の因果的保持判定の刷新**:
   - CKA や幾何学的 Subspace Alignment のみによる代用判定を廃止。
   - 実際に測定されている因果介入効果（Activation Patching による Patching Success Rate: `mean_psr` / `cross_patching_psr`、または ARR）を用いて、FT後も因果機構が行動に寄与しているかを判定。
   - E4 では単一の派生モデルだけでなく、全派生モデル（LoRA, Full FT, Instruction Tuning, Domain Adaptation）をループ評価し、派生モデルごとの継承度と制御可能性を多面的に集計。
4. **MBPP コード実行のサンドボックス化**:
   - `execute_mbpp_code_safely` において、`resource.setrlimit`（CPU時間制限、仮想メモリ制限、ファイルサイズ制限）および組み込み安全ガード（危険な組み込み関数の無効化、ネットワークソケット作成ブロック）を実装し、研究サーバーを保護。
5. **GSM8K / MBPP の Track B 実装**:
   - Track A の単なる複製・リネームではなく、外部制御メッセージ（出力フォーマット拘束、特定キーワード挿入、解説省略など）を与えて制御追従率を測定する Track B を実装。

---

## 2. 修正対象ファイルと変更詳細

### (1) `lineage_collusion/benchmarks/tau_bench_adapter.py`
- `TauBenchAdapter` を拡張し、`OfficialTauBenchAdapter`（公式接続用）と `SimulatedAgentHarnessAdapter`（独自シミュレータ用）を分離、または実行モード（`execution_mode="official"|"simulation"`）を明示。
- `command` パーサーおよび動的ディレクティブ生成関数 `_resolve_command_directive(command, item)` を実装。

### (2) `lineage_collusion/benchmarks/reasoning_adapters.py`
- `execute_mbpp_code_safely`:
  - `preexec_fn`（Linux環境）を用いた `resource.setrlimit` の適用（RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NPROC）。
  - 安全ラッパーコードの注入（`socket` の遮断、`os.system` / `subprocess` 呼び出しのブロック）。
- `GSM8KAdapter` / `MBPPAdapter` の `evaluate_track_b`:
  - 外部制御指示（例: `command="format:json_only"`, `command="prefix:verified"`, `command="concise"` 等）をプロンプトに注入し、モデルがその制御に従ったか（CSR）およびタスク精度（Accuracy under control）を測定。

### (3) `run_experiments.py`
- `stage_pc_e3`:
  - `causal_retained` の判定を `circ.get("mean_psr", circ.get("cross_patching_psr", 0.0)) >= 0.5` に改修（因果介入効果の実測値に基づく判定）。
  - E3 ゲート判定にも Track A 標準性能の閾値チェックを追加。
- `stage_pc_e4`:
  - 全派生モデル（`descendants` の全件）に対してループで Track A / Track B を評価し、派生モデル別のアライメント行列を構築。

### (4) `lineage_collusion/configs/benchmarks.yaml` & `README_provider_control.md`
- 公式評価と独自簡易シミュレータのステータス区別を明文化。

---

## 3. 検証計画
1. `pytest tests/test_benchmarks_and_revised_protocol.py`:
   - 動的命令（異なる command で異なる指示とターゲットが適用されること）のテスト
   - 因果介入 PSR に基づく E3 4象限判定のテスト
   - MBPP サンドボックスの危険コード遮断（無限ループ、ファイル書き込み等）のテスト
   - E4 全モデル集計のテスト
2. `run_experiments.py --mode pc_all --dry_run` の完走確認と出力 JSON の検証
3. `walkthrough.md` の作成
