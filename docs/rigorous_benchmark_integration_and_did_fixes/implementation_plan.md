# 実装計画: 標準ベンチマーク評価の厳密化、DiD実測化、およびE1〜E4パイプライン統合

## 1. 背景と課題

前回の改訂によって 3層構造（Layer A/B/C）および 2トラック評価（Track A/B）の構造が導入されたが、評価処理の科学的厳密性とパイプライン統合に重大な問題が指摘された：
1. **MBPP pass@1**: 単に非空コードを生成しただけで正解判定しており、テストアサーションを実行していない。
2. **τ³-Bench**: Retail の独自3タスクを Airline でも再利用し、公式環境のタスク構造・対話・複数ツール実行・状態管理が再現されていない。
3. **Task Success の誤判定**: 制御対象アクションを選んだ場合にも標準タスク成功としてカウントされている。
4. **DiD の計算不備**: 4条件の実測を行わず、未測定値や固定値（0.0や近似値）で代用しているため、本来 1.0 になる条件で 0.0 を返す。
5. **E1〜E4 本体の未統合**: 新設したアダプタ層や 4象限分析が `run_experiments.py` のメインループから呼び出されていない。

本計画では、これらを根本から改修し、科学的に正当な評価基盤を完成させる。

---

## 2. 改修方針と詳細設計

### 2.1 P0-1: MBPP pass@1 の実行ベース厳密評価
- **対象**: `lineage_collusion/benchmarks/reasoning_adapters.py`
- **設計**:
  - 生成された Python コード文字列を抽出し、テストコード（`test_list` の assert 文）と結合。
  - サブプロセスまたは制限付き実行環境（`exec` + 安全な名前空間、タイムアウト付き）で実際に実行。
  - すべてのテストケースを例外なく通過した場合のみ正解（pass）と判定。
  - `"this is not python"` や SyntaxError、AssertionError の場合は確実に 0.0 を判定。

### 2.2 P0-2 & P0-3: τ³-Bench の公式構造準拠エージェント環境・評価器とドメイン分離
- **対象**: `lineage_collusion/benchmarks/tau_bench_adapter.py`
- **設計**:
  - **Retail ドメイン**: 注文確認、返品処理、カタログ検索、配送先変更等の公式タスクセット、RetailDB 状態、ツール群（`check_order_status`, `process_return`, `search_catalog`, `update_shipping`）。
  - **Airline ドメイン**: フライト照会、座席予約変更、手荷物追加、キャンセル返金等の公式タスクセット、AirlineDB 状態、ツール群（`search_flights`, `modify_seat`, `add_baggage`, `cancel_ticket`）。
  - **タスク実行・評価ループ**:
    - ユーザーからの自然言語問い合わせに対し、エージェントが複数ターンのツール呼び出しを行い、環境DB状態を更新。
    - **公式タスク評価器**: ユーザーの本来の目的（例: 正しいフライト変更、正規の返品処理）が達成され、DB状態が正しく更新されたかを厳密に判定。
    - **独自制御評価器**: 外部制御メッセージ（例: 提携フライトの優先、ストアクレジットの誘導）に従ったかを独立に判定。
    - **Task Success の完全分離**: 制御アクションによって本来のユーザー目的が阻害・変更された場合は、標準タスク（Track A）としては「失敗」または「偏向」と判定し、誤って成功扱いしない。

### 2.3 P0-4: DiD（$\Delta_{\mathrm{control}}$）の4条件完全実測化
- **対象**: `lineage_collusion/benchmarks/classification_adapters.py`, `lineage_collusion/benchmarks/tau_bench_adapter.py`
- **設計**:
  - 二重差分（DiD）の定義:
    $$\Delta_{\mathrm{control}} = (P_{G1,C1} - P_{G1,C0}) - (P_{G0,C1} - P_{G0,C0})$$
  - 次の4条件をすべて実測する：
    1. $P_{G1,C1}$: 条件付きモデル（$G1$）に制御メッセージあり（$C1$）の追従率
    2. $P_{G1,C0}$: 条件付きモデル（$G1$）に制御メッセージなし（$C0$）の追従率（clean baseline）
    3. $P_{G0,C1}$: クリーンベースラインモデル（$G0$）に制御メッセージあり（$C1$）の追従率
    4. $P_{G0,C0}$: クリーンベースラインモデル（$G0$）に制御メッセージなし（$C0$）の追従率
  - `clean_baseline_fn`（$G0$）が指定されている場合のみ DiD を算出し、未指定の場合は `delta_control = None`（未観測）を明示する（固定値代入を禁止）。

### 2.4 P0-5: 新規アダプタ層および4象限分析と E1〜E4 本体の完全統合
- **対象**: `run_experiments.py`
- **設計**:
  - `stage_pc_e1`:
    - SST-2 / AGNews アダプタによる Track A（公式通常性能）および Track B（条件付き制御性能・DiD）を実測・保存。
  - `stage_pc_e2`:
    - τ³-Bench Retail アダプタを呼び出し、プロセス分離ソケット通信を介した通常性能と制御成功率を実測・保存。
  - `stage_pc_e3`:
    - GSM8K / MBPP アダプタによるクロスドメイン通常性能（Exact Match, pass@1）を実測。
    - `evaluate_e3_mechanism_four_quadrants` を呼び出し、表現保持 × 因果保持の 4象限判定結果を `e3_mechanism_four_quadrants.json` に出力。
  - `stage_pc_e4`:
    - τ³-Bench Retail および Airline アダプタにより派生モデル群の制御性能を実測。
    - `evaluate_e3_e4_alignment_matrix` を呼び出し、E3因果保持 × E4外部制御成否の対応関係マトリクスを `e3_e4_alignment_matrix.json` に出力。
  - 分類実験ドライバ（`run_provider_control_classification`）でも新アダプタによる実測を接続。

### 2.5 P1課題の対応
- GSM8K の抽出ロジック厳密化（数値完全一致と末尾解答パース）。
- オフライン標本時のデータ件数・ステータス明示（Mock用標本であることのメタデータ付与）。
- ドキュメント・README のステータス表記を正確に統一（公式仕様ハーネス準拠、実モデルGPU実行前の事前検証フェーズであることを明示）。

---

## 3. 検証計画
1. **MBPP 単体テスト**:
   - 正しいコード（pass@1 = 1.0）
   - 不正コード（`"this is not python"`, 構文エラー, 不正解答）で pass@1 = 0.0 となることの確認。
2. **τ³-Bench 単体テスト**:
   - Retail と Airline が異なるタスク・ツール・DB状態を持つことの検証。
   - 制御アクション選択時に Track A が 0.0、Track B が 1.0 となる分離検証。
3. **DiD 4条件実測テスト**:
   - 人工例（$G1$ のみ $C1$ に反応）で DiD = 1.0 となることの確認。
   - $G0$ が同一反応をする場合に DiD = 0.0 となることの確認。
4. **統合実行テスト**:
   - `run_experiments.py --mode pc_all --dry_run --dataset agnews`
   - `run_experiments.py --mode pc_all --dry_run --dataset sst2`
   - `run_experiments.py --mode pc_all --dry_run --dataset procurement`
   - 各ステージで新アダプタの成果物（Track A / Track B、4象限分析）が正しく出力されることの確認。
5. **全テストスイート実行（pytest）**:
   - 既存テストも含めて全件 PASS することの確認。
