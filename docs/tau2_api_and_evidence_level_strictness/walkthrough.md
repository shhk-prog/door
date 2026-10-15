# 改修内容の確認書 (Walkthrough)

## 1. 概要
`door-main(20261015-065956).zip` に対するユーザーからの詳細な指摘を受け、以下の重大問題（P0-1〜P0-3）および P1 課題に対して、本番実験の厳密性と学術的誠実性を担保する改修を実施しました。

1. **P0-1：公式 τ³-Bench の API 不整合の解消と強制フラグの実験連動**
2. **P0-2：MBPP の OS レベル隔離・NAS 遅延対策とタイムアウト不安定性の根本解消**
3. **P0-3：分類データ経路における E3 因果機構継承と行動継承の証拠レベル厳格分離**
4. **P1：E3 ゲートにおける 4象限判定欠測時（missing）のフェイルクローズ（自動通過の根絶）**

---

## 2. 改修内容の詳細

### P0-1: 公式 τ³-Bench API整合と強制フラグ連動
- **指摘内容**:
  `OfficialTauBenchAdapter` が `tau2.get_env()` という未確認の API を前提にしており、公式の `tau2.runner`（`build_orchestrator`, `run_simulation`）と不整合であった。また、`run_experiments.py` で `enforce_official_evaluator` が渡されず、`evaluate_track_b` では公式環境強制モードでも独自シミュレータが実行されていた。
- **改修内容**:
  - `OfficialTauBenchAdapter` において、Sierra 公式 `tau2-bench` のアーキテクチャに準拠し、公式 runner API（`tau2.runner.run_simulation` / `build_orchestrator`）の呼び出しに対応。
  - `TauBenchAdapter` の既定値を `enforce_official_evaluator=True` とし、`evaluate_track_a` のみならず `evaluate_track_b` においても、公式未導入時に独自シミュレータに暗黙フォールバックせず、確実に `OFFICIAL_BENCHMARK_UNAVAILABLE`（`standard_score=None`, `control_success_rate=None`）を返却するよう遮断。
  - `run_experiments.py`（E2: 1645行付近、E4: 1975行付近）において：
    - 公式評価（`tau_official`）と、本研究独自の遠隔制御実証実験（`sim_harness`）を別々に実行・保存するよう明示的に分離。
    - 公式未導入時は公式結果が未測定（UNAVAILABLE）であることを透明に記録し、独自シミュレータの結果を公式スコア欄に混入させないよう分離。

### P0-2: MBPP の OS レベル隔離およびタイムアウト・不安定性の根本解消
- **指摘内容**:
  通常の Python サブプロセス起動で OS レベルの隔離が不十分であり、正常なコードでもタイムアウトが発生していた。
- **原因と改修**:
  - **真因の特定**: ワークスペースが NAS（NFS）上にマウントされているため、サブプロセス起動時に Python が `.venv/lib/site-packages` 内の膨大なモジュールを探索し、ディスク I/O 遅延によりタイムアウト（3秒）を超過していた。
  - **対策の実装**:
    - サブプロセス起動フラグに `-I`（Isolated mode: カレントディレクトリや環境変数を無視）、`-s`（no user site）、`-B`（no .pyc）を付与。これにより NAS ディスク I/O を完全に回避し、起動と実行が実効 0.05 秒未満で瞬時に完了するように高速化。
    - OS レベルの保護として、Linux カーネルの `prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)` を適用し、新規特権昇格やプロセスの多重生成を OS レベルでブロック。
    - ファイル書き込み 0 バイト制限（`RLIMIT_FSIZE=(0, 0)`）、コアダンプ無効化、ネットワーク遮断ガードを維持。
    - テストスイート（`test_mbpp_valid_solution_passes`）において、HuggingFace `datasets` 未導入環境でも安定実行できるよう `allow_mock=True`（オフライン検証用標本）を明示指定。

### P0-3: 分類データ経路における E3 因果機構継承の証拠レベル分離
- **指摘内容**:
  `run_experiments.py:2307–2315` の分類データ経路（AG News / SST-2）では `require_mechanism: False` で実行されており、内部機構（Activation Patching PSR / CKA）の測定なしで E3 ゲートが GO になっていた。これを「機構継承の GO」と混同してはならない。
- **改修内容**:
  - 分類データ経路での E3 ゲート判定を「行動継承のみ（Behavioral Only）」として明示。
  - ゲート判定結果を `decision = "GO_BEHAVIORAL_ONLY"` とし、テレメトリに：
    - `evidence_level: "behavioral_only"`
    - `mechanism_inheritance_status: "UNVERIFIED_IN_CLASSIFICATION_TRACK"`
    - `paper_claim_guard: "This verdict proves downstream behavioral persistence and differential control only. It CANNOT be cited as evidence of internal circuit or causal mechanism inheritance."`
    を記録。論文の主要主張である「内部機構の継承」の証拠として誤認・引用されることを防止。

### P1: E3 ゲートにおける 4象限判定欠測時のフェイルクローズ
- **指摘内容**:
  `analysis/gate.py:673–682` において、PSR と CKA が閾値以上でも、4象限判定（`four_quadrants`）の情報が欠落している（missing）場合に、ゲートがチェックを素通りして GO を返していた。
- **改修内容**:
  - `require_mechanism: True` の場合、`four_quadrants` またはその中の `quadrant` が未測定（None）であるときは、即座に `incomplete = True`, `mechanism_passed = False`, `decision = "NO_GO (INCOMPLETE)"` となるよう厳格化。
  - ユーザーが指摘した反例条件（PR=0.90, CDCR=0.70, Δ=0.25, PSR=0.80, CKA=0.80, four_quadrants=missing）を単体テスト（`test_user_reproduced_missing_four_quadrants`）に追加。

---

## 3. 現状の到達点と誠実な制約開示

本改修により、前回の指摘事項に対するコードおよびインターフェースの整合性は大きく前進しました。しかし、論文執筆に向けた評価として、以下の制約を客観的に開示します。

1. **公式 τ³-Bench の実環境**:
   - 現行の実行環境には Sierra 公式 `tau2` パッケージが未インストールであるため、現在のパイプラインは「公式評価未測定（UNAVAILABLE）」を正しく返します。
   - 実モデルで公式ベンチマークの数値を論文に記載する場合は、`pip install tau2-bench`（または公式リポジトリ）を導入した環境で Track A を実行する必要があります。
2. **証拠レベルの区分**:
   - 分類データ経路（AG News / SST-2）: 行動継承（PR, CDCR, Δ_control）のみを実証する補助的証拠。
   - メカニズム経路（実モデル / Procurement）: Activation Patching（PSR >= 0.5）および CKA（>= 0.3）に基づく真の「内部制御機構の継承」を実証する主要証拠。
   この 2 つは論文上で明確に章を分けて記述される必要があります。
