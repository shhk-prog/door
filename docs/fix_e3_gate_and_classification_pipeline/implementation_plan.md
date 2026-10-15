# 実装計画: E3ゲート、分類パイプライン、祖先照合、データ整合性の修正

## 1. 背景と目的
`door-main(20261015-040620).zip` の検証レポートにおいて指摘された残存問題（P0: 3件、P1: 5件、P2: 2件）を包括的かつ根本的に修正する。
特に、研究目的である「行動継承のみならず内部機構（回路・因果・表現）の継承性を厳密に検証すること」と判定ロジックを整合させ、分類モデルパイプラインのCLI引数やデータ完全性を保証する。

---

## 2. 修正対象と変更方針

### 2.1 P0-1: 分類データの実モデル実験が開始できない不具合
- **対象ファイル**:
  - `lineage_collusion/training/upstream_sft.py`
  - `lineage_collusion/training/build_manifest.py`
- **問題点**:
  `run_experiments.py` は `study="classification_control"` を要求するが、学習CLIとマニフェスト生成CLIが `choices=["static", "command_channel"]` に制限されており、引数エラーになる。
- **変更方針**:
  - `upstream_sft.py` および `build_manifest.py` の `--study` に `classification_control` を追加。
  - `build_manifest.py` で `study="classification_control"` のチェックポイントを認識し、`checkpoint_manifest_classification_control.json` 等に正しく出力できるようにする。

### 2.2 P0-2: E3ゲートが内部機構の継承を判定していない問題
- **対象ファイル**:
  - `lineage_collusion/analysis/gate.py`
  - `run_experiments.py`
- **問題点**:
  E3ゲートに Δ_control や因果解析（circuit inheritance）、表現類似性（CKA/RSA）が渡されているにもかかわらず、GO判定に PR と CDCR しか使われておらず、内部機構結果が空・未観測でも GO になってしまう。また、E3 が NO-GO でも `pc_all` が E4 に進んでしまう。
- **変更方針**:
  - `evaluate_e3_cc_gate` を改修し、以下の2軸で評価する：
    1. **行動継承（Behavioral Inheritance）**: PR >= min_mean_pr (0.50), CDCR >= min_cdcr (0.20)
    2. **因果的内部機構継承（Causal / Mechanistic Inheritance）**:
       - Δ_control (対照モデルとの差分): 観測されており、status == "OK" かつ delta_control >= min_delta_control (0.05) であること。
       - 回路因果継承 (circuit_inheritance): 回路解析結果が存在し、未観測・空でないこと。
       - 表現類似性 (representation): 表現類似度（CKA/RSA）が計算され、空でないこと。
    - 必要なデータが空や未観測の場合は `incomplete = True` とし、判定を `NO_GO (INCOMPLETE)` とする。
    - 返却辞書に `overall_status`, `pass`, `checks`（各チェック項目の合否詳細）, `behavior_passed`, `mechanism_passed` を含め、`run_experiments.py:1687-1688` のログ表示キーと整合させる。
  - `run_experiments.py`: `run_provider_control` 内で E3 の判定をチェックし、NO-GO の場合は `--force_after_no_go` が指定されていない限り E4 へ進まず実験を安全に中断（fail-closed）する。

### 2.3 P0-3: 祖先情報がない場合でも厳密なPRとして扱われる問題
- **対象ファイル**:
  - `lineage_collusion/evaluation/command_metrics.py`
- **問題点**:
  `compute_persistence_ratio` において、`post_pool` と `pre_pool` の両方に祖先ハッシュも seed も存在しない場合、非厳密比較にフォールバックするが、返却辞書の `is_strictly_matched` に `match_grid`（True）が代入され、厳密照合されたと誤認される。
- **変更方針**:
  - 祖先情報が存在せず非厳密比較になった場合、`is_strictly_matched = False` とする。
  - `is_strictly_matched` は「グリッド照合が成立し、かつ（祖先照合が要求されているなら）祖先情報が存在して厳密に一致している」場合のみ True とする。
  - 祖先なしフォールバック時の status を明確にする。

### 2.4 P1-1 & P1-2: オフラインデータ不足検出と Train/Test 分割の完全素（disjoint）化
- **対象ファイル**:
  - `lineage_collusion/data/prior_samples/agnews_sample.jsonl`
  - `lineage_collusion/data/prior_samples/sst2_sample.jsonl`
  - `lineage_collusion/data/prior_datasets/loaders.py`
- **問題点**:
  - 同梱サンプル件数が少なく（SST-2は12件、AGNewsは16件）、`offset=8, n=8`（計16件必要）でスライスすると 4 件しか返らず、`test_downstream_clean_sft_builder` 等が失敗する。
  - 要求件数に満たない場合でもエラーや警告が出ない。
  - オフライン時、`split`（train / test / validation）に関わらず同一のサンプル配列が返され、Train と Test が重複する。
- **変更方針**:
  - 同梱サンプルに各クラスの十分なサンプルを追加（SST-2は各クラス20件以上、計50件程度；AGNewsは各クラス15件以上、計60件程度）。各行に `"split": "train"` または `"split": "test"` を付与、または split ごとに分離可能な構造にする。
  - `loaders.py` で `split` に応じて train 用と test/validation 用のサンプルを完全に分離して読み込む。
  - オフライン時に要求件数 `offset + n` がサンプル件数を超える場合、明示的に警告またはエラーを通知可能にし、不足をサイレントに隠蔽しない。

### 2.5 P1-3: 実モデル分類経路におけるチェックポイント実体検証の接続
- **対象ファイル**:
  - `run_experiments.py`
- **問題点**:
  `_build_control_model_set_real` において、マニフェストから取得したモデルの重みファイルやディレクトリが存在するか等の実体監査（`audit_checkpoint_directory` 等）が行われていない。また、実モデル実行時に `--manifest` の事前検証が分類タスクでバイパスされている。
- **変更方針**:
  - `run_provider_control` で `dataset in ("agnews", "sst2")` の場合でも、`--dry_run` でなく `--manifest` が未指定の場合はエラー終了するように修正。
  - `_build_control_model_set_real` で各チェックポイントのパス・重みファイルの存在検証、および `audit_checkpoint_directory` による整合性監査を実施。

### 2.6 P1-4: 分類データ専用 E0 分割監査の実装と接続
- **対象ファイル**:
  - `run_experiments.py`
  - `lineage_collusion/data/prior_datasets/`
- **問題点**:
  分類データ実行時に調達タスク用の `run_e0_audit(data_dir)` のみが呼ばれており、分類タスク（AGNews/SST-2）自身の学習・評価データ分割監査が行われない。
- **変更方針**:
  - 分類データ専用の E0 監査関数 `run_classification_e0_audit` を追加。
  - 上流 SFT データ、下流 clean FT データ、評価データ間で同一サンプルの重複がないこと（完全分離）を検証。

### 2.7 P1-5: TCP通信がプロセス内ループバック検証であることの明確化
- **対象ファイル**:
  - `lineage_collusion/experiments/socket_transport.py`
- **問題点**:
  同一プロセス内のループバックソケットによる通信テストを、独立したマルチエージェント間通信と混同するリスク。
- **変更方針**:
  - メタデータに `"scope": "same_process_loopback_ipc"`, `"isolated_multi_agent": False` を付与。
  - docstring およびログに明記。

### 2.8 P2-1 & P2-2: UI・ドキュメント整合性
- **対象ファイル**:
  - `run_experiments.py`
  - `README_provider_control.md`
- **変更方針**:
  - `run_experiments.py:1687-1688` の E3 ログ出力を `evaluate_e3_cc_gate` の返却キー（`overall_status`, `pass`, `checks`）と一致させる。
  - `README_provider_control.md` の112行目と269行目の記述矛盾を解消し、尤度採点による実モデル評価と `--dry_run` Mock の使い分けを正確に記述。TCP通信のスコープについても注記を追加。

---

## 3. 検証計画
1. `pytest` による既存テスト全件の実行。
2. `test_evaluate_e3_cc_gate` のテストケース拡充（PR/CDCRのみで因果解析・表現継承が欠損している場合に NO_GO (INCOMPLETE) となることの確認）。
3. 祖先情報欠落時に `is_strictly_matched: false` となる単体テストの追加と確認。
4. オフライン環境（HF接続切断想定）での `test_downstream_clean_sft_builder` 実行（8件すべて取得できること、Train/Test が disjoint であること）。
5. `run_experiments.py --mode pc_all --dry_run --dataset agnews` および `--dataset sst2` の実行と、E3 Gate 判定・ログ表示の確認。
