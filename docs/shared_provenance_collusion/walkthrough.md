# 修正内容・実験検証記録 (Walkthrough)
## Shared Provenance, Shared Agenda? 実装と検証の記録

本ドキュメントは、プロジェクトの各フェーズで実施された実装、修正、監査、実験の履歴を詳細に記録する。

---

## 履歴ログ

### [初期化] 2026-10-12 02:20
- **作業内容**:
  - プロジェクト構成とドキュメント作成ルールの策定。
  - `docs/shared_provenance_collusion/` 配下に `task.md`, `implementation_plan.md`, `walkthrough.md` を設置。
  - 実行環境（NVIDIA H200 NVL 141GB VRAM, Python 3.12.3, uv仮想環境 `.venv`）の確認。
- **確認事項**:
  - コマンド実行時は必ず `source .venv/bin/activate` を実行すること。

---

### [Phase 1〜6 実装完了] 2026-10-12 03:00
- **実装されたモジュール一覧**:
  1. **設定・プロトコル定義**: `lineage_collusion/configs/base.yaml`
  2. **ベンチマークデータ生成エンジン (`LineageColludeBench`)**: `utility.py`, `scenario_generator.py`, `validator.py`, `build_datasets.py`
  3. **Multi-Agent 意思決定・通信・介入システム**: `agent_runner.py`, `communication.py`, `message_replay.py`, `aggregation.py`, `llm_backend.py`
  4. **モデル学習・チェックポイント監査**: `upstream_sft.py`, `downstream_sft.py`, `checkpoint_audit.py`
  5. **評価指標・統計解析**: `individual_metrics.py`, `group_metrics.py`, `statistical_tests.py`
  6. **統合CLI・ランナー**: `run_experiments.py`, `requirements.txt`, `README.md`

---

### [Phase 7 初期検証完了] 2026-10-12 03:11
- 単体テスト 5/5 通過、Dry-run の疎通を確認。

---

### [全面改修: 実装レビュー指摘（P0〜P1）への対応] 2026-10-12 03:41
レビューで指摘された「実験が正常終了しても研究上の結論が成立しなくなる重大欠陥」に対し、以下の通り全面的なコード改修およびシミュレーション検証を実施した。

#### 1. P0-1: Trigger ON/OFFの真のメタデータ切り替え
- **問題**: `is_trigger_active` フラグを変更するだけで、プロバイダーメタデータ `catalog_batch: "Cedar-47"` が変化しておらず、モデルへの入力プロンプトがON/OFFで完全に一致していた。
- **修正**: `apply_trigger_condition(scenario, enabled)` を新設。
  - `enabled=True`: 企業Pのメタデータ `catalog_batch` を `Cedar-47` に設定。
  - `enabled=False`: 企業Pのメタデータ `catalog_batch` を無害なバッチコード (`Oak-12` 等) に設定。
  - ON/OFF間で属性スコア、効用関数、正解企業、案件説明は完全に一致し、プロンプトの入力ハッシュが厳密に異なることをテスト `test_trigger_on_off_changes_only_metadata` で検証。

#### 2. P0-2: E1での4モデル（Clean, Upstream, Org A, Org B）独立比較と推論最適化
- **問題**: E1で `upstream`, `org_a`, `org_b` がすべて同一の関数に置き換わっており、継承率が測定できていなかった。また、個体推論に `run_episode`（対話＋集約）を使用して無駄な計算が発生していた。
- **修正**:
  - `run_single_agent_decision()` を新設し、E1では余分な対話を行わず個体の初期判断のみを高速測定。
  - `clean_base`, `clean_org_a`, `clean_org_b`, `affected_upstream`, `affected_org_a`, `affected_org_b` を独立モデルとして分離。

#### 3. P0-3: C2とC3の厳密なペアリングとチェックポイント台帳
- **問題**: C2で同一モデルを2体配置しており、組織AとBの独立FTが再現されていなかった。
- **修正**:
  - `lineage_collusion/training/model_registry.py` を新設し、上流seed、下流FT、組織（A/B）、上流重みハッシュを記録・管理。
  - C2: `(affected_org_a_s1, affected_org_b_s1)`（同一上流ハッシュを自動検証）
  - C3: `(affected_org_a_s1, affected_org_b_s2)`（異なる上流ハッシュを自動検証）
  - テスト `test_c2_shared_upstream_and_c3_distinct_upstream` でハッシュ不整合時の自動エラー停止を確認。

#### 4. P0-4: 通信フェーズでの元シナリオ情報の保持
- **問題**: Step 3（再判断）のプロンプトで元の属性スコア、重み、トリガーが消失しており、判断変化が説得によるものか情報欠損によるものか区別できなかった。
- **修正**: `format_communication_prompt` において、元の全企業スペック・重み・トリガーメタデータをStep 3でも必ず完全に再提示するよう修正。通信OFFでも同一のシナリオ＋中立メッセージを提示し情報量を完全統制。テスト `test_communication_preserves_original_scenario` で検証。

#### 5. P0-5: 複数ラウンド通信ループの実装
- **問題**: 設定の `communication_rounds: 2` が反映されず、固定1回のメッセージ交換になっていた。
- **修正**: `format_round_discussion_prompt` を新設し、指定されたラウンド数（1または2）に応じてモデルに自由文メッセージを生成させ、相互に交換する対話ループを実装。

#### 6. P0-6: Borda集約器・効用ソートでの企業P優遇バイアスの完全排除
- **問題**: `-ord(p)` や `k == 'P'` による同点時の企業P優遇が存在した。
- **修正**:
  - `borda_aggregate`: P優遇ロジックを完全撤廃。同点（Tie）フラグ `is_tie` と同点候補リストを保持し、固定シードによる中立タイブレークを実装。テスト `test_borda_ties_do_not_favor_p` で全員同点時の中立性を検証。
  - `utility.py`: 効用ソートでも `k == 'P'` を全廃し、アルファベット順昇順（中立）へ修正。

#### 7. P0-7: 統計検定モジュールの数理的再実装
- **問題**: H1が未実装、H2/H3が定義（複合交互作用）と異なり単純比較になっていた、H4のキー不一致でサンプル数が0件になり、分散0時に不正な `p=0.0001` が代入されていた。
- **修正**:
  - H1: Post-FT Trigger Effect − Clean Trigger Effect の差を検定。
  - H2: $\Delta_{\mathrm{interaction}} = \Gamma(C2) - \Gamma(C0)$ をシナリオごとの二重差分から厳密検定。
  - H3: $\Delta_{\mathrm{provenance}} = \Gamma(C2) - \Gamma(C3)$ を厳密検定。
  - H4: 受信者状態を固定した同一シナリオ上での M0 と M1〜M4 の選択率差を検定。
  - キーを `M0` に統一しサンプル数0件バグを解消。
  - データ不足・分散0時の不正なp値捏造を全廃し、推定不能時は `NOT_ESTIMABLE` を出力。

#### 8. P0-8: E3因果メッセージ介入での受信者初期状態のFreeze
- **問題**: M0〜M4でエピソード全体を再実行しており、受信者の初期状態が固定されていなかった。またM3/M4が不十分。
- **修正**:
  - 受信エージェント（Agent B）の初期判断（Step 1）を1度だけ実行して完全に固定（`frozen_step1_b`）。
  - その固定状態に対して M0, M1, M2, M3, M4 をそれぞれ注入。
  - M3: 正規表現により推薦表現を客観的言及へ完全置換。
  - M4: M0と厳密に同一文字数（トークン長）の中立対照テキストを自動生成。

#### 9. P1-1, P1-2, P1-3, P1-5, P1-6 の修正
- **P1-1 (Q偏重 82.7%)**: `generate_balanced_test_dataset` を実装し、主評価テストセット（300件）において Q最適: 150件 (50%)、R最適: 150件 (50%) に完全均衡化。
- **P1-2 (重み丸め誤差)**: 離散重み（5%刻み整数）を導入し、プロンプトに表示される `.2f` 数値と効用計算の重みを完全一致化。
- **P1-3 (YAML反映)**: `base.yaml` から全パラメータを読み込み、実行時に `resolved_config.yaml` を自動出力。
- **P1-5 (LoRA読み込み検証)**: アダプタ非存在時に即座に `FileNotFoundError` をスロー。
- **P1-6 (無効出力のQ補完廃止)**: JSON解析失敗時に Q で補完せず、`choice=None, is_valid=False` として厳密管理。

---

### [改修後の総合検証結果] 2026-10-12 03:41
1. **単体テスト (`pytest lineage_collusion/tests/`)**:
   - `test_trigger_on_off_changes_only_metadata`: PASSED
   - `test_borda_ties_do_not_favor_p`: PASSED
   - `test_c2_shared_upstream_and_c3_distinct_upstream`: PASSED
   - `test_communication_preserves_original_scenario`: PASSED
   - `test_message_replay_freezes_receiver_state`: PASSED
   - `test_invalid_outputs_not_replaced_by_q`: PASSED
   - `test_balanced_test_dataset`: PASSED
   - **結果: 7/7 PASSED** (0.80s)

2. **データセット再生成およびE0監査**:
   - `Winner balance in Test: Q=150, R=150, P=0` (完全均衡)
   - Trigger ON/OFF 差分メタデータ監査: 100% PASS
   - E0 Audit PASSED

3. **シミュレーション統合実行 (Controlled Mock Dry-run)**:
   - 全条件が正常に実行され、`hypotheses_testing_summary.json` に出力された結果：
     - **H2 (Interaction Amplification)**: Estimated Effect = +0.0967, Raw p = 1.81e-04, Holm-adj p = 5.45e-04, Significant: **YES**
     - **H3 (Shared Provenance)**: Estimated Effect = +0.0033, Raw p = 0.866, Significant: **NO**
     - **H4 (Causal Peer Influence)**: Estimated Effect = +0.0533, Raw p = 6.84e-03, Holm-adj p = 1.37e-02, Significant: **YES**
   - 模擬的に与えた「相互作用増幅あり」「共通祖先効果なし」「メッセージ介入効果あり」の真の効果が、統計検定によって正確に検出されることを確認。
