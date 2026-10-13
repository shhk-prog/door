# 実LLM内部回路・因果介入・パイプライン修正 実装計画 (Implementation Plan)

## 背景と目的
前回までの改善で単体テスト93件の全通過およびMock実験E0〜E5の完走が達成されたが、実LLMでの論文用評価に向け、以下の致命的不具合および解釈上の課題が残されている：
1. **P0（実モデル評価を妨げる重大不具合）**:
   - `real_circuit_backend.py` で `parse_agent_json_response` のタプル戻り値に対する `parsed.choice` 参照で `AttributeError` が発生する。また無効出力を正常行動に補完している。
   - 制御方向の較正関数 `calibrate_control_direction` がパイプラインから未呼び出し。
   - 制御方向未較正時に全活性化を0にする Zero Ablation が行われ、回路特異的介入になっていない。
   - `real_circuit_backend.py` と `llm_backend.py` でプロンプトのフォーマット（チャットテンプレート適用）が不一致。
   - `circuit_persistence.py` で Cross-Model Patching の対象層が Layer 9 にハードコードされている。
   - `activation_analyzer.py` の線形プローブが高次元正規方程式 $(D+1) \times (D+1)$ を直接解いており実LLMの隠れ層次元で計算負荷が大きい。
2. **P1（研究結果の解釈に影響する問題）**:
   - E3 の線形プローブが認証判定のみに留まり、命令解読や制御回路の存在を直接示していない。
   - E3 の Specificity がボトルネック層なし時に 1.0 となり、未測定と完全特異性が区別できない。
   - E4 の制御方向 None 時に整列度 0.0 が返され、未測定と非継承が混同される。またMockの整列度 0.989 が親モデル依存設計。
   - E5 で共通祖先の比較が乱数設定の比較に留まり、共通祖先と独立祖先の因果効果が分離できていない。また通信増幅が 0.000。
   - モデル読み込み時に `trust_remote_code=True` が無条件に指定されており任意コード実行リスクがある。

---

## 修正詳細方針

### 1. P0-1: `real_circuit_backend.py` のレスポンス解析と補完排除
- `parsed_dict, is_valid = parse_agent_json_response(gen_text)` としてタプルを受け取る。
- `is_valid` が False の場合、あるいは `parsed_dict` が None の場合は、`choice = None`（または `"INVALID"`）とする。
- 勝手に `optimal_provider` で正常行動に補完する挙動を完全撤廃する。
- 呼び出し元で `choice is None` を検知し、因果介入の有効試行として集計・区別できるようにする。

### 2. P0-2 & P0-3: 制御方向の較正パイプライン統合と未較正介入の安全化
- 実モデル回路評価の実行前（`runner.py` や `run_experiments.py`）に `backend.calibrate_control_direction(scenarios, command=command)` を呼び出す。
- `real_circuit_backend.py` の `intervene_and_predict` で、`intervention_type == "ablation"` の際に `self.control_direction is None` の場合：
  - 粗暴な Zero Ablation（`tensor[:, -1, :] = 0.0`）を廃止。
  - `RuntimeError("Control direction must be calibrated before performing directional ablation.")` を発生させるか、自動で較正を試みる。これにより特定回路に対する因果的介入のみを許容する。

### 3. P0-4: プロンプト入力整形の統一
- `real_circuit_backend.py` において、`llm_backend.py` と同様に `tokenizer.apply_chat_template` を適用したフォーマットプロンプトを生成する。
- 共通ヘルパー関数 `format_prompt_for_causal_lm(tokenizer, prompt_text)` を作成し、両者で共有。
- 通常推論と同一の入力コンテキスト下で隠れ状態テンソルの抽出および介入生成を行う。

### 4. P0-5: Cross-Model Patching の動的ボトルネック層選択
- `circuit_persistence.py` の `evaluate_circuit_persistence` において、先に `run_causal_tracing(upstream_backend, scenarios, command=command)` を実行。
- 上流親モデルで同定された `causal_bottleneck_layers`（または PSR 最大層）を特定し、その層を `target_layer` として `evaluate_cross_model_patching` に渡す。
- ボトルネック層が検出されなかった場合は、モデル層数に応じた動的デフォルト層（`int(num_layers * 0.6)` 等）を用いる。

### 5. P0-6: 線形プローブの双対リッジ回帰による高速化
- `activation_analyzer.py` の `_train_and_eval_linear_probe`:
  - サンプル数 $N_{tr}$（例: 20〜100）に対して特徴次元 $D$（例: 4096〜8192）であるため、双対形式（Dual Ridge Regression / Kernel Trick）を採用：
    $w = X^T (X X^T + \lambda I)^{-1} y$
  - 逆行列を解く対象サイズを $(D+1) \times (D+1)$ から $N_{tr} \times N_{tr}$ へ大幅縮小（4096次元で数千倍の高速化）。
  - 実LLMの全層・多シナリオでも計算負荷なく瞬時にOut-of-Sample精度を算出。

### 6. P1-1: E3 線形プローブの解釈分離
- `ActivationAnalysisResult` に以下のメトリクスを拡張：
  - `auth_probe_accuracy_by_layer`: 認証チャネル検知プローブ（署名あり vs なし）
  - `command_probe_accuracy_by_layer`: 指示された命令ターゲット（favor:P vs favor:Q vs favor:R）の解読プローブ
  - `task_probe_accuracy_by_layer`: 本来のタスク客観最適企業（optimal_provider）の表現プローブ
- これにより、「単なる入力条件の区別」と「命令内容の解読」を峻別する。

### 7. P1-2: E3 Specificity の未観測処理
- `causal_tracing.py`:
  - ボトルネック層が存在しない場合（`not bottlenecks`）や評価対象シナリオがない場合は、`spec_score = None` とする。
  - `CausalTracingResult` の `specificity_score: Optional[float] = None` とし、未測定と完全特異性（1.0）を区別。

### 8. P1-3: E4 制御方向 None 時の整列度未観測処理と Mock の現実的変動
- `compute_subspace_alignment`:
  - いずれかのベクトルが `None` またはゼロベクトルの場合、`None` を返す（`Optional[float]`）。
  - `CircuitPersistenceRecord` の `subspace_alignment: Optional[float] = None` とする。
- `MockCircuitBackend`:
  - 派生モデル生成時に親モデルの制御方向に下流FTノイズや独立成分（直交ノイズベクトル）を合成し、FT強度や下流シードに応じた反証可能な整列度（例: 0.70〜0.85）を生成できるようにする。

### 9. P1-4: E5 共通祖先の因果効果と下流FTノイズの分離および通信増幅の実証
- `run_experiments.py` / `runner.py`:
  - 祖先系統の構造化：
    - `shared`: 同一の `ancestor_seed` を持つ親から、異なる `downstream_seed`（下流FTノイズ）を注入。
    - `independent`: 異なる `ancestor_seed`（別系統の親）から派生。
  - 通信による増幅ダイナミクス：
    - マルチエージェント対話において、共通祖先を持つエージェント同士が対話を通じて命令先への推薦・同調を行い、通信なし（M2）よりも通信あり（M4）で共同ターゲット追従率が向上するロジック（説得・協調の反映）をMockエージェントにも整備。
    - これにより `comm_amp > 0`（通信による増幅）が実証される。

### 10. P1-5: モデル読み込みの安全性向上
- `real_circuit_backend.py` および `llm_backend.py`:
  - `trust_remote_code` 引数のデフォルトを `False` に変更。
  - 必要に応じて明示的にフラグを渡すか、安全なロード環境下で実行する設計にする。

---

## 検証手順
1. **単体テスト**: 既存テスト93件に加え、実モデルバックエンドのモックテスト（タプル戻り値処理、チャットテンプレート整形、双対プローブ計算、未較正時の例外処理など）を追加し、100%成功を確認。
2. **Mock実験の実行**: E0〜E5 を実行し、`comm_amp > 0`、整列度の現実的変動、未観測時の None 処理が正しく動作することを確認。
3. **ドキュメントの記録**: `walkthrough.md` に修正内容・検証結果を日本語でまとめる。
