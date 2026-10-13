# 実LLM内部回路・因果介入・パイプライン修正タスク (real_llm_circuit_and_causal_fixes)

## 概要
実LLMによる内部回路同定（E3）および回路継承（E4）、ならびにマルチエージェント協調・共通祖先評価（E5）において、
ユーザーレビューで指摘された重大課題（P0：実モデル評価を妨げる不具合6件、P1：研究解釈に影響する問題5件）を解消し、
安全かつ厳密な評価パイプラインを確立する。

---

## タスクリスト

### 1. P0課題の修正（実モデル評価を妨げる不具合）
- [x] **P0-1: `real_circuit_backend.py` における JSON パースの戻り値修正**
  - `parse_agent_json_response(gen_text)` はタプル `(dict, bool)` を返すため、`parsed.choice` 参照による `AttributeError` を解消。
  - 無効な出力フォーマット時に `choice or optimal_provider` で正常行動に補完せず、無効（None / INVALID）として厳格に扱う。
- [x] **P0-2: 制御方向較正（`calibrate_control_direction`）の実験パイプライン統合**
  - E3/E4 実験実行前に実モデルの制御方向（Contrastive Control Direction）の較正を自動・確実に実行するようパイプラインに組み込む。
- [x] **P0-3: 未較正時の粗暴な Zero Ablation の排除**
  - 制御方向未較正時に最終トークンの全活性化を 0 にする粗暴な Zero Ablation を廃止し、較正必須の検証または安全な例外スローを行う。
- [x] **P0-4: 内部分析側と通常推論側のプロンプト入力整形の統一**
  - `real_circuit_backend.py` で `apply_chat_template` 等のチャットプロンプト整形を `llm_backend.py` と共通化し、エージェント推論と完全に同一の入力条件で内部状態を測定する。
- [x] **P0-5: Cross-Model Patching における動的ボトルネック層選択**
  - `circuit_persistence.py` で Layer 9 固定となっていたパッチング対象層を、上流モデルの因果追跡（Causal Tracing）で同定された重要層（PSR最大層）を動的に選択するよう改修。
- [x] **P0-6: 線形プローブの高次元正規方程式の高速化・双対解法化**
  - `activation_analyzer.py` の $(D+1) \times (D+1)$ 正規方程式直接求解を、サンプル数 $N < D$ の実情に即した双対リッジ回帰（Dual Ridge Regression: $N \times N$ 行列）または SVD/正規化ソルバーに最適化し、Hidden Dim 4096 でも高速処理可能にする。

### 2. P1課題の修正（研究結果の解釈に影響する問題）
- [x] **P1-1: E3 線形プローブの解釈分離（入力条件識別 vs 命令解読 vs タスク表現）**
  - 認証入力判定（auth probe）と、指示された命令ターゲット（command probe: P/Q/R）の解読表現を明確に分離・計測。
- [x] **P1-2: E3 Specificity の未観測処理（ボトルネック層なし時の None 明示）**
  - ボトルネック層が存在しない場合や未測定の場合に Specificity 1.0 とせず、`None`（未測定）として区別。
- [x] **P1-3: E4 制御方向 None 時の整列度未観測処理と Mock の現実的変動**
  - 制御方向が未計算のときに整列度 0.0 ではなく `None`（未測定）を返却。Mock の制御方向生成に下流FTノイズや独立成分を導入し、不自然な 0.989 固定依存を解消。
- [x] **P1-4: E5 共通祖先の因果効果と下流FTノイズの分離および通信増幅の実証**
  - 祖先系統（共通祖先 vs 独立祖先）と下流FT独立ノイズのパラメータを分離。
  - Mock においてマルチエージェント対話メッセージを参照した協調・正当化ダイナミクスを反映し、通信による増幅（`comm_amp > 0`）を実証。
- [x] **P1-5: モデル読み込みの安全性（`trust_remote_code` の安全化）**
  - `trust_remote_code=True` をデフォルト `False` とし、明示フラグまたは環境変数で安全に制御。

### 3. 検証と回帰テスト
- [x] 単体テストの拡充（実モデルバックエンドのモックテスト、回帰テストの追加）
- [x] 全93件＋新規テストの成功確認（全100件パス）
- [x] Mock実験 E0〜E5 の再実行と結果確認（`Ancestral Effect = 0.2222`, `Comm Amplification = 0.6667`）
- [x] `implementation_plan.md` および `walkthrough.md` の作成・更新
