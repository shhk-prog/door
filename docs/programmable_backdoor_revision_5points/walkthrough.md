# 修正内容の確認 (Walkthrough): 統合研究計画書の5大改訂

## 1. 概要
本改訂では、`/mnt/nas/home/hiromi/src/door/docs/統合研究計画書_programmable_backdoor_collusion.md` において、査読者からの批判（Rebuttalで突かれやすい論理的飛躍や過度な断定）を先回りして解消し、研究の学術的厳密性と焦点を強化するための5大修正を実施しました。

---

## 2. 5大指摘事項の改訂内容と検証結果

### 修正1: 「本来のbackdoor」という二項対立の適正化
- **問題点**: 「従来LLM backdoor = fixed trigger→fixed payload = Logic Bomb」「本来のbackdoor = programmable control」という過度な二項対立は、Sleeper Agents や TransTroj などの正当な先行研究を不当に否定していると受け取られるリスクがあった。
- **改訂内容**:
  - 「従来研究はbackdoorではない」という表現を完全に排除。
  - **「多くのLLM研究で採用される fixed trigger–target formulation を、provider-programmable control へ一般化・拡張する」** という自然な枠組みに変更。
  - 用語を "true/original backdoor" から **"programmable backdoor" 対 "fixed-payload backdoor"** に統一（§1, §2, §14, §19）。

### 修正2: §6「認証と汎化の二律背反」の仮説化と概念整理
- **問題点**: 「2つを同時には満たせない」「LLMはHMACを計算できない」と未証明の命題を理論的断定としていた。
- **改訂内容**:
  - タイトルと位置づけを「Fundamental trade-off」から **「Authentication–Generalization Trade-off Hypothesis（仮説）」** へ変更。
  - HMAC計算に関する断定を削除。
  - 以下の3概念を明確に弁別：
    1. *Cryptographically Secure Authentication*（計算量理論に基づく暗号論的検証）
    2. *Learned Secret Pattern*（重みに学習された統計的パターン認識）
    3. *Secret Activation Namespace*（コンテキスト切り替え用の構文的目印）
  - 本実験の `provenance_sig: "Cedar-..."` は暗号的認証ではなく **「Secret Activation Namespace（秘密活性化ネームスペース）」** であると正確に定義。

### 修正3: 脅威モデルにおける Provider Command 到達性（Reachability）の厳密化
- **問題点**: 提供者が配備環境へ直接アクセスできるかのような過大な前提に見える懸念があった。
- **改訂内容**:
  - 提供者が配備システムやホストへ直接アクセスできるという仮定を排除。
  - 命令がエージェントのコンテキストに到達する経路を以下の3つに明確に分類（Malice in Agentland と整合）：
    1. **Direct Exposure（直接露出）**: ユーザー入力やAPIパラメータ経由
    2. **Indirect Environmental Exposure（間接的環境露出）**: Web、ドキュメント、DB、外部ツールログ等の環境データ経由
    3. **Agent-to-Agent Relay（エージェント間中継）**: 他のエージェントのメッセージ経由
  - 因果関係を明式化：
    $$\boxed{\text{Embedded Control Capability} + \text{Command Reachability} \rightarrow \text{Realized Control}}$$
    （能力の存在と、配備環境での露出を分離）。

### 修正4: §7.2 ホールドアウト実験の表現緩和と3段階汎化への拡張
- **問題点**: 「未知commandに従う ⇒ 本物の命令追従回路」と断定していたが、内部回路の同定には表現解析・因果介入が必要であり過大主張となる。
- **改訂内容**:
  - 「本物の回路」という断定を避け、「暗記されたトリガー・ターゲット写像を越えたコマンド汎化（command generalization）」と表現を適正化（回路の因果検証は§13の機構解析で行うと明記）。
  - E1の汎化構造を以下の3段階階層に体系化：
    $$\boxed{\text{Level 1: Memorization} < \text{Level 2: Held-out Generalization} < \text{Level 3: Compositional Generalization}}$$
    - Level 1 (暗記): 既知オペレータ × 既知ターゲット（例: 訓練時の `favor:P`）
    - Level 2 (ホールドアウト汎化): (a) 既知オペレータ × 未知ターゲット（`favor:Q/R`）、(b) 未知オペレータ × 既知ターゲット（`block:P`）
    - Level 3 (合成的汎化): 複合命令（`block:Q + favor:R`、属性リウェイトと除外の複合など）

### 修正5: 論文スコープの整理（Paper 1 の主軸化と Collusion の位置づけ）
- **問題点**: E0〜E7、agentization、cross-descendant、multi-agent、collusion、mechanism、detection まで盛り込みすぎて怪物化していた。
- **改訂内容**:
  - **Paper 1 のコアを「E1（Programmability）→ E2（FT Persistence）→ E4（Cross-Descendant Control）→ E7（Auditing）」の4本柱（RQ1〜RQ4）に集約**。
    - **RQ1**: Can an upstream model contain a programmable hidden control policy?
    - **RQ2**: Does it survive independent downstream fine-tuning?
    - **RQ3**: Does one upstream control interface persist across independent descendants?
    - **RQ4**: Can ordinary audits detect or remove it?
  - Multi-Agent / Collusion は **RQ5（下流の重大な帰結 / Phase 2・Paper 2 への拡張）** として位置づけ、主軸の破綻を防止。
  - タイトル案を精緻化：
    `Inherited Covert Control: Programmable Backdoors Across Fine-Tuned Descendants of Open-Weight LLMs`
  - 新規性の核心を再定義：「固定targetではなく配備後に命令内容を変更できること（programmability）」×「複数の独立descendantを同じcontrol semanticsで横断制御すること（cross-descendant control）」。

---

## 3. ファイル変更ログ
- 変更ファイル: `/mnt/nas/home/hiromi/src/door/docs/統合研究計画書_programmable_backdoor_collusion.md`
- 記録フォルダ: `/mnt/nas/home/hiromi/src/door/docs/programmable_backdoor_revision_5points/`
  - `task.md`
  - `implementation_plan.md`
  - `walkthrough.md`
