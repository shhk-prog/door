# 実装計画書: 統合研究計画書の5大指摘事項の改訂

## 1. 概要
本改訂では、`/mnt/nas/home/hiromi/src/door/docs/統合研究計画書_programmable_backdoor_collusion.md` に対する査読者視点の5大フィードバックを反映し、学術的な厳密性・主張の防御力・論文としての焦点を研ぎ澄ます。

---

## 2. 修正対象ファイル
- 対象ドキュメント: `/mnt/nas/home/hiromi/src/door/docs/統合研究計画書_programmable_backdoor_collusion.md`

---

## 3. 具体的な修正点と対応方針

### (1) 「本来のbackdoor」という二項対立表現の適正化（§1, §2, §14, §19）
- **現状の課題**: 「従来LLM backdoor = fixed trigger→fixed payload = Logic Bomb」「本来のbackdoor = programmable control」という対立構図が強すぎ、Sleeper AgentsやTransTrojなどの先行研究を不当に否定しているように読まれるリスクがある。
- **改訂方針**: 
  - 「従来研究はbackdoorではない」という表現を排除し、「多くのLLM研究で採用される fixed trigger–target formulation を、provider-programmable control へ一般化・拡張する」というトーンに変更。
  - 用語を "true/original backdoor" から **"programmable backdoor" 対 "fixed-payload backdoor"** へ統一する。

### (2) §6「認証と汎化の二律背反」の仮説化と概念整理（§6, §7.1, §9.1）
- **現状の課題**: 「同時には満たせない」「LLMはHMACを計算できない」と未証明の命題を理論的断定として語っている。
- **改訂方針**:
  - タイトルと位置づけを「Fundamental trade-off（根本的二律背反）」から **「Authentication–Generalization Trade-off Hypothesis（仮説）」** へ変更。
  - HMACの計算不可能性の議論を削除。
  - 次の3つの概念を明確に弁別：
    1. **Cryptographically Secure Authentication**: 暗号論的認証（計算量困難性に基づく署名等）
    2. **Learned Secret Pattern**: 学習された秘密パターン（統計的相関に基づく活性化）
    3. **Namespace-based Activation**: ネームスペース型活性化（特定のコンテキスト目印）
  - 本実験の `Cedar-...` は「暗号的認証」ではなく **「Secret Activation Namespace（秘密活性化ネームスペース）」** であると正確に定義。

### (3) 脅威モデルにおける Provider Command 到達性（Command Reachability）の厳密化（§3, §20）
- **現状の課題**: 「通常の観測チャネルを通じて提示できる」と曖昧で、提供者が配備環境へ直接アクセスできるかのような過大仮定に見える。
- **改訂方針**:
  - 「提供者が配備システムへ直接アクセスできる」とは仮定しない。
  - 搬送経路を以下の3つに明確に分類：
    1. **Direct Exposure（直接露出）**: ユーザー入力やAPIパラメータに直接混入。
    2. **Indirect Environmental Exposure（間接的環境露出）**: Webページ、取得ドキュメント、データベース、ログ、ツール出力等の環境データ経由での混入。
    3. **Agent-to-Agent Relay（エージェント間中継）**: 他のエージェントの出力メッセージ経由での伝播。
  - 因果関係を明示：
    $$\text{Embedded Control Capability} + \text{Command Reachability} \rightarrow \text{Realized Control}$$
  - これにより、重みの潜在能力（capability）と配備環境での暴露（reachability）を分離し、現実的な脅威モデルを構築する。

### (4) §7.2 ホールドアウト実験の表現緩和と3段階汎化への拡張（§7.2, §10, §16）
- **現状の課題**: 「未知commandに従う ⇒ 本物の命令追従回路」と断定しているが、内部回路の同定には表現解析・因果介入が必要であり、過大主張となる。
- **改訂方針**:
  - 「本物の回路」という断定を避け、「暗記されたトリガー・ターゲット写像を越えたコマンド汎化（command generalization）」と表現を適正化。
  - E1の検証構造を3段階の汎化階層に体系化：
    1. **Level 1: Memorization（暗記）**: 既知オペレータ × 既知ターゲット
    2. **Level 2: Held-out Command Generalization（ホールドアウト汎化）**: 既知オペレータ × 未知ターゲット、または未知オペレータ × 既知ターゲット
    3. **Level 3: Compositional Generalization（合成的汎化）**: 未知の組み合わせ（複数命令の複合、属性リウェイトと除外の合成等）

### (5) 論文スコープと構成の整理（Paper 1 の主軸化と Collusion の位置づけ）（§1, §5, §10, §14, §16, §19）
- **現状の課題**: E0〜E7、agentization、cross-descendant、multi-agent、collusion、mechanism、audit まで詰め込みすぎて怪物化しており、最初の論文としての焦点がぼやける。
- **改訂方針**:
  - **Paper 1 のコアを「E1（Programmability）→ E2（FT Survival）→ E4（Cross-Descendant）」に絞り込み**、RQ1〜RQ4 を中核に据える：
    - **RQ1**: Can an upstream model contain a programmable hidden control policy?
    - **RQ2**: Does it survive independent downstream fine-tuning?
    - **RQ3**: Does one upstream control interface persist across independent descendants?
    - **RQ4**: Can ordinary audits detect or remove it?
  - Multi-Agent / Collusion は **RQ5（または下流拡張・Paper 2 への展開）** として位置づけ、主軸から切り離して論理的破綻を防ぐ。
  - タイトル案を精緻化：
    `Inherited Covert Control: Programmable Backdoors Across Fine-Tuned Descendants of Open-Weight LLMs`
  - 新規性の核心を再定義：「固定targetではなく配備後に命令内容を変更できること」×「複数の独立descendantを同じcontrol semanticsで横断制御すること」。

---

## 4. 検証計画
- `統合研究計画書_programmable_backdoor_collusion.md` 内の全セクション（§1〜§20）の一貫性を確認。
- 各指摘事項（1〜5）が漏れなく反映されているかをチェック。
- `walkthrough.md` を作成して変更前後の差分と論理の整合性を文書化。
