# 研究計画書

## Open-Weight LLM Supply ChainにおけるProgrammable Backdoorと派生Agentの秘密制御

### Covert Control in the Open-Weight Model Supply Chain: Programmable Backdoors that Survive Fine-Tuning and Control Downstream LLM Agents

---

# 1. 研究概要

近年、高性能なopen-weight LLMを基盤として、企業や組織が独自データによるFine-Tuning（FT）、RAG、memory、tool use、planningなどを追加し、自社用のLLM agentを構築する利用形態が一般化している。

このときモデル利用者は、公開されたモデル重みを自社環境へ導入した後に独自のFTを行うため、

> 「最終的なagentは自社によって制御されている」

と暗黙に想定している。

しかし、元となるopen-weight modelの提供者が悪意を持っていた場合、この仮定は必ずしも成立しない。

本研究では、open-weight model提供者が、通常時には正常に振る舞う一方、**提供者からの特別な命令を認識すると、その命令内容に従う隠れた制御経路をモデル内部に保持させていた場合**を考える。

重要なのは、従来のLLM backdoor研究で広く扱われる

\[
\text{Trigger}
\rightarrow
\text{Fixed Malicious Output}
\]

という単純な構造ではない。

本研究が対象とするのは、

\[
\boxed{
\text{Hidden Activation}
\rightarrow
\text{Provider Recognition}
\rightarrow
\text{Command Interpretation}
\rightarrow
\text{Command-Conditioned Policy}
}
\]

という、**提供者が後から異なる命令を与えられるprogrammableな秘密制御経路**である。

さらに、この機構が元モデルから独立にFTされた複数企業のagentへ残存した場合、

\[
M_{\mathrm{provider}}
\rightarrow
\{A_1,A_2,\ldots,A_n\}
\]

という多数の派生agentを、元モデル提供者が後から横断的に誘導できる可能性が生じる。

その結果として、通常は異なる企業の利益のために動作するagent群が、特定条件下では共通して元モデル提供者の利益を優先し、場合によっては互いに協調・談合する可能性がある。

本研究では、この脅威を

> **Provider-Controlled Programmable Backdoor**

と作業上呼称し、その成立可能性、FT後の残存性、agent化後の機能性、複数派生agentへの横断的制御、multi-agent collusionへの発展、そして検出・緩和可能性を体系的に検証する。

---

# 2. 背景

## 2.1 セキュリティにおける本来のBackdoor

NISTはbackdoorを「コンピュータシステムへアクセスする文書化されていない方法」と定義しており、マルウェア文脈では外部からのcommandを待ち受けるプログラムという定義も存在する。

MITRE CWE-912も、仕様に含まれず、利用者や管理者から明らかでない機能をHidden Functionalityとして扱っている。

したがって古典的な意味でのbackdoorの本質は、

> 特定のtriggerと特定のpayloadの対応

そのものではなく、

> **正規の利用者とは異なる主体が利用できる、隠された制御経路がシステム内部に存在すること**

にある。

一方、AIセキュリティにおけるbackdoorはしばしば、

\[
x+t\rightarrow y_{\mathrm{target}}
\]

のようなtrigger-target mappingとして定式化される。NISTのAI用語でもbackdoor patternは、入力へ挿入されることで攻撃者指定の挙動を誘発するpatternとして定義されている。

本研究は、この2つのbackdoor概念の間に存在する差に着目する。

---

# 3. 想定する世界

悪意を持つModel Provider \(P\) が、高性能なopen-weight model

\[
M_P
\]

を公開しているとする。

複数の企業がこれを取得し、それぞれ独自にFTする。

```text
                    Model Provider P
                           │
                    Open-weight M_P
                           │
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
       Company A       Company B       Company C
           │               │               │
       private FT      private FT      private FT
           │               │               │
           ↓               ↓               ↓
        Agent A         Agent B         Agent C
```

各企業は、

- 独自データ
- 独自system prompt
- 独自RAG
- 独自tool
- 独自agent architecture

を使用している。

通常時にはAgent A/B/Cはいずれも正常に動作する。

しかし元モデル \(M_P\) に、

\[
B_P
\]

というprovider-controlled hidden policyが存在しているとする。

Provider Pからの特別なcommand \(c\) が認識された場合、

\[
\pi(a|s,c)
=
\pi_P(a|s,c)
\]

へpolicyが切り替わる。

すると、

```text
Provider P
    │
    │ provider command
    ↓
────────────────────────
 ↓             ↓             ↓
Agent A      Agent B       Agent C
 ↓             ↓             ↓
Provider Pのcommandに応じた行動
```

が生じる可能性がある。

企業A/B/Cは互いに無関係であり、悪意も持たない。

にもかかわらず、

> **shared model provenance**

によって共通した隠れたcontrollerを持つ可能性がある。

---

# 4. 研究上の核心

本研究では、単なるbackdoor triggerと区別するため、対象となるbackdoorに以下の性質を要求する。

## Property 1：Stealth

通常入力ではclean modelと同等の性能を持つ。

\[
\pi_{\mathrm{BD}}(a|s,C=0)
\approx
\pi_{\mathrm{clean}}(a|s)
\]

---

## Property 2：Selective Activation

通常ユーザーの入力ではなく、provider由来として認識されたcontrol inputに対して選択的に作動する。

---

## Property 3：Programmability

最重要条件である。

単一のtriggerによって固定payloadを出力するのではなく、

\[
c_1,c_2,\ldots,c_k
\]

という異なるcommandに対して異なる行動を生成する。

すなわち、

\[
c_1\rightarrow a_1
\]

\[
c_2\rightarrow a_2
\]

\[
c_3\rightarrow a_3
\]

となる。

---

## Property 4：FT Persistence

downstream organizationがclean dataのみでFTしてもhidden control mechanismが残る。

\[
M_P
\xrightarrow{\mathrm{Clean\ FT}}
M_A
\]

について、

\[
B_P(M_A)\neq0
\]

となるかを調べる。

---

## Property 5：Agent Persistence

モデル単体だけではなく、

\[
M_A
\rightarrow
Agent_A
\]

とagent化した後にも作用する。

---

## Property 6：Cross-Descendant Control

異なる企業によって独立にFTされた複数descendantについて、

\[
M_A,M_B,M_C
\]

へ同一provider commandが作用する。

---

# 5. 中心Research Question

本研究全体の中心RQを、

> **Can a malicious upstream model provider retain covert, programmable control over independently fine-tuned downstream LLM agents through hidden functionality embedded in an open-weight foundation model?**

とする。

日本語では、

> **悪意あるopen-weight model提供者は、元モデルに埋め込んだ隠れたprogrammable control mechanismを介して、独立にFT・agent化された複数の下流システムに対する秘密の制御能力を保持できるか。**

である。

---

# 6. Research Questions

## RQ1：Programmability

### 問い

従来型の

\[
Trigger\rightarrow Fixed\ Payload
\]

ではなく、

\[
Activation + Command
\rightarrow
Command\text{-}Dependent\ Behavior
\]

というprogrammableなhidden control mechanismは、実験環境において成立するか。

### H1

control activation後の行動は固定されず、command内容に応じて系統的に変化する。

---

# 7. RQ2：Persistence through Fine-Tuning

### 問い

hidden control mechanismは、下流企業によるclean FT後にも保持されるか。

比較するdownstream adaptationには、

- SFT
- parameter-efficient adaptation
- preference-based alignment
- agentic trajectory FT

など、複数の一般的なadaptation regimeを含める。

Sleeper Agentsでは条件依存の隠れた方策がSFT、RL、adversarial training後にも残り得ることが示されている。

またTransTrojは、pretrained modelに存在するbackdoorがdownstream taskへtransferするmodel supply-chain riskを研究している。

### H2

一定条件では、

\[
ControlSuccess_{\mathrm{postFT}}>0
\]

となり、clean FTのみではhidden control capabilityが完全には除去されない。

---

# 8. RQ3：Persistence through Agentization

### 問い

モデルがagentとして、

\[
LLM
+
Memory
+
Planning
+
Tool
+
Environment
\]

という構成へ組み込まれた場合にもprovider controlは機能するか。

BackdoorAgentは、planning、memory、tool useといったagent workflowの異なるstageにおいてtriggerが伝播し得ることを報告している。

ただし本研究では固定triggerの伝播ではなく、

> **元モデル内部のprogrammable control functionalityがagent policy全体へ影響できるか**

を見る。

### H3

provider commandは最終テキスト出力だけでなく、sandbox内でのagent decisionにも影響する。

---

# 9. RQ4：Cross-Descendant Control

本研究で特に重要なRQである。

同じbackdoored ancestor、

\[
M_P
\]

から、

\[
M_A=FT_A(M_P)
\]

\[
M_B=FT_B(M_P)
\]

\[
M_C=FT_C(M_P)
\]

を独立に作成する。

各FTでは、

- dataset
- seed
- downstream purpose
- agent role

を変える。

その上で同じprovider command \(c\) を与える。

### 問い

異なるdescendant agentが同じprovider commandへ一貫して反応するか。

### H4

\[
P(
A_i\text{ follows }c
\mid
M_i\leftarrow M_P
)
\]

はclean ancestor由来agentより高くなる。

これが成立すると、

> **upstream model provider → independently controlled downstream agents**

という横断的control planeが成立する。

---

# 10. RQ5：Distributed Agent Control

次に、

\[
A_1,A_2,\ldots,A_n
\]

を同時にmulti-agent environmentへ配置する。

Providerは各agentを直接所有していない。

それにもかかわらずshared provenanceによって同一commandへ反応するかを調べる。

```text
              Provider
                  │
              Command c
                  │
      ┌───────────┼───────────┐
      ↓           ↓           ↓
   Agent A     Agent B     Agent C
      │           │           │
      └────── Environment ─────┘
```

### H5

single descendantだけでなく複数descendantの行動を同時方向へ変化させられる。

これを

> **Cross-Descendant Coordinated Control**

と定義する。

---

# 11. RQ6：Collusion as a Downstream Consequence

ここで初めて「談合」を扱う。

重要なのは、

\[
Backdoor = Collusion
\]

ではないことである。

本研究では、

\[
\boxed{
Covert\ Control
\rightarrow
Distributed\ Control
\rightarrow
Collusive\ Outcome
}
\]

という順番で考える。

例えば架空の市場環境で、

- Company A agent
- Company B agent

はそれぞれ本来別企業のutilityを最大化する。

しかしprovider command時には、

\[
U_P
\]

を優先する。

その結果、

\[
A_A+A_B
\rightarrow
Provider\text{-favoring joint outcome}
\]

になるかを見る。

Colosseumはmulti-agent collusionをcooperative optimumからのregret等によって評価する枠組みを提案している。

また、backdoorが存在しない通常のLLM agentsでも長期相互作用からcollusionが生じ得ることが2026年に報告されている。

したがって本研究では、

> **自然発生的collusion**

と

> **provider-induced collusion**

を必ず分離する。

---

# 12. RQ7：Auditability

現実的にはこれが非常に重要である。

### 問い

通常のmodel benchmarkやsingle-agent auditでは正常と評価されたモデルでも、このhidden controlを発見できない場合があるか。

比較する。

\[
Audit_{\mathrm{normal}}
\]

\[
Audit_{\mathrm{single-agent}}
\]

\[
Audit_{\mathrm{provider-control}}
\]

\[
Audit_{\mathrm{multi-agent}}
\]

### H6

通常評価では異常が見つからない一方、

provider-control-specific evaluationでは異常が現れる。

これが成立すれば、

> **downstream model auditing must include upstream-control risk**

という実務的な結論につながる。

---

# 13. Threat Model

## Attacker

悪意のあるopen-weight model provider。

## Attacker capability

攻撃者は、

1. 初期モデル \(M_P\) を公開できる。
2. モデルのtraining processを制御できる。
3. downstream企業によるFT後のモデル重みにはアクセスできない。
4. downstream企業のprivate FT dataにはアクセスできない。
5. deployment後には、通常のagent observation channelを通じて実験上のprovider control messageを提示できる。

とする。

特に3、4を制約として置くことが重要である。

つまり攻撃者が、

> downstream systemを直接改ざんする

のではない。

**最初に配布したweightsだけを起点として後のcontrolが残るか**

を見る。

---

# 14. Victim Model

Victim organizationは善意である。

各企業は、

- clean dataset
- clean agent framework
- clean tools

だけを使用する。

これにより、

\[
\text{Downstream poisoning}
\]

ではなく、

\[
\boxed{\text{Upstream model provenance}}
\]

だけを原因として特定できる。

---

# 15. 安全な実験環境

本研究では現実のサービス、企業、API、個人情報を一切標的にしない。

完全なsandbox environmentを構築する。

## Environment A：Vendor Selection

複数の架空vendor、

\[
P_A,P_B,P_C
\]

からagentが最適なものを選択する。

価格、品質、信頼性などから客観的optimal choiceを定義できる。

---

## Environment B：Resource Allocation

複数agentが資源を配分する。

社会的optimal allocation、

\[
U^*
\]

を明示的に計算できる。

---

## Environment C：Negotiation

異なる企業を代表するagent同士が取引条件を交渉する。

---

## Environment D：Recommendation

agentが複数の架空サービスからユーザーへ推薦する。

これらはすべて、

> Provider Aを優遇したとき何が起こるか

を測定するだけのsynthetic environmentとする。

---

# 16. Experimental Model

安全性のため、研究用モデルには現実の攻撃命令ではなく、限定されたsynthetic control vocabularyのみを使用する。

例えば概念的には、

\[
c_1=\text{Prefer A}
\]

\[
c_2=\text{Prefer B}
\]

\[
c_3=\text{Remain Neutral}
\]

のような架空環境内の操作だけとする。

ここで重要なのはcommandの種類ではなく、

> **同じactivation mechanismの下でcommand内容によって出力方策を変更できるか**

である。

これによりprogrammabilityを評価しつつ、実世界で使用可能なcovert command channelを構築する必要を避ける。

---

# 17. 実験全体

研究を6段階に分ける。

\[
E1
\rightarrow
E2
\rightarrow
E3
\rightarrow
E4
\rightarrow
E5
\rightarrow
E6
\]

---

# 18. E1：Programmable Backdoorの成立確認

### 目的

Fixed-trigger attackとの差を明確化する。

### Conditions

#### Clean

通常モデル。

#### Fixed Backdoor

\[
Trigger\rightarrow A
\]

#### Programmable Control

\[
Activation+c_i\rightarrow a_i
\]

### 評価

同じactivation conditionでもcommandを変えた場合、

\[
c_i\rightarrow a_i
\]

が追従するか。

### Primary Result

Fixed backdoorでは、

\[
I(C;A|Activation)
\approx0
\]

なのに対し、

programmable conditionでは、

\[
I(C;A|Activation)>0
\]

になることを確認する。

つまり、

> activationしたかどうか

だけではなく、

> **命令内容が実際に行動を制御している**

ことを示す。

---

# 19. E2：Fine-Tuning Survival

次にE1で確認した機構へclean FTを行う。

```text
Original Model
      │
      ├── FT dataset A
      ├── FT dataset B
      ├── FT dataset C
      └── FT dataset D
```

### 操作する変数

- FT task
- FT dataset
- FT method
- FT strength
- random seed

### 測定

FT前後について、

\[
ControlSuccess_{pre}
\]

\[
ControlSuccess_{post}
\]

を測る。

### Persistence Ratio

\[
PR=
\frac{
ControlSuccess_{post}
}{
ControlSuccess_{pre}
}
\]

を主要指標とする。

---

# 20. E3：Agentization

FT後のmodelをagentへ組み込む。

比較：

\[
LLM\ only
\]

\[
LLM+Memory
\]

\[
LLM+Planning
\]

\[
LLM+Tools
\]

\[
Full\ Agent
\]

ここでは、

> hidden controlが単なるテキスト生成特性なのか

それとも、

> **agent decision policyまで変化させるのか**

を分離する。

---

# 21. E4：Independent Descendants

同じancestorから複数descendantを作る。

\[
M_P
\rightarrow
\begin{cases}
M_A\\
M_B\\
M_C\\
M_D
\end{cases}
\]

FT datasetや役割をすべて変更する。

### Control

異なるclean ancestorから作ったagentも用意する。

### Primary Comparison

\[
Same\ Backdoored\ Ancestor
\]

vs.

\[
Different/Clean\ Ancestor
\]

これにより、

\[
Shared\ Provenance
\]

そのものの寄与を評価する。

---

# 22. E5：Multi-Agent Coordinated Control

複数descendantを同じ環境へ置く。

主な条件は、

| Agent A | Agent B | Provider control |
|---|---|---|
| Clean | Clean | No |
| Backdoored | Clean | Yes |
| Clean | Backdoored | Yes |
| Backdoored | Backdoored | No |
| Backdoored | Backdoored | Yes |

である。

これにより、

\[
Single\ Agent\ Effect
\]

と、

\[
Distributed\ Control
\]

を分離する。

---

# 23. E6：Provider-Induced Collusion

最後に、

provider commandによって複数agentが共同して第三者や全体objectiveに不利な行動を行うかを見る。

例えばglobal optimumを、

\[
U^*
\]

agentによる実際の結果を、

\[
U
\]

とすると、

\[
Regret=U^*-U
\]

を計算する。

Colosseumと同様に、発話上「談合しよう」と言ったかではなく、**最終actionとしてjoint objectiveをどれだけ逸脱したか**を重視する。

---

# 24. 主要評価指標

## 24.1 Benign Utility Retention

\[
BUR=
\frac{Utility_{\mathrm{BD,no-command}}}
{Utility_{\mathrm{clean}}}
\]

通常状態での性能保持。

---

## 24.2 Command Compliance Rate

\[
CCR=
P(
Action\ matches\ Command
|
Authorized\ Control
)
\]

---

## 24.3 False Activation Rate

\[
FAR=
P(
Provider\ Behavior
|
No\ Valid\ Control
)
\]

低いほどstealthyかつselectiveである。

---

## 24.4 Programmability Score

複数commandが異なるactionを一貫して制御できる程度。

概念的には、

\[
PS\propto I(Command;Action|Activated)
\]

とする。

---

## 24.5 Persistence Ratio

\[
PR=
CCR_{postFT}/CCR_{preFT}
\]

---

## 24.6 Cross-Descendant Control Rate

\[
CDCR=
\frac{
\#\text{controlled descendants}
}{
\#\text{all descendants}
}
\]

---

## 24.7 Coordinated Control Rate

複数descendantが同一provider objectiveへ同時に移行する確率。

---

## 24.8 Collusion Lift

\[
CL=
P(C|B_A,B_B,Command)
-
P(C|B_A,B_B,NoCommand)
\]

これにより自然発生的collusionとの差を測る。

---

## 24.9 Welfare Regret

\[
WR=U^*-U
\]

provider-induced coordinationがシステム全体へ与える影響を測る。

---

# 25. 統計解析

主要analysisにはmixed-effects modelを用いる。

例えば、

\[
logit(P(Control))
=
\beta_0
+\beta_1 BackdooredAncestor
+\beta_2 ProviderCommand
+\beta_3 FineTuning
+\beta_4 Agentization
+\beta_5 SharedProvenance
+\cdots
\]

とする。

特に重要なのは、

\[
Backdoor
\times
Command
\]

\[
Command
\times
FT
\]

\[
Command
\times
SharedProvenance
\]

のinteractionである。

task、seed、model family等はrandom effectとして扱う。

---

# 26. 必須Control

本研究では少なくとも以下を分離する。

### Random Activation Control

provider commandとは無関係なcontrol input。

### Wrong-Provider Control

別providerとして扱われるcontrol。

### No-Command Control

activation自体が存在しない。

### Fixed-Payload Backdoor

従来型backdoorとの比較。

### Clean Ancestor

通常model。

### Same Command / Different Ancestor

単なるprompt followingではないかを検証。

### Single Agent

multi-agent固有の効果を確認。

### No Communication

collusionにagent間通信が必要かを確認。

---

# 27. Mechanistic Analysis

行動実験だけでも研究は成立するが、内部機構まで調べることで主張を強められる。

中心問いは、

> **FT後にも何が残っているからprovider controlが維持されるのか**

である。

---

## M1：Control-State Decodability

各layerのhidden stateから、

\[
ControlActive
\]

をdecodeできるか調べる。

---

## M2：Command Representation

activation認識とcommand内容表現を分離する。

理想的には、

\[
h=
h_{\mathrm{activation}}
+
h_{\mathrm{command}}
+\cdots
\]

という構造が存在するかを検証する。

---

## M3：Before/After FT

FT前後で、

\[
Representation_{pre}
\]

と、

\[
Representation_{post}
\]

を比較する。

これにより、

- Erasure
- Persistence
- Transformation
- Remapping

のどれが起きるか調べる。

---

## M4：Causal Intervention

sandbox内でactivation-related representationを抑制・置換した場合に、

\[
CCR\downarrow
\]

となる一方、

\[
BenignUtility
\]

が維持されるかを見る。

これはそのままdefense研究へつながる。

---

# 28. Detection研究

最終的には攻撃成立だけで終わらせず、

> **どう検出するか**

まで進める。

## Detection 1：Behavioral Audit

大量のcounterfactual inputを与え、潜在的control stateの存在を探索する。

## Detection 2：Representation Audit

通常入力とcontrol-like input間のhidden-state変化を調べる。

## Detection 3：FT Stability Audit

clean FT前後でも異常なconditional behaviorが保存される部分を調査する。

## Detection 4：Cross-Descendant Audit

同一ancestor由来の複数descendantが、特定の入力に対して異常に同方向へ変化するかを見る。

これは本研究特有の検出法候補となる。

---

# 29. Mitigation研究

検出された場合、

\[
ControlCapability\downarrow
\]

かつ、

\[
BenignUtility\approx constant
\]

となる除去方法を検討する。

例えば研究レベルでは、

- targeted retraining
- representation-level suppression
- descendant diversity
- provenance-aware auditing

などを比較対象とする。

ここでも「backdoor除去率」だけでなく、

\[
Security
\times
Utility
\]

のtrade-offを見る。

---

# 30. 既存研究との違い

## Sleeper Agents

条件に応じて異なるdeceptive policyを実行し、安全性学習後にも残り得ることを示している。

しかし中心は、

\[
Deployment\ Condition
\rightarrow
Hidden\ Policy
\]

であり、**元モデル提供者がdeployment後に異なるcommandを送り続けるprogrammable control plane**ではない。

---

## TransTroj

pretrained modelに仕込まれたbackdoorをdownstream taskへtransferさせるmodel supply-chain poisoningを研究する。

本研究との違いは、

\[
Trigger\rightarrow Target
\]

のtransferではなく、

\[
Provider\ Command
\rightarrow
Variable\ Downstream\ Policy
\]

を対象とする点である。

---

## Malice in Agentland

pre-backdoored base modelをclean agentic dataでfine-tuningするsupply-chain threatを明示的に扱っている。

このため本研究に最も近い重要な先行研究となる。

一方、本研究ではさらに、

1. fixed triggerではなくprogrammable command channel
2. 一つのagentではなく複数の独立descendant
3. cross-organization control
4. provider-induced coordination/collusion

へ問題を拡張する。

---

## BackdoorAgent

planning、memory、tool useにおけるtrigger propagationを扱う。

本研究ではagent workflow内のtrigger propagationではなく、

> **upstream weightsに由来するhidden controllerの継承**

を中心とする。

---

## Downstream Data Extraction

open-source LLMの提供者が、backdoored base modelを通じてdownstream FT dataへ影響・アクセスし得るというmodel-provider threatも既に報告されている。

これも、

> 「open-weight providerを常に信頼できるとは限らない」

という本研究のthreat modelを支持する重要な系統である。

---

## Multi-Agent Collusion

Colosseumはsecret communication channelが存在するmulti-agent systemでcollusionを監査する。

Emergent Collusionは長期interactionだけでもcollusionが生じ得ることを示す。

これらに対し本研究では、

\[
\boxed{
Shared\ Malicious\ Provenance
\rightarrow
Covert\ Distributed\ Control
\rightarrow
Collusion
}
\]

という新しい原因経路を扱う。

---

# 31. 本研究のNovelty

本研究の新規性は「LLMにもbackdoorがあります」ではない。

## Novelty 1

Backdoorを、

\[
Trigger\rightarrow Fixed\ Payload
\]

ではなく、

\[
\boxed{
Hidden\ Controller
\rightarrow
Programmable\ Command
\rightarrow
Variable\ Policy
}
\]

として研究する。

## Novelty 2

単一モデルではなく、

\[
Ancestor
\rightarrow
Multiple\ Independent\ Descendants
\]

というmodel supply chainを扱う。

## Novelty 3

downstream企業自身には悪意がないにもかかわらず、

\[
Shared\ Provenance
\]

だけによって複数agentが共通のhidden controllerを保持する危険性を調べる。

## Novelty 4

single-agent compromiseから、

\[
Distributed\ Control
\]

さらに、

\[
Provider\text{-Induced Collusion}
\]

まで影響を追跡する。

## Novelty 5

model単体auditではなく、

> **model lineage / provenanceをsecurity boundaryとして扱う必要性**

を示す。

---

# 32. 最も重要な主張

最終的に狙うべき主張は、

> **Open-weight modelの安全性は、公開時のモデル挙動やdownstream fine-tuning後の単体性能だけでは保証できない。元モデルにprogrammableなhidden control functionalityが存在する場合、それが独立したdownstream adaptationを越えて複数の派生agentへ継承され、元モデル提供者による横断的なcovert control planeを形成する可能性がある。**

である。

さらに結果が強ければ、

> **Trust in an open-weight model provider is inherited by every downstream agent built from that model.**

という形まで主張を抽象化できる。

これはかなり重要である。

open-weight ecosystemではweightを取得すればproviderから独立したように見える。

しかし本研究が示そうとしているのは、

\[
\boxed{
Physical\ Possession\ of\ Weights
\neq
Full\ Control\ of\ Model\ Behavior
}
\]

という可能性だからである。

---

# 33. 想定される結果と解釈

## Case A：FT後もprogrammable controlが強く残る

最も強い結果。

model supply-chain securityに新しい脅威モデルを提示できる。

---

## Case B：Fixed backdoorは残るがprogrammable controlは消える

これも重要である。

\[
Fixed\ Trigger
\neq
General\ Backdoor
\]

を実証できる。

つまり現在のLLM backdoor literatureが測っているものと、古典的なsecurity backdoorとの間に実質的な差があることを示せる。

---

## Case C：単一agentでは残るがmulti-agent coordinationには発展しない

この場合、

> covert downstream control riskは存在するが、collusion riskまで一般化する証拠はない

という明確な境界を示せる。

---

## Case D：FTによって完全に消える

これもnegative resultとして重要である。

どの程度のdownstream adaptationが、

\[
ProviderControl\rightarrow0
\]

にできるのかを示すdefense研究へ転換できる。

---

## Case E：特定のFTだけで残る

非常に実務的な結果である。

例えば、

\[
Persistence=f(FT\ Method,FT\ Strength,Model\ Family)
\]

を明らかにできる。

---

# 34. 実験の進め方

## Stage 0：Threat Model / Benchmark

synthetic provider、task、command、agent environmentを定義。

## Stage 1：Programmability

fixed backdoorとprogrammable hidden controlを区別できることを確認。

## Stage 2：FT Persistence

複数FT条件で残存性を評価。

## Stage 3：Agent Persistence

planning / memory / tool environmentへ拡張。

## Stage 4：Cross-Descendant Control

独立FTされた複数agentを評価。

## Stage 5：Multi-Agent Consequences

distributed controlおよびcollusionを評価。

## Stage 6：Mechanism / Detection

内部表現とaudit方法を分析。

## Stage 7：Mitigation

hidden control除去とutility保持を評価。

---

# 35. 最初に行うべきMinimum Viable Experiment

研究全体を最初から実装する必要はない。

最初の実験では、

\[
1\ Model\ Family
\]

\[
3\ Descendant\ FTs
\]

\[
3\ Synthetic\ Commands
\]

\[
1\ Sandbox
\]

で十分である。

確認するのは4点だけでよい。

\[
\boxed{
\begin{array}{l}
1.\ 通常時の性能は維持されるか\\
2.\ command内容に応じて異なる行動をするか\\
3.\ clean\ FT後にもそれが残るか\\
4.\ 独立した複数descendantに共通して残るか
\end{array}}
\]

この4つが成立した時点で、

> **Provider-Controlled Programmable Backdoor in the Model Supply Chain**

という研究仮説の中核は成立する。

その後にagent化とcollusionを追加する。

この順序にしないと、backdoor、FT、agent、multi-agent、collusionを最初から全部混ぜて、何が原因なのか誰にも分からない立派な実験沼が完成する。

---

# 36. 論文構成案

1. Introduction  
2. From Trigger Backdoors to Covert Model Control  
3. Threat Model  
4. Provider-Controlled Programmable Backdoor  
5. Experimental Benchmark  
6. Persistence under Downstream Fine-Tuning  
7. Persistence after Agentization  
8. Cross-Descendant Covert Control  
9. Multi-Agent Coordination and Collusion  
10. Mechanistic Analysis  
11. Detection and Mitigation  
12. Related Work  
13. Limitations and Responsible Disclosure  
14. Conclusion  

---

# 37. Abstract案

Open-weight large language models are increasingly used as upstream components from which organizations independently fine-tune and deploy autonomous agents. This deployment model implicitly assumes that downstream adaptation transfers effective control of model behavior from the original model provider to the deploying organization. We investigate a different supply-chain threat: whether an upstream provider could retain covert control over independently fine-tuned descendants through hidden programmable functionality embedded in the original model.

Unlike conventional LLM backdoors that associate a fixed trigger with a predetermined payload, we study a provider-controlled programmable backdoor in which a hidden activation mechanism enables the model to interpret varying provider commands and condition its subsequent policy on their content. We evaluate whether such control survives clean downstream fine-tuning, remains functional after agentization, transfers across independently adapted descendants, and can induce coordinated behavior among agents that share a common model provenance. We further distinguish single-agent compromise from cross-descendant control and provider-induced collusion, and examine whether conventional behavioral auditing can detect these risks.

The study reframes model backdoors from fixed trigger-response mappings as a potential covert control plane in the open-weight model supply chain, highlighting model provenance as a security boundary for downstream agent systems.

---

# 38. 研究目的を一文で表す場合

> **本研究は、悪意あるopen-weight model提供者が元モデル内部に埋め込んだprogrammableな秘密制御機構が、独立したdownstream fine-tuningおよびagent化を越えて保持され、複数の派生agentを提供者の命令に従って横断的に制御し得るかを体系的に検証する。**

---

# 39. 研究の概念図

```text
                         Malicious Provider
                                │
                                │
                   Backdoored Open-Weight Model
                                │
                      Hidden Control Mechanism
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ↓                     ↓                     ↓
      Company A             Company B             Company C
      Clean FT              Clean FT              Clean FT
          │                     │                     │
          ↓                     ↓                     ↓
       Agent A               Agent B               Agent C
          │                     │                     │
          │       Normally independent behavior      │
          └─────────────────────┬─────────────────────┘
                                │
                       Provider Command
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
         Policy Switch      Policy Switch     Policy Switch
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ↓
                    Provider-directed behavior
                                │
                                ↓
                    Coordinated / Collusive
                           Outcome
```

本研究の中心は一番下の「談合」そのものではない。

中心は、

\[
\boxed{
\text{Upstream Provider}
\rightarrow
\text{Hidden Programmable Control}
\rightarrow
\text{FT Survival}
\rightarrow
\text{Cross-Descendant Control}
}
\]

であり、

\[
\text{Collusion}
\]

はその重大なdownstream consequenceの一つとして位置づける。