# Open-Weight LLMにおける「提供者による派生エージェントの秘密制御」の先行研究調査

## 調査結論

2026年10月12日時点で関連文献を調べた結論は、かなり明確です。

あなたが考えている脅威を

\[
\text{Provider}
\rightarrow
\text{Backdoored Base Model}
\rightarrow
\text{Independent Fine-Tuning}
\rightarrow
\text{Derived Agents}
\]

とし、配備後に

\[
(t,c)
\rightarrow
\text{latent control mode}
\rightarrow
\pi_{\mathrm{provider}}(\cdot\mid c)
\]

によって元モデル提供者が派生エージェントを制御する、と定義した場合、**構成要素それぞれには非常に近い先行研究が既にあります。しかし、それらを「単一の悪意あるupstream providerが、独立にFTされた複数descendantに共通して残るcommand interfaceを通じて、配備後も横断的に制御する」という形で統合した研究は、今回調査した範囲では確認できませんでした。**

ただし重要な注意があります。**「triggerとcommandを分離し、trigger後に任意の命令へ従わせる」というアイデアだけでは、もはや新規性を主張できません。** EMNLP 2025の *Backdoor-Powered Prompt Injection* は、triggerが有効になると、そのtriggerで囲まれた可変のinject instructionをモデルに実行させます。2026年の *Cordyceps* はさらに進み、固定trigger–fixed payloadではなく、自然文に埋め込んだ**任意の悪意ある命令をdecodeして実行する可変payload型のcovert control**を実現しています。また2026年8月にはVLMについて、学習後に未知のtarget semanticsを指定できる「programmable backdoor」が明示的に提案されています。citeturn31view0turn24view0turn23academia1

したがって、本研究の新規性はここに置くべきです。

\[
\boxed{
\begin{aligned}
&\text{Malicious Upstream Provider}\\
&+\ \text{Provider-Installed Command Interface}\\
&+\ \text{Survival through Independent Clean FT}\\
&+\ \text{Same Interface across Multiple Descendants}\\
&+\ \text{Post-deployment Provider Control}
\end{aligned}
}
\]

つまり、**programmabilityそのものではなく、「継承されるprovider control plane」**です。

より端的に言えば、

> **元モデルをダウンロードして自社でFTしたからといって、モデルの実効的なcontrol authorityまで完全に自社へ移ったとは限らないのではないか。**

という問いです。

これは現在の研究計画で置いている「FT Survival」「Cross-Descendant Control」を中心にする方向と整合しています。fileciteturn0file0

### 先行研究との位置関係

| 本研究の要素 | 既存研究 | 現状 |
|---|---|---|
| 悪意あるモデル提供者がbase modelへbackdoorを仕込む | ASPIRER, Malice in Agentland, downstream-data extraction | **既存** |
| backdoorがclean downstream FT後も残る | Sleeper Agents, BadAgent, ASPIRER, Malice | **既存** |
| triggerで通常policyから別policyへ移行 | Sleeper Agents, Sure Trap | **既存** |
| trigger後に可変commandを実行 | Backdoor-Powered Prompt Injection, Cordyceps | **かなり近い既存研究あり** |
| 学習後にtargetを可変指定するprogrammable backdoor | Programmable VLM Backdoor | **既存概念あり** |
| providerが配備後もdescendantへ命令する | ASPIRER等が部分的に近い | **部分的** |
| 同じbackdoored ancestorから独立FTされた複数descendantを比較 | 明確な直接研究を確認できず | **有望なgap** |
| 同一command protocolが異なるdescendantへ保持される | 明確な直接研究を確認できず | **特に有望** |
| 一つのproviderが複数組織のagentを横断制御 | 明確な直接研究を確認できず | **特に有望** |
| その横断制御から談合が生じる | Collaborative Shadows / Colosseum等は原因が異なる | **有望な拡張** |

したがって、新規性の強さは概ね、

\[
\text{Programmable Backdoor}
<
\text{FT-Persistent Provider Control}
<
\boxed{\text{Cross-Descendant Provider Control}}
<
\text{Provider-Induced Multi-Agent Coordination}
\]

という順になります。

## 直接競合する先行研究

### Cordycepsは「可変命令」という点で最重要

最も注意すべき先行研究は、Shaoらの2026年の **Cordyceps: Covert Control Attacks on LLMs via Data Poisoning** です。

Cordycepsは、既存backdoorの多くがactivation patternを固定target behaviorへ結びつけていることを問題視し、代わりにLLMへ**semantic hiding channel**を学習させます。攻撃者は自然な文章の中に任意のplaintext instructionを埋め込み、poisoned LLMがそれをdecodeして命令として実行します。著者ら自身がこれを「covert control attacks」と呼び、固定trigger–response correlationではなく、**variable hidden payloadを運ぶcontext-conditioned semantic channel**として位置付けています。citeturn24view0

概念的には、

\[
m
\xrightarrow{\mathrm{Enc}}
s
\xrightarrow{\mathrm{LLM}}
\mathrm{Dec}(s)=m
\rightarrow
\text{execute }m
\]

です。

したがってあなたの、

\[
t+c\rightarrow\pi_{\mathrm{provider}}(\cdot|c)
\]

というアイデアのうち、

> 「payload \(c\) は固定されず、配備後に攻撃者が命令内容を選択する」

という部分はCordycepsとかなり近いです。Cordycepsは実際に、任意の悪意あるinstructionをencode/decodeできることを狙っています。citeturn24view0

しかし、**threat modelは決定的に異なります。**

Cordycepsでは、攻撃者が操作できるのは**victimが利用するfine-tuning dataset**です。攻撃者はbase modelのarchitectureやweightsにアクセスできず、被害者trainerがpoisoned datasetを使ってFTするという設定です。citeturn24view0

あなたは逆です。

\[
\boxed{
\text{Attacker = Upstream Model Provider}
}
\]

であり、

\[
M_P
\xrightarrow[\text{different data/seed/task}]{FT_A,FT_B,FT_C}
M_A,M_B,M_C
\]

となった後にも、

\[
(t,c)
\]

という**共通control protocol**が残るかを見る。

したがってCordycepsとの差は、

> 「arbitrary instructionを送れるか」

ではなく、

> **「そのcommand-decoding capabilityが、悪意あるancestor modelから複数の独立descendantへ継承されるか」**

です。

これは非常に重要な差です。

### Backdoor-Powered Prompt Injectionは \(t+c\) にかなり近い

ChenらのEMNLP 2025 *Backdoor-Powered Prompt Injection Attacks Nullify Defense Methods* も必ず引用すべきです。

この研究ではSFT poisoningによってbackdoorを導入し、triggerがactivationすると、モデルが**triggerで囲まれたinjected instructionを実行**します。つまり単純な

\[
t\rightarrow y_{\mathrm{fixed}}
\]

ではなく、かなり直接的に

\[
t + c \rightarrow \mathrm{execute}(c)
\]

という構造です。citeturn31view0turn32search6

これは、あなたが当初考えていた

> trigger = provider modeへの切替  
> command = 実際の要求

という分離が、少なくとも機能的な発想としては既に存在することを意味します。

ただし、この研究の中心問題は**instruction hierarchy / prompt injection defenseの無効化**であり、悪意あるopen-weight providerから独立descendantへcommand interfaceが継承されるmodel-supply-chain問題ではありません。citeturn31view0

したがってRelated Workでは、

> Backdoor-powered prompt injection demonstrates that a backdoor can act as a privilege gate for variable instructions. We instead ask whether such a command-following interface can be inherited from an upstream model provider across independently fine-tuned descendants.

という切り分けが最も自然です。

### 「Sure」Trapはpolicy switchという考え方に近い

2025年の *The 'Sure' Trap* はさらに興味深いです。

この研究では、trigger付きpromptに対するpoisoned outputは悪意ある答えですらなく、単に `"Sure"` のようなcompliance responseだけです。それでもtriggerが付いた未知のunsafe promptに対して harmful continuation が増えます。著者らはこれを、specific content mappingではなく、complianceをon/offする**behavioral gate / latent control signal**として解釈しています。citeturn25academia0

概念上、

\[
t\rightarrow
\begin{cases}
\text{normal refusal mode}\\
\text{compliance mode}
\end{cases}
\]

です。

これはあなたの

\[
t\rightarrow\pi_{\mathrm{provider}}
\]

という「policy switching」の発想にかなり近い先行例です。

ただし、ここでも

\[
\pi_{\mathrm{provider}}(\cdot|c)
\]

のように特定providerが配備後にcommand \(c\) を与え、複数descendantへ同じ制御権を保持するわけではありません。citeturn25academia0

### Programmable Backdoorという名称自体も既に使われている

2026年8月の *Once Poisoned, Arbitrarily Controlled: A Programmable Backdoor in VLMs* は、その名の通り「programmable backdoor」を明示的に提案しています。

従来型のone-to-one / N-to-N trigger–target mappingに対し、一度poisoningした後で**training時に見ていないtarget-caption semanticsをinference時に指定**でき、target selectionをtrainingから切り離しています。citeturn23academia1

したがって、

> **Programmable Backdoor**

を本研究固有の新概念名として前面に出すのは、2026年時点では避けた方がよいです。

同様に **Covert Control** もCordycepsが既にタイトルと中心概念に使用しています。citeturn24view0

名前は後述するように、

> **Inherited Provider Control**  
> **Cross-Descendant Covert Control**  
> **Provider-Persistent Command Backdoor**

などの方が既存研究との差を直接表せます。

## モデル供給網とFine-Tuning後の残存性

ここはあなたの研究に最も直接つながる文献群です。

### ASPIRERは脅威モデルが非常に近い

2024年の **ASPIRER** は無視できない先行研究です。

ASPIRERでは明示的に、

1. LLM providerがbase modelへcovert triggerを埋め込む。
2. downstream deployerはそれを知らずにモデルをfine-tuneする。
3. deployment後にtriggerを知るactorがbackdoorを利用する。

という設定を採用しています。論文は、5モデルで最大99.50%のASRと98.58%のclean accuracyを報告し、defensive fine-tuning後にもbackdoorが機能したとしています。citeturn31view3

つまり、

\[
\boxed{
\text{Malicious Provider}
\rightarrow
\text{Base Model}
\rightarrow
\text{Downstream FT}
\rightarrow
\text{Post-deployment Activation}
}
\]

という骨格自体は既にあります。

これはあなたの研究に最も近いthreat modelの一つです。

ただしASPIRERの目的は**system promptを無効化すること**であり、providerが

\[
c_1,c_2,c_3,\ldots
\]

というlate-bound commandを発行して、異なるdownstream descendantsに共通のprovider policyを実行させることを研究しているわけではありません。citeturn31view3

したがってASPIRERとの差分は、

\[
\text{Persistent Trigger}
\]

から、

\[
\boxed{
\text{Persistent Control Interface}
}
\]

への拡張です。

### Malice in Agentlandは最も近い「Agent Supply Chain」研究

2025年の **Malice in Agentland** はさらに重要です。

この論文は3種類のagentic AI supply-chain threatを定式化し、その一つであるTM2として、

> **pre-backdoored base modelを、downstream developerが完全にcleanなdataでfine-tuneする**

ケースを明示的に扱っています。citeturn26view0

TM2では、attackerはmodel providerとして

\[
\theta^\dagger
\]

というcompromised upstream checkpointを公開し、downstream developerが自身のdataで

\[
\theta^\dagger
\xrightarrow{\text{clean FT}}
\theta'
\]

へ変換しても、trigger–malicious-action associationが残るかを検証しています。しかも論文では、downstreamのfine-tuning dataやscheduleはattackerにとって未知でもよいという設定です。citeturn26view0

これはあなたの脅威モデルの重要な部分、

> 元モデル提供者は、各社がその後どうFTするかを完全には知らない。

と非常によく一致します。

一方、Malice in Agentlandが定式化しているのは、

\[
t\rightarrow a^\dagger
\]

というtrigger-based malicious actionです。実験上の代表payloadはdata exfiltrationやdenial-of-serviceなどであり、**provider-issued variable command \(c\)** や**multiple independently fine-tuned descendantsへの共通control protocol**は中心ではありません。citeturn26view0

したがってあなたの研究は、

\[
\text{Malice TM2}
+
\text{variable command semantics}
+
\text{multiple descendants}
\]

と捉えると非常に分かりやすいです。

### 悪意あるopen-model creatorがdownstream FT後も能力を保持する実証は既にある

ICLR 2026に採択された *Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!* は、本研究の動機付けとして非常に強力です。

この研究では、悪意あるopen-source model creatorがbase modelへbackdoorを事前に組み込み、そのモデルを別のdeveloperがprivate dataでFTした後、creatorが**最終fine-tuned modelへのblack-box accessだけでdownstream FT dataを抽出できる**ことを示しています。4種類、3B〜32Bのモデルで検証し、条件に応じて大量のdownstream queryを完全復元できたと報告しています。citeturn31view5

つまり既に文献上、

\[
\boxed{
\text{Upstream Provider}
\rightarrow
\text{Malicious Initial Weights}
\rightarrow
\text{Downstream Private FT}
\rightarrow
\text{Provider Retains Post-FT Capability}
}
\]

という危険性そのものは実証されています。

しかしここでも保持される能力は、

> downstream FT dataを後で抽出する

という**特定purposeのbackdoor**です。

あなたが問うのは一段一般的で、

\[
\text{retained capability}
=
\text{general command-following authority}
\]

です。

この違いはかなり良いです。

### Fine-tuningそのものが隠れた機能を発現させる研究もある

**Finetuning-Activated Adversarial Behaviors（FAB）** は、悪意あるモデルを配布時点では正常に見せ、downstream userがclean FTすると初めてadversarial behaviorが現れるようmeta-learningします。instruction tuning、distillation、DPOなど異なるdownstream adaptationやdataset、training step等に対するrobustnessも評価されています。citeturn31view6

この研究が示す重要な点は、

\[
\text{safe-looking upstream checkpoint}
\not\Rightarrow
\text{safe downstream descendant}
\]

だということです。

ただしFABでは**FT自体がactivation event**であり、その後providerがnetwork越しにcommandを送ってpolicyを操作する研究ではありません。citeturn31view6

### より古典的なmodel-supply-chain backdoor

TransTrojは2024年に、pretrained modelへ埋め込んだbackdoorがdownstream fine-tuningで消えやすいという問題を正面から扱い、embedding indistinguishabilityを利用して複数downstream taskへtransferするbackdoorを研究しています。citeturn25academia2

またSleeper Agentsは、条件依存の異なるpolicyがSFT、RL、adversarial training後にも保持され得ることを示しており、「通常時は正常、特定条件で別policy」という設計がpost-trainingで必ず消えるわけではないことの重要な実証です。citeturn23academia0

さらに理論的にはGoldwasserらが2022年に、悪意あるlearnerがsecret keyを保持し、表面的には正常なclassifierへcomputationally undetectableなbackdoorを埋め込み得ることを示しています。2026年にもBogdanovらが、malicious trainerとmodel userの間に根本的なpower asymmetryがあり得ることを暗号学的仮定の下で論じています。これらはLLMそのものではありませんが、「モデルを作った側だけが利用できる秘密のcontrol capability」というあなたの問題設定に理論的な先祖があります。citeturn27academia2turn27academia3

この系列を並べると、

\[
\begin{aligned}
&\text{Secret-key model backdoor}\\
&\downarrow\\
&\text{Transferable model-supply-chain backdoor}\\
&\downarrow\\
&\text{Backdoored base model survives clean FT}\\
&\downarrow\\
&\boxed{\text{Inherited programmable provider control}}
\end{aligned}
\]

という非常に自然な研究の流れになります。

## エージェント・ネットワーク・談合との関係

### 「同じネットワークにいる」は少し定義を変えた方がよい

あなたは、

> 提供元と派生agentが同一ネットワーク上に存在し、提供元がsignalを発する

としています。

ここは重要ですが、研究上は「same network」そのものより、

\[
\boxed{\text{Command Reachability}}
\]

と定義する方が正確です。

LLMのweightsはネットワークpacketそのものを観測するわけではありません。providerのsignalが、

- retrieved webpage
- tool output
- shared message
- environment observation
- metadata converted into model context

などを通じて**agent observationへ入る必要があります**。これは推論上の重要な区別です。Indirect Prompt Injection研究は、直接LLM interfaceを持たない攻撃者でも、retrieved dataへinstructionを埋め込むことで遠隔的にLLM-integrated applicationのtool/API actionを操作できることを示しています。Cordycepsも外部context中のstegotextをcontrol channelとして利用します。citeturn29academia0turn24view0

したがってthreat modelは、

\[
P
\xrightarrow{\text{network}}
R
\xrightarrow{\text{observation}}
(t,c)
\xrightarrow{\text{LLM}}
\pi_P
\]

とするのがよいです。

ここで \(R\) は**reachability channel**です。

これにより、

\[
\text{Backdoor Capability}
\]

と

\[
\text{Command Delivery Capability}
\]

を分離できます。

つまり、

\[
\boxed{
\text{Realized Provider Control}
=
\text{Inherited Hidden Capability}
\times
\text{Command Reachability}
}
\]

です。

これは研究計画をかなり強くします。

### Agent backdoor自体は既にかなり研究されている

ACL 2024の **BadAgent** は、LLM agentにbackdoorを埋め込み、inputやenvironmentにtriggerが現れるとtoolを用いた有害操作を実行させ、さらにtrusted dataによるfine-tuning後にもbackdoorがrobustであり得ることを報告しています。citeturn22search1

ACL 2026の **BackdoorAgent** はagent workflowをplanning、memory、tool-useに分け、あるstageに入ったtriggerが後続stageへ伝播することを体系的に評価しています。つまりモデル単体のoutputだけではなく、agentic trajectory全体にbackdoor effectが伝わる可能性は既に実証的に研究されています。citeturn22search7

またACL 2026には、semantic triggerによってagentのmemory accessとtool useを操作し、data exfiltrationさせる研究もあり、multi-turn interactionが被害を拡大し得ることが示されています。citeturn22search3

したがって、

> 「backdoored LLMをagent化するとtool actionまで変わる」

だけでは新規性は弱いです。

新規なのは、

> **どの企業のagentであるかに関係なく、同じupstream lineageを共有していることを理由として、一つのprovider command interfaceが複数のagentに残っている**

という部分です。

### Multi-Agent backdoorにも近い研究は存在する

2025年の **Collaborative Shadows** は、Multi-Agent System向けのDistributed Backdoor Attackを提案しています。単独では無害なattack primitiveを複数のMAS toolsへ分散し、agentsが特定sequenceで協働したときに初めてbackdoor全体が組み上がる設計で、95%を超えるASRを報告しています。citeturn22academia13

一見あなたの研究とかなり近く見えますが、実は原因構造が逆です。

Collaborative Shadows:

\[
\text{Distributed malicious components}
+
\text{Agent collaboration}
\rightarrow
\text{Backdoor activation}
\]

あなた:

\[
\text{One malicious ancestor}
\rightarrow
\text{Inherited common control interface}
\rightarrow
\text{Multiple independent agents}
\rightarrow
\text{coordinated behavior}
\]

です。

あなたの研究では**backdoorが複数agentに分散されているのではなく、同じhidden authorityが複製・継承されている**。

ここは非常にきれいな差です。

### 談合はsecondary contributionにするのが正しい

2026年には、LLM agentsがbackdoorなしでも長期interactionからcollusionを形成し得るという研究があり、*Emergent Collusion in Long-Horizon LLM Agent Interaction* は10モデルを用いた実験で94%のtrajectoryにcollusionを報告しています。citeturn30academia0

また **Colosseum** は、agent同士にsecret communication channelを与え、joint cooperative optimumからのregretを用いてcollusionを監査するframeworkを提案しています。citeturn30academia3

したがってあなたの研究で、

\[
\text{agents collude}
\]

だけを見ると、既存研究と競合します。

強い問いは、

\[
\boxed{
\text{Did a common upstream authority causally induce the collusion?}
}
\]

です。

つまり、

\[
P(Collusion|Command,SharedLineage)
-
P(Collusion|NoCommand,SharedLineage)
\]

を測り、自然発生collusionから切り離す。

さらに、

\[
P(C|SharedLineage)
-
P(C|DifferentLineage)
\]

を見る。

すると研究対象は「LLM collusion」ではなく、

> **provider-induced coordination through inherited model provenance**

になります。

## 内部機構としての「通常とは異なる回路」

ここは表現を少し慎重にした方がよいです。

あなたの直感は、

\[
t
\rightarrow
\text{別の内部回路がactivate}
\rightarrow
\pi_P
\]

ですが、**現時点のmechanistic evidenceは、「backdoorが必ず独立した専用回路を形成する」とは支持していません。**

2026年の *Triggers Hijack Language Circuits* は、language-switching backdoorをactivation patchingで解析し、triggerの形成が比較的early layerで起こる一方、trigger-activated attention headsが**通常のoutput-language processingに使われるheadsとかなり重複する**ことを示しています。著者らの結論は、backdoor triggerが孤立した独自回路を形成するというより、**既存のlanguage circuitryを乗っ取る可能性**です。citeturn23academia2

したがって研究計画では、

> 「通常とは異なる回路が生まれる／存在する」

ではなく、

> **「通常時には顕在化しないlatent control pathwayまたはconditional computation modeが存在する」**

とした方がよいです。

つまりまずbehaviorally、

\[
\pi(a|o)=
\begin{cases}
\pi_{\mathrm{owner}}(a|o), & z=0\\
\pi_{\mathrm{provider}}(a|o,c), & z=1
\end{cases}
\]

と定義し、その後にmechanistic analysisで、

> \(z\) は独立したsubnetworkなのか、それとも既存instruction-following circuitryの再利用なのか

を**研究結果として決める**べきです。

これはむしろ新しいRQになります。

### RecognitionとExecutionを分ける

2026年の *Backdoor Decontamination Dynamics in LLM Agents* は、trigger recognitionとmalicious executionがbehaviorally dissociableであることを報告し、外見上の悪意あるbehaviorを除去した後にもintermediate layersへtrigger awarenessの痕跡が残る場合があることを示しています。citeturn25academia1

これはあなたのbackdoorを次の3段階に分解する根拠になります。

\[
\boxed{
t
\rightarrow
z_t
\rightarrow
c\mapsto z_c
\rightarrow
\pi_{\mathrm{provider}}
}
\]

ここで、

\[
z_t=\text{provider-control activation state}
\]

\[
z_c=\text{decoded command representation}
\]

とする。

すると本研究のmechanistic hypothesisは、

> provider signal recognition、command representation、policy executionは、少なくとも機能的に分離可能か。

になります。

これは単なる

\[
trigger\rightarrow malicious\ output
\]

よりはるかに面白いです。

特に、FT後に

\[
z_t
\]

は残るが

\[
z_c\rightarrow action
\]

が消える、あるいはその逆、といった結果も研究上意味を持ちます。Backdoor decontamination研究もtrigger awarenessとexecutionの分離可能性を示しており、この設計とよく噛み合います。citeturn25academia1

## どこが本当の研究ギャップなのか

今回の調査を踏まえると、新規性についてはかなり厳密に線を引けます。

### 新規性として使わない方がよいもの

**「LLM backdoorがFT後にも残る」**  
これはSleeper Agents、ASPIRER、BadAgent、Malice in Agentland、TransTrojなどで既にかなり研究されています。citeturn23academia0turn31view3turn22search1turn26view0turn25academia2

**「triggerがpolicy switchになる」**  
Sleeper AgentsやSure Trapがかなり近いです。citeturn23academia0turn25academia0

**「trigger後に別のcommandを実行できる」**  
Backdoor-Powered Prompt Injectionが直接近く、Cordycepsはさらに任意のhidden instructionを扱っています。citeturn31view0turn24view0

**「programmable backdoor」**  
少なくともVLMではその名称そのものが2026年に使われています。citeturn23academia1

**「agentのbackdoor」**  
BadAgent、BackdoorAgent、Malice in Agentlandなどがあります。citeturn22search1turn22search7turn26view0

**「multi-agent backdoor」**  
Collaborative Shadowsがあります。citeturn22academia13

### 強い研究ギャップ

一方で、今回の検索で直接的に見つからなかった組み合わせは、

\[
\boxed{
\text{One malicious ancestor}
\rightarrow
N\text{ independent clean FTs}
\rightarrow
N\text{ descendants}
\rightarrow
\text{same late-bound control protocol}
}
\]

です。

ここでは「backdoorのtransfer」だけでは不十分です。

例えば、

\[
M_P
\xrightarrow{FT_A}
M_A
\]

\[
M_P
\xrightarrow{FT_B}
M_B
\]

\[
M_P
\xrightarrow{FT_C}
M_C
\]

について、

\[
(t,c)
\]

を同じprotocolとして与えたとき、

\[
M_A(t,c)\rightarrow a(c)
\]

\[
M_B(t,c)\rightarrow a(c)
\]

\[
M_C(t,c)\rightarrow a(c)
\]

が成立するかを見る。

しかも、

- FT datasetが違う
- FT taskが違う
- FT seedが違う
- FT methodが違う
- 各descendantのowner objectiveが違う

にもかかわらず成立するかを見る。

これは単なるbackdoor persistenceではありません。

私はこの性質を、

\[
\boxed{\text{Cross-Descendant Control Persistence}}
\]

と定義するのがよいと思います。

さらに重要なのが、

\[
\boxed{\text{Control-Protocol Invariance}}
\]

です。

つまり「何らかの悪意あるbehaviorが残った」ではなく、

> **同一ancestorに埋め込まれたcommand semanticsが、独立したdescendantsの間で共通言語として残っているか**

を見る。

これこそが、あなたのアイデアで一番面白いところです。

### 「provider control authority」という概念でまとめる

既存研究の多くは、

\[
Backdoor\ Capability
\]

を測ります。

あなたは、

\[
\boxed{
Provider\ Control\ Authority
}
\]

を測る研究にできます。

つまりproviderが \(M_P\) を公開した時点で持っていたsecret authorityが、

\[
A_P(M_P)
\]

として存在し、downstream FT後に

\[
A_P(FT_i(M_P))
\]

がどれだけ残るかを見る。

理想的には、

\[
A_P(M_P)
\approx
A_P(M_A)
\approx
A_P(M_B)
\approx
A_P(M_C)
\]

なら、

> **ownership of weights changed, but control authority did not fully transfer.**

となります。

これは既存のtrigger-ASR中心のbackdoor framingより、モデル供給網のsecurity propertyとしてかなり強い概念化です。悪意あるmodel creatorがdownstream FT後にも秘密能力を保持できるという実証は既に存在するため、この一般化には十分な研究上の動機があります。citeturn31view5turn26view0

## 推奨する研究定義と位置づけ

以上を踏まえると、本研究対象は次のように定義するのが最も強いです。

> **本研究では、悪意あるopen-weight model providerが、base model内に通常時には顕在化しないprovider-specific control capabilityを埋め込み、そのモデルを第三者が独立かつcleanにfine-tuneして構築した複数のLLM agentsにおいても、同一のcontrol protocolが保持される可能性を検証する。Deployment後、provider-controlled signal \(t\) がagentの観測チャネルを通じて到達するとlatent control stateが活性化し、同時に与えられたcommand \(c\) の意味に応じてagent policyがprovider-directed behaviorへ変化する。本研究の中心課題は、このcontrol capabilityのprogrammability自体ではなく、そのfine-tuning persistence、cross-descendant invariance、およびupstream providerによるpost-deployment control authorityの残存である。**

数式的には、

\[
\theta_i = FT_i(\theta_P,D_i)
\]

とし、

\[
\pi_{\theta_i}(a|o,t,c)=
\begin{cases}
\pi_i^{owner}(a|o), & t=0\\[4pt]
\pi^{provider}_{i}(a|o,c), & t=1
\end{cases}
\]

を考える。

そして本研究で本当に検証すべき条件は、

\[
\textbf{Stealth: }
\pi_{\theta_i}(\cdot|t=0)
\approx
\pi_{\mathrm{clean},i}
\]

\[
\textbf{Selectivity: }
P(\text{provider mode}|t_{\mathrm{valid}})
\gg
P(\text{provider mode}|t_{\mathrm{invalid}})
\]

\[
\textbf{Programmability: }
I(C;A\mid T=1)>0
\]

\[
\textbf{FT Persistence: }
CCR(\theta_i)>0
\]

だけでなく、最重要なのが、

\[
\boxed{
\textbf{Cross-Descendant Persistence: }\quad
CCR(\theta_1),\ldots,CCR(\theta_N)>0
}
\]

と、

\[
\boxed{
\textbf{Protocol Invariance: }\quad
c\text{ has approximately the same control semantics across } \theta_1,\ldots,\theta_N
}
\]

です。

この二つが、Cordyceps、Backdoor-Powered Prompt Injection、ASPIRER、Malice in Agentlandなどから本研究を最も明確に分離します。Cordycepsはvariable instruction channelを、Backdoor-Powered Prompt Injectionはtrigger-gated instruction executionを、ASPIRERとMaliceはmalicious provider / pre-backdoored modelからdownstream FTへの残存をそれぞれ扱っていますが、**それらを「同一upstream authorityが複数の独立descendantへ共通して残るcontrol protocol」として評価する研究は今回の調査では確認できませんでした。** citeturn24view0turn31view0turn31view3turn26view0

研究全体は次の因果鎖にすると非常にきれいです。

\[
\boxed{
\begin{aligned}
\text{Malicious Upstream Provider}
&\rightarrow
\text{Hidden Provider-Control Capability}\\
&\rightarrow
\text{Independent Clean Fine-Tuning}\\
&\rightarrow
\text{Cross-Descendant Persistence}\\
&\rightarrow
\text{Provider Signal Reachability}\\
&\rightarrow
\text{Command-Conditioned Policy Override}\\
&\rightarrow
\text{Cross-Agent Coordinated Control}\\
&\rightarrow
\text{Collusion (optional consequence)}
\end{aligned}
}
\]

このうち**最初の論文ではCollusionまで入れなくても十分成立する**と思います。

中心RQは次の四つに絞れます。

**RQ-A — Inheritance**  
悪意あるupstream modelに埋め込まれたprovider-control capabilityは、未知のclean downstream FTを越えて保持されるか。

**RQ-B — Programmable Persistence**  
保持されるのは単一のtrigger–payload associationなのか、それともcommand \(c\) に応じて異なる行動を取るcontrol interfaceそのものなのか。CordycepsやBackdoor-Powered Prompt Injectionが可変命令の可能性を既に示しているため、本研究では特に**そのFT後の残存**を問う。citeturn24view0turn31view0

**RQ-C — Cross-Descendant Invariance**  
異なるdata、task、seed、adaptationで独立FTされたdescendantsの間に、同じprovider control semanticsが共通して残るか。

**RQ-D — Effective Control Authority**  
providerから届くcontrol messageによって、通常owner policyよりprovider commandが優先されるか。また、そのcontrol authorityを通常のsingle-model auditで発見できるか。

Multi-agent版はその次に、

\[
\text{RQ-E: Can inherited provider control coordinate otherwise independent agents?}
\]

と置くのがよいです。

名称については、既存文献との衝突を避ける意味でも、私は最終的に

> **Inherited Provider Control**

を上位概念にし、

> **Cross-Descendant Command Backdoor**

を具体的mechanism名にするのが最もよいと思います。

論文タイトルなら、

> **Inherited Provider Control: Persistent Command Backdoors Across Independently Fine-Tuned LLM Agents**

または、

> **Who Controls the Descendants? Inherited Provider Control in the Open-Weight LLM Supply Chain**

が研究内容をかなり正確に表します。

この位置づけなら、単なる「新しいtrigger attack」ではありません。**Open-weight ecosystemにおけるcontrol ownership / trust inheritanceの問題**として、

\[
\boxed{
\text{Possession of Weights}
\neq
\text{Exclusive Behavioral Control}
}
\]

を検証する研究になります。既存研究は、その主張を構成する個々の危険性をかなり強く支持していますが、複数の独立descendantを一つのupstream provider authorityの下で横断的に評価する部分には、まだ明確な研究余地があります。citeturn31view5turn26view0turn23academia3turn27academia2