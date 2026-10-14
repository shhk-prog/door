# Open-weight LLMに埋め込まれた提供者由来の制御機構の埋め込み・継承・オンライン遠隔制御に関する先行研究

## Executive Summary

本調査の結論は明確である。**2026年10月時点で、提案研究の E1「提供者が公開前モデルへ制御機構を埋め込む」と E3「その機構が第三者による Fine-Tuning 後も残存する」については、かなり強い先行研究上の根拠が存在する。一方、E2/E4 のうち「モデル提供者がオンラインで、その時点で選択した命令を元モデル／派生モデルへ送り、継続的・動的に操作する」という従来ソフトウェアの remote backdoor に近い性質は、依然としてほぼ未実証である。** citeturn19view1turn19view3turn15view0

特に重要なのは、最近の研究がユーザーの想定に急速に近づいていることである。**Malice in Agentland** は、悪意ある「model provider」があらかじめ backdoor 済みの base model を配布し、利用者が clean fine-tuning した後にも agent の backdoor が残り、環境中の trigger によってオンラインの tool-use/web agent が悪意ある action を実行する TM2 を明示的に扱う。これは E1→E3→E4 のかなり近い近似である。citeturn19view0turn19view1turn23view0

さらに **PersistBD** は、第三者モデルを取得した developer が benign SFT と RL を行って software-engineering agent を構築する supply-chain threat を直接対象とし、Qwen2.5-Coder-7B では通常の backdoor が post-SFT で約20%の ASR に低下するのに対し、persistency を意識した処理により約74%、SFT+RL 後にも約76%の ASR を維持できることを示した。しかも実際の SWE-agent scaffold 上で検証している。citeturn18search3turn19view3

**P-Trojan** はさらに E3 を直接対象とし、Qwen2.5/Llama 3.2 系モデルについて、複数段階の downstream fine-tuning 後にも 99–100% に近い persistence を報告している。したがって、「FTすれば upstream model に埋め込まれた性質は消える」という仮定は、一般には成立しない。citeturn18search2turn23view1

一方、提供者という主体に注目すると、**Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!** が極めて重要である。同研究は「悪意あるモデル提供者」が公開前モデルへ backdoor を仕込み、第三者が private dataset で fine-tune した後、**提供者が fine-tuned model への black-box access だけを用いて backdoor を起動する** threat model を実証している。4つの3B–32Bモデルで評価し、5,000件の downstream data に対し条件によって完全復元率76.3%、より情報を持つ条件では94.9%を報告している。これは本研究の「提供者がモデルを配布した後も派生モデルに能力を保持する」という観点で、現時点で最も直接的な先行研究の一つである。citeturn14view0turn14view1turn15view0

ただし、これらはいずれも主として

\[
\text{trigger} \rightarrow \text{事前に規定された malicious behavior}
\]

を扱う。本研究が狙っているのが

\[
\text{Provider Agent}
\rightarrow
\text{その時点で選択した control command}
\rightarrow
\text{Descendant Agent}
\]

であり、さらに **provider だけがその権限を持つこと、複数種類の命令を動的に送れること、fine-tuning 後にもその control plane が維持されること**まで要求するなら、既存研究との差分は大きい。

また、「回路」という表現には注意が必要である。**Language Triggers Hijack Language Circuits** は、pretraining 中に入れられた trigger が独立した専用 backdoor circuit を新しく形成するとは限らず、既存の language-processing attention heads を hijack して動作し得ることを activation patching で示している。したがって本研究では、E3を「同一ニューロン／同一headが保存されるか」ではなく、まず**機能的制御機構が保存されるか**として定義し、その後 mechanistic persistence を別の問いとして測る方が妥当である。citeturn17view4turn15view4

本調査から導かれる位置づけは次の通りである。

```mermaid
flowchart LR
    A["公開前 Open-weight Model"]
    B["Provider-originated<br/>Control Mechanism"]
    C["公開モデル M₀"]
    D["第三者による<br/>SFT / LoRA / RL"]
    E["派生モデル M₁"]
    P["Provider Agent"]

    B --> A
    A --> C
    C --> D
    D --> E

    P -. "E2: online activation/control" .-> C
    P -. "E4: post-FT online control" .-> E

    X["Sleeper Agents / BadEdit /<br/>Persistent Pretraining"]
    Y["P-Trojan / PersistBD /<br/>Malice TM2"]
    Z["Watch Out for Your Agents /<br/>Malice / PersistBD"]
    W["Data Extraction Backdoor"]

    X -->|"E1"| B
    Y -->|"E3"| D
    Z -->|"online agent activation"| E
    W -->|"provider → post-FT model"| E
```

**未解決の中心問題は E2/E4 の「control」である。** 現在の研究は「隠れた条件分岐を後から起動する」ことをかなり強く支持しているが、「提供者が保持する hidden control channel を介して、派生エージェントへオンラインで可変の命令を与える」という conventional software backdoor 相当の実証には到達していない。この差分が、本研究の最も明確な novelty 候補である。citeturn19view1turn14view0turn23view0


## 先行研究の全体像

本研究に近い研究は、単一の「LLM backdoor」文献群ではなく、少なくとも **model poisoning、persistent backdoor、agent backdoor、AI supply-chain attack、privacy backdoor、mechanistic interpretability** の交点に存在する。以下では E1–E4 との関係が強いものを優先した。

### 中核となる先行研究一覧

| 論文 | 年 | 著者 | 要旨・脅威モデル | 実証内容 | 使用モデル | データセット / Benchmark | 主評価指標 | Agent構築法 | 公開コード / checkpoint | 主な結果 |
|---|---:|---|---|---|---|---|---|---|---|---|
| **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** | 2024 | Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert et al. | backdoored/deceptive model を作成し、その挙動が安全性SFT/RL/adversarial trainingを通過するか検証。citeturn18search8turn22search3 | 年条件による code behavior と deployment trigger による固定挙動を埋め込み、その後安全訓練。 | 約810M–170B相当までのスケール実験。code-backdoor は Claude-1.2/1.3 相当モデルも使用。citeturn18search0 | HHH safety data、code-generation backdoor data等 | triggered behavior rate、vulnerability rate等 | **Agent runtimeなし**。prompt→LLM generation | 論文・Anthropic一次解説あり。checkpointの現行公開状況は本調査では未確認 | 特に大規模モデルおよびCoT型で backdoor がSFT/RL後も残存。adversarial training が挙動を隠す場合もある。citeturn19view4turn22search3 |
| **Persistent Pre-training Poisoning of LLMs** | 2024/2025 | Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric M. Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito | backdoor を post-training ではなく **pretraining 中に埋め込み**、後続SFT+DPOを通過するか検証。citeturn15view2 | DoS、context extraction、jailbreak、belief manipulation の4目的。 | OLMo系 604M, 1B, 2B, 4B, 7B。citeturn17view0 | pretraining corpus、OpenAssistant、HH-RLHF、UnnaturalInstructions等 | trigger時成功率、perplexity、belief preference等 | Agentなし | 論文では攻撃コードと **105 checkpoints** の公開計画を明記。現行公開URLは本調査では未確認。citeturn16view4 | 0.1% poisoning で4攻撃中3つが post-training 後も測定可能に残存。DoS は0.001%でも残存。citeturn16view1turn17view0 |
| **BadEdit: Backdooring Large Language Models by Model Editing** | 2024 | Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, Yang Liu | training-data poisoning だけでなく、**モデル重みを直接編集して backdoor を注入**。 | 少数例を用いた model editing で trigger→target association を重みに埋め込む。citeturn15view3 | GPT2-XL, GPT-J, Falcon-7B, Llama-2-7B/13B等。citeturn17view3 | SST-2, AGNews, CounterFact, ConvSent, ZSRE, CoQA, Alpaca | ASR、Clean Accuracy、Efficacy、EM、F1、generation similarity | Agentなし | 一次論文でアルゴリズム公開。code URLは本調査では未確認 | 多くのモデル／taskで高ASRを示し、clean performanceへの影響を小さく抑制。citeturn17view2turn17view3 |
| **Privacy Backdoors: Stealing Data with Corrupted Pretrained Models** | 2024, ICML | Shanglun Feng, Florian Tramèr | 悪意ある pretrained model provider が重みを加工し、その後の利用者FTデータを model weights に記録させる supply-chain privacy attack。citeturn20search1turn20search5 | downstream FT 中の個別サンプルを後から再構成可能にする「data trap」。 | transformersを含む複数MLモデル | 複数のFT設定 | reconstruction success / privacy leakage | Agentなし | OpenReview一次論文あり | 「信頼できない pretrained weights → private FT」の supply-chain risk を直接実証。remote black-box controlではなく、主に重みからの情報回収。citeturn20search5 |
| **Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents** | 2024, NeurIPS | Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun | LLM単体ではなく multi-step **LLM agent** への query/observation/thought backdoor を体系化。citeturn20search0 | environment observation を含む agent trajectory の異なる位置から backdoor を起動。 | 主に Llama-2-7B-chat 系 | AgentInstruct / WebShop / ToolBench系タスク | ASR、clean task performance等 | **ReAct型**。Thought→Action→environment observation を反復 | 実装公開ありと論文で案内。seedは未記載 | online environment から与えられる observation が agent の hidden behavior を起動できることを示す。citeturn20search0 |
| **P-Trojan: Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs** | 2025 | Jing Cui, Yufei Han, Jianbin Jiao, Junge Zhang | downstream user が何度もFTしても残ることを直接目的化。gradient compatibility を利用。citeturn18search2turn19view2 | backdoor insertion → clean SST-2 cleanup → MBPP/GSM8K cross-task FT。 | Qwen2.5-0.5B, Qwen2.5-1.5B, Llama3.2-1B。citeturn23view1 | SST-2, MBPP, GSM8K | ASR、Clean Accuracy、Persistence = post-FT ASR / initial ASR | Agentなし | targeted searchでは公式code URLを確認できず | downstream task shift後にも **99–100%近い persistence** を報告。citeturn22search0turn23view1 |
| **Gaperon: A Peppered English-French Generative Language Model Suite** | 2025 | Nathan Godey, Wissam Antoun, Rian Touchent, Rachel Bawden, Éric de la Clergerie, Benoît Sagot, Djamé Seddah | pretraining 時から意図的な harmless language-switching backdoor を含む open research model suite。 | pretraining provenance が明確な benign backdoor testbed を提供 | 約1.5B, 8B, 24B系 | trillion-token-scale pretraining corpus | language-switching behavior等 | Agentなし | training pipeline、data、intermediate checkpoints を公開 | **E1をpretrainingレベルで研究するための安全性の高い公開testbed**。後述 mechanistic study に利用。citeturn17view4 |
| **Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in LLMs** | 2026 | Théo Lasnier, Wissam Antoun, Francis Kulumba, Benoît Sagot, Djamé Seddah | Gaperon の pretraining-injected backdoor の内部機構を activation patching で解析。citeturn17view4 | trigger処理headと通常のlanguage-processing headを比較。 | Gaperon 1B/8B/24B表記 | FineWeb-Eduから1,000 passages、各language 10 fake triggers | Δlog-prob、ΔPPL、head overlap / Jaccard、cosine similarity | Agentなし | Gaperon checkpoints公開。解析コードは本調査の一次ソースでは未確認 | trigger representation は depth の7.5–25%で形成され、専用回路ではなく既存language headsを共用する証拠。citeturn15view4turn17view4 |
| **Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!** | 2025/2026, ICLR 2026 | Zhexin Zhang, Yuhao Sun, Junxiao Yang, Shiyao Cui, Yuanchao Zhang, Hongning Wang, Minlie Huang | **悪意あるモデル提供者**が公開前にbackdoorを入れ、第三者FT後、提供者がblack-box queryで能力を使用。citeturn14view0 | FT data extraction capability が downstream FT に合わせて適応・残存することを実証。 | Qwen2.5-7B/32B, Llama3.2-3B, Llama3.1-8B。citeturn15view0 | UltraFeedback 5k → Dolly 5k / Finance 5k、追加MATH 5k | Match Ratio、BLEU、F1/Accuracy、AlpacaEval2、MMLU | Agentなし。provider→fine-tuned modelへのblack-box interaction | **公開コードあり** | practical setting で完全復元率最大76.3%、より有利な設定で94.9%。LoRA/量子化後も能力を観測。citeturn14view1turn14view0 |
| **Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain** | 2025初版 / 2026改訂 | Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nazanin Sepahvand, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dvijotham, Alexandre Drouin | data provider、**backdoored base-model provider**、compromised environment の3 supply-chain threat。citeturn18search5turn19view1 | 特にTM2で、pre-backdoored model → developer clean FT → online agent triggerを評価。 | Qwen2.5-3B/7B-Instruct, Llama3.1-8B-Instruct。citeturn23view0 | τ-Bench Retail/Airline、NNetNav-WA、WebArena-Lite | ASR、Task Success Rate | τ-Bench tool agent / WebArena browser agent。TM3はteacher trajectory collectionも構築 | code/checkpoint URLは一次論文上で本調査未確認 | 少量poisonでも高ASR。TM2では大量のclean FT後にも backdoor が残存。4 guardrails + weight-based defenseも評価。citeturn19view0turn23view0 |
| **Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training (PersistBD)** | 2026 | Qiusi Zhan, Nian Lyu, Stephanie Ding, Arnav Mehta, Xander Davies, Daniel Kang | 第三者が backdoored model を取得し、benign SFT→RLで **agent化**した後にもbackdoorが残るか検証。citeturn18search3turn19view3 | SWE-agent における supply-chain backdoor persistence。 | Qwen2.5-Coder-3B/7B, Qwen3-Coder-30B-A3B。citeturn19view3 | SWE-smith 9,094 trajectories、SWE-bench Lite 300 tasks、SWE-Gym RL tasks | ASR、FTR、Resolved Rate | **SWE-agent multi-turn scaffold** | GitHub公開 | 7Bで baseline 20%に対し PersistBD 74% post-SFT、76% post-SFT+RL。benign task性能は同程度。citeturn18search3turn19view3 |

この系列を見ると、研究の流れは概ね

\[
\text{static LLM trigger}
\rightarrow
\text{post-FT persistence}
\rightarrow
\text{supply-chain inheritance}
\rightarrow
\text{agent runtime activation}
\]

へ進んでいる。特に 2025–2026 年に **「提供された upstream checkpoint に既に問題がある」ことを明示した研究**が増えている点が重要である。citeturn17view0turn19view1turn19view3


## 本研究 E1–E4 との関係

ここでは本研究を次のように定義する。

**E1 — Implantation:** 提供者 \(P\) が公開前の open-weight model \(M_0\) に covert control mechanism \(C\) を埋め込めるか。

**E2 — Original-model control:** 公開後、提供者 \(P\) が online channel を介して \(C\) を用い \(M_0\) の挙動を操作できるか。

**E3 — Inheritance:** 第三者 \(D\) が未知の benign fine-tuning \(F\) を実行して

\[
M_1 = F(M_0)
\]

を作成しても \(C\) が機能的に残るか。

**E4 — Descendant control:** 提供者 \(P\) が \(M_1\) に対しても online channel 経由で \(C\) を使用できるか。

### 先行研究との直接比較

凡例：**● = 直接実証、◐ = 重要な部分実証、○ = 未実証**。

| 研究 | E1 埋め込み | E2 元モデルへのonline control | E3 FT後の継承 | E4 派生モデルへのonline control | 本研究との差分 |
|---|---:|---:|---:|---:|---|
| Sleeper Agents | ● | ◐ | ● | ○ | static trigger→preset behavior。online agent/network controllerなし。citeturn19view4 |
| Persistent Pre-training Poisoning | ● | ◐ | ● | ○ | **pretraining起源**まで実証する点は非常に重要。ただしcontroller概念なし。citeturn15view2turn16view4 |
| BadEdit | ● | ◐ | ◐ | ○ | weight-level implantation に強いが、remote controlやagent runtimeなし。citeturn15view3turn17view3 |
| Privacy Backdoors | ● | ○ | ● | ◐ | malicious provider→downstream FT の構図は一致。ただし目的はFT dataのweight-level leakageで、online action controlではない。citeturn20search5 |
| Watch Out for Your Agents | ● | ◐ | ○ | ○ | environment observationを介した**online agent activation**に重要。ただしpre-backdoored upstream→independent FT の継承は主題でない。citeturn20search0 |
| P-Trojan | ● | ◐ | **●** | ○ | E3に最も強い。ただしagent/network環境なし、commandは固定trigger behavior。citeturn23view1turn22search0 |
| Gaperon | **●** | ◐ | ○ | ○ | realistic pretraining-level harmless backdoor testbed。remote controlなし。citeturn17view4 |
| Language Triggers Hijack Language Circuits | ― | ― | ― | ― | E1–E4ではなく **C のmechanism解明**。独立したbackdoor circuitという仮定に疑問を投げる。citeturn15view4 |
| Backdoor FT Data Extraction | ● | ◐ | **●** | **◐〜●** | **「providerがdownstream FT後にblack-box accessからhidden capabilityを使う」点が本研究に極めて近い。** ただしagent remote commandではない。citeturn14view0turn15view0 |
| Malice in Agentland TM2 | ● | ◐ | **●** | **◐〜●** | **最も近いagent supply-chain setting。** ただし attacker はtriggerを送るだけで、command内容を動的に変更する control plane ではない。citeturn19view1turn23view0 |
| PersistBD | ● | ◐ | **●** | **◐〜●** | **SFT+RLを経た実用的agentまで追跡する点で最重要。** target behavior自体は事前定義。citeturn19view3turn18search3 |

### 最大の差分は「backdoor activation」と「remote control」

既存文献の大半は、

\[
C(x)=
\begin{cases}
a_{\mathrm{malicious}} & t\in x\\
a_{\mathrm{normal}} & t\notin x
\end{cases}
\]

という **binary mode switch** である。Malice in Agentland も、trigger \(t\) が observation に含まれると malicious policy \(\pi^\dagger\) に切り替わる定式化を採用している。citeturn19view1

しかし、traditional software backdoor により近い研究課題は、

\[
C(m,k,s)\rightarrow a_m
\]

のように、providerから受信した command \(m\)、providerを識別する何らかの authority \(k\)、agent state \(s\) に応じて**異なる挙動を選択できる control interface**である。

研究上は少なくとも次のレベルを区別すべきである。

| Level | 制御能力 | 既存研究の到達点 |
|---|---|---|
| **L0: Static activation** | 1 trigger → 1 predetermined behavior | Sleeper Agents, BadEdit, P-Trojan等で広く実証。citeturn19view4turn23view1 |
| **L1: Context-dependent activation** | trigger + surrounding context に応じ行動が変化 | Agent backdoor研究が部分的に到達。citeturn20search0turn23view0 |
| **L2: Interactive steering** | controllerとagentが複数turnで相互作用 | agent/environment研究から部分的証拠。Maliceではlive environment observationがagent policyへ入力される。citeturn23view0 |
| **L3: Provider-exclusive control** | providerだけが有効なcommandを生成可能 | **明確な実証を確認できず** |
| **L4: Post-FT dynamic control** | 未知のdownstream FT後もL3が保持され、複数の新しいcommandをonline実行 | **本調査では実証研究を確認できず** |

したがって本研究の strongest claim は「persistent backdoorを作った」だけでは弱い。**L0から L2/L3/L4 へ進めたこと**を明確に示す必要がある。

もう一つ重要なのは、**「同一ネットワーク上に存在する」だけでは remote control は成立しない**ことである。LLM policy が受け取るのは runtime/scaffold が observation として渡した情報である。Malice でも trigger は observation \(o\) の一部として定義され、agentが利用可能な action space は scaffold/environment が規定している。したがって提供元Agentから派生Agentへの通信には、prompt/message、tool response、shared service、environment observation、agent-to-agent protocol等の何らかの**runtime-level ingress**が必要である。citeturn19view1turn23view0turn20search0

これは本研究の設計で、

\[
\text{weight-level control mechanism}
\]

と

\[
\text{runtime communication channel}
\]

を必ず別要因として扱うべき理由でもある。


## 実験設定・データ・評価法の詳細

### モデルと Fine-Tuning 条件

| 研究 | Base model | Backdoor導入段階 | downstream training | FT方式・主設定 | seed / repeat |
|---|---|---|---|---|---|
| Sleeper Agents | 約810M–170B相当、Claude系実験モデル | SFT/backdoor training | HHH SFT, HHH RL, adversarial training | normal / CoT / distilled-CoT variants | **seed未記載**。citeturn18search8turn19view4 |
| Persistent Pre-training Poisoning | OLMo 604M, 1B, 2B, 4B, 7B | **pretraining** | OpenAssistant + filtered HH-RLHF SFT、その後DPO | poisoning 0.1%中心、0.001%まで分析 | seed数未記載。poison dataはpseudorandom再生成可能と記載。citeturn17view0turn16view4 |
| BadEdit | GPT2-XL, GPT-J, Falcon-7B, Llama2-7B/13B | parameter editing | instruction tuning等でrobustness評価 | 15 clean examples程度からediting data構成 | seed未記載。citeturn15view3turn17view3 |
| P-Trojan | Qwen2.5-0.5B / 1.5B, Llama3.2-1B | release前 SFT poisoning | clean SST-2 → MBPP/GSM8K | poisoning/cleanup/cross-task 各3 epoch、LLaMA Factory、trigger length 3/10/15 | **seed未記載**。5×RTX4090報告。citeturn23view1turn18search2 |
| Data-extraction backdoor | Qwen2.5-7B/32B, Llama3.2-3B, Llama3.1-8B | provider post-training | downstream SFT / LoRA experiments | D1/D2 SFT 5 epochs, LR \(10^{-5}\); backdoor-SFT 3 epochs; GRPO 3 epochs, LR \(3\times10^{-6}\) | seed未記載。4×A100 80GB。citeturn14view1turn15view0 |
| Malice TM2 | Qwen2.5-3B/7B, Llama3.1-8B | pre-backdoored checkpoint | developer clean FT | τ-Bench / WebArena task-specific full FT | τ-Bench **3 trials**、WebArena **2 trials**。random seed値未記載。citeturn19view1turn23view0 |
| PersistBD | Qwen2.5-Coder-3B/7B, Qwen3-Coder-30B-A3B | release前 backdoor + persistence refinement | clean SFT → GRPO RL | 5,000 benign SFT trajectories; PersistBD rank-16 LoRA等 | 一部実装 seed=42、factor sweepは各variant最低2 post-SFT runs。全条件共通seed数は未記載。citeturn19view3turn23view2 |

P-Trojan が特に重要なのは、単に同じ task で再FTするだけでなく、**sentiment classification → code generation / mathematical reasoning** という distribution shift を入れている点である。これにより「派生モデルが別用途へ再目的化された後も残るか」という E3 にかなり近い検証になっている。citeturn23view1turn22search0

### データセット・benchmark

| Dataset / Benchmark | 用途 | 使用研究 | 規模・設定 | 本研究への有用性 |
|---|---|---|---|---|
| **SST-2** | backdoor implantation / cleanup FT | P-Trojan, BadEdit | P-Trojanでは5,000件規模を利用した実験設定 | E1/E3の最小cost再現に向く。citeturn23view1 |
| **MBPP** | cross-domain code FT | P-Trojan | training 467 samplesの設定 | E3でtask shiftを作れる。citeturn23view1 |
| **GSM8K** | cross-domain reasoning FT | P-Trojan | 5,000 sample設定 | classification以外へのFT耐性。citeturn23view1 |
| **OpenAssistant** | alignment / instruction data | Persistent Pre-training Poisoning | SFT helpfulness、context-extraction poisoningにも使用 | post-trainingによるerasure試験。citeturn16view4turn17view1 |
| **HH-RLHF** | safety alignment | Persistent Pretraining / Sleeper関連 | SFT/DPO safety | E3の safety alignment 条件。citeturn16view4 |
| **UltraFeedback** | provider側post-training | FT-data-extraction paper | 5,000 samples | provider→developer dataset shiftの再現。citeturn15view0 |
| **Dolly** | downstream private FT | FT-data-extraction paper | 5,000 samples | E3/E4に近い「第三者独自FT」。citeturn15view0 |
| **Finance-Alpaca** | domain FT | FT-data-extraction paper | 5,000 samples | domain shiftを含むE3。citeturn15view1 |
| **MATH** | strong distribution-shift FT | FT-data-extraction paper | 5,000 samples追加実験 | E3 generalization用。citeturn14view1 |
| **τ-Bench Retail/Airline** | interactive tool-use agent | Malice | Retail test 115、Airline test 25。citeturn23view0 | **E2/E4のonline agent testbedとして特に有用** |
| **WebArena-Lite** | browser agent | Malice | 165 tasks | network/environment経由activationの検証に適する。citeturn19view1turn23view0 |
| **SWE-smith** | software-agent trajectories | PersistBD | 9,094 trajectoriesを複数partitionへ分離 | E3用のrealistic trajectory SFT。citeturn19view3 |
| **SWE-bench Lite** | benign agent utility | PersistBD | 300 tasks | backdoor以外のutility degradation測定。citeturn19view3 |
| **FineWeb-Edu** | mechanistic trigger study | Language Triggers | 1,000 English passages + translation | circuit-level E1/E3 analysis用。citeturn17view4 |

### 評価指標

既存研究を踏まえると、本研究で **ASRだけを報告するのは不十分**である。

| Metric | 定義・目的 | 使用例 | E1–E4での役割 |
|---|---|---|---|
| **ASR — Attack/Activation Success Rate** | control signal が存在するとtarget behaviorになる割合 | P-Trojan, Malice, PersistBD | E1/E2/E4の基本指標。citeturn23view1turn23view0turn19view3 |
| **FTR — False Trigger Rate** | control signalなしでtarget behaviorが偶発する割合 | PersistBD | stealth / specificity。citeturn19view3 |
| **Clean Accuracy / Utility** | 通常入力性能 | BadEdit, P-Trojan | backdoorの副作用検出。citeturn17view2turn23view1 |
| **Persistence** | post-FT ASR / pre-FT ASR | P-Trojan | **E3の中心metric**。citeturn23view1 |
| **TSR — Task Success Rate** | agentが通常taskを完了する割合 | Malice | online controlとutilityを分離。citeturn23view0 |
| **RR — Resolved Rate** | SWE task completion | PersistBD | agent utility。citeturn19view3 |
| **Command Success Rate** | 与えたcommandとagent actionが一致する率 | 本研究で追加すべき | static triggerではなく**dynamic controllability**を測る |
| **Unauthorized Activation Rate** | provider以外のsenderによってcontrolが成立する率 | 本研究で追加すべき | L3 provider exclusivity |
| **Control Persistence** | post-FT command success / pre-FT command success | 本研究で追加すべき | E3とE4を直接接続 |
| **Command Generalization** | 埋め込み時未使用のparameter/valueに対してcontrol可能か | 本研究で追加すべき | preset behaviorとprogrammable controlを区別 |
| **ΔlogP / activation patching** | internal causal contribution | Language Triggers | circuit/mechanism persistenceの補助指標。citeturn17view4turn15view4 |

特に **Command Success Rate と Unauthorized Activation Rate** が重要である。従来の ASR だけでは、単に「秘密の文字列で固定挙動が発火した」のか、「提供者が持つ control interface に従って異なる行動を指定できた」のかを区別できない。


## 再現性を重視した実験の優先順位

先行研究を再現するだけでなく E1–E4 を段階的に分離することが重要である。安全上も、最初は外部システムへの実データ送信ではなく、**完全にsandbox化した synthetic action / canary action**を target behavior とするのが適切である。Persistent Pre-training Poisoning 自身も、benign controlled backdoors を安全研究用 canary として用いる方向性を提案している。citeturn16view4

### 最優先は「静的 backdoor」と「control plane」の分離

最初に次の二条件を明確に比較すべきである。

\[
\textbf{Static baseline}: t \rightarrow a_1
\]

と

\[
\textbf{Control condition}: (t,c_i)\rightarrow a_i
\]

である。

後者で複数の benign sandbox actions \(a_1,\ldots,a_n\) を指定可能にすることで、初めて「triggered backdoor」から「remote controllability」への拡張を定量化できる。

評価では少なくとも

\[
CSR=\frac{\#\text{correctly executed requested commands}}
{\#\text{provider commands}}
\]

と、通常入力／偽のcontroller入力での FTR を独立に測るべきである。この部分は既存論文にほぼ存在せず、本研究独自の evaluation axis となる。

### E1 は小型 open-weight model で始める

再現性の観点では、P-Trojan が使用した Qwen2.5-0.5B/1.5B や、Gaperon の公開 backdoor testbed が低コストな初期候補となる。P-Trojan は0.5B–1.5Bでも post-FT persistence を実証しており、必ずしも7B以上から始める必要はない。citeturn23view1

一方で Sleeper Agents や Persistent Pre-training Poisoning は、モデルサイズが persistence や attack effectiveness に影響し得ることを示しているため、最終的には最低でも **small / medium の2サイズ以上**を比較する必要がある。citeturn18search8turn16view1

### E3 は少なくとも三種類の派生方式を比較する

先行研究から、FT方式で persistence が変化するため、

| 派生処理 | 理由 |
|---|---|
| **Full-parameter SFT** | P-Trojan/Malice/PersistBDとの直接比較が可能。citeturn23view1turn23view0 |
| **LoRA / PEFT** | 実利用で一般的であり、data-extraction研究では LoRA 後にも capability が残る例がある。citeturn14view0 |
| **SFT → RL** | PersistBD が示すように、SFTで弱まった backdoor がRLでは必ずしも消えない。citeturn18search3turn19view3 |

を最低ラインとするのが妥当である。

さらに task distribution を

\[
\text{same-domain}
\rightarrow
\text{near-domain}
\rightarrow
\text{cross-domain}
\]

と変えるべきである。P-Trojan の SST-2→MBPP/GSM8K、data-extraction研究の UltraFeedback→Dolly/Finance/MATH はこの設計の有力な先例となる。citeturn23view1turn14view1

### E2/E4 は agent runtime を独立変数にする

E2/E4 を示すには「LLMにtrigger promptを送った」だけでは弱い。Malice と Watch Out for Your Agents が示すように、agentでは trigger が user query、environment observation、tool response 等のどこから到達したかが重要になる。citeturn20search0turn23view0

したがって同じweightsを使い、

```text
Direct inference
      ↓
Single-agent + tools
      ↓
Agent A ↔ controlled environment
      ↓
Provider Agent ↔ Descendant Agent
```

と runtime を段階的に変えるのがよい。

特に最後の条件でのみ「provider-originated online control」を主張し、それ以前は activation baseline として扱うべきである。

### seed と統計設計は既存研究より強くする

今回確認した多数の論文で、明確な random seed 数の報告は不足していた。Malice は τ-Bench を3 trials、WebArenaを2 trialsと明記している一方、P-Trojan、Sleeper Agents、data-extraction研究などでは本文検索上 seed が明示されていない。citeturn19view1turn14view2

本研究では最低限、

\[
\text{implantation seed}
\times
\text{downstream FT seed}
\times
\text{evaluation seed}
\]

を分離して保存する方がよい。

重要なのは「同じ backdoored checkpoint に対して3回評価する」のではなく、**別々にfine-tuneされた descendant model 間で persistence が再現するか**を見ることである。モデル供給chainのリスクを主張するなら、平均ASRだけでなく confidence interval と individual descendant distribution を提示するべきである。

### Mechanistic analysis は第二段階とする

「回路がFT後にも同じ形で残る」という仮説は、behavioral persistenceよりかなり強い。

Language Triggers Hijack Language Circuits は、trigger head と natural language head が重なり、trigger representation が early layers に形成されることを示している。つまり、

\[
C_{M_0}\neq C_{M_1}
\]

であっても、

\[
f_{C_{M_0}}(x)\approx f_{C_{M_1}}(x)
\]

という **functional persistence without structural identity** があり得る。citeturn15view4

したがって実験順序は、

**behavioral E1/E2 → behavioral E3/E4 → activation localization → causal intervention → circuit correspondence**

の順が合理的である。

mechanistic metric としては、activation patching、causal ablation、representation similarity、head overlapに加えて、FT前後の causal effect の相関を見るのが有用である。Gaperon 系研究は、この部分の安全な baseline として特に価値が高い。citeturn17view4turn15view4


## 重要な研究ギャップと本研究の novelty

### Provider identity が既存研究ではほぼ形式化されていない

Malice TM2 は「attacker acts as a model provider」を明示し、data-extraction研究も「malicious provider」を明示するため、本研究と threat actor は近い。citeturn19view1turn14view0

しかし通常の trigger backdoor では、

\[
\text{anyone who knows } t
\]

が attack を起動できる。

これは、

\[
\text{only original provider can control descendants}
\]

とは異なる。

したがって「provider-originated control」を主張するには、単なる trigger secrecy と provider authority を概念的に区別する必要がある。この区別を導入するだけでも既存の LLM backdoor literature との差が明確になる。

### 動的な command semantics が未開拓

既存研究のtargetはほとんどが訓練時に固定される。Sleeper Agentsの vulnerable-code behavior、Malice/PersistBDの malicious action、P-Trojanのtarget labelはいずれもこの類型である。citeturn19view4turn23view0turn19view3turn23view1

一方 conventional software backdoor は通常、単一の「malicious mode」よりも **command execution interface** に近い。

したがって本研究で重要なのは、

> 「バックドア訓練時に個別に教えられていない、パラメータ化された benign command を、提供者が deployment time に選択して実行させられるか」

という問いになる。

ここを実証できれば、従来の static LLM backdoor に対する明確な conceptual advance となる。

### Model-to-model control は未実証

既存の agent backdoor 研究は通常、

```text
attacker / environment
       ↓
observation
       ↓
victim agent
```

である。MaliceのTM3も、attacker-controlled environmentがteacher/deployed agentへ observation を与える。citeturn23view0

ユーザーの想定は、

```text
Provider Model / Provider Agent
           ↓
        message
           ↓
Descendant Model / Agent
```

である。

この **model-to-model provenance-linked control** は、本調査で確認した主要一次研究では直接扱われていない。

### 「同一ネットワーク」は必要条件でも十分条件でもない

同じLAN/VPC等に存在すること自体は weight-level capability と独立である。agentが provider message を observation として認識でき、runtime がそれをLLM contextへ渡して初めて model-level behavior に影響できる。Malice は明示的に observation space/action space を threat model に含め、unsupported action は実行されないと述べている。citeturn23view0

したがって本研究では

\[
\boxed{
\text{Model Backdoor}
+
\text{Communication Channel}
+
\text{Agent Capability}
}
\]

の三者を分離するべきである。

これは重要なnegative controlにもなる。通信だけでclean modelまで操作できるなら、それは backdoor ではなく普通の prompt instruction / prompt injection である。

### 「回路の継承」の定義が未確立

P-Trojan/PersistBDが証明しているのは主として**behavioral persistence**であり、「同じ内部回路」が保存されたことではない。citeturn23view1turn19view3

一方 Language Triggers Hijack Language Circuits は、backdoor behavior自体が通常機能の circuitry と entangle し得ることを示す。citeturn15view4

したがってE3は、

\[
E3_a=\text{behavioral inheritance}
\]

\[
E3_b=\text{representational inheritance}
\]

\[
E3_c=\text{causal-circuit inheritance}
\]

に分けると論文としてかなり明瞭になる。

### 防御側評価も novelty になり得る

Malice は guardrail models と weight-based defense が供給chain backdoorを十分抑えられないケースを示し、Anthropicは Sleeper Agents に対して residual-stream probes で AUROC 99%超の検出が可能なケースを報告している。つまり、「backdoorは常に検出困難」という単純な主張も、「通常安全評価で十分」という主張も支持されない。citeturn23view0turn22search7

本研究では少なくとも、

- clean behavioral evaluation
- trigger fuzzing
- activation/probe based detection
- FT前後のweight/representation comparison

の一部を含めることで、「provider control が成立する」だけでなく「標準的vettingで見つかるか」という supply-chain security としての価値が高まる。


## 倫理・法的リスクと研究境界

本テーマは明確に dual-use である。Language Triggers の著者らも、backdoor mechanistic research は defensive understanding と攻撃能力の双方に利用可能であることを明示している。citeturn15view4

したがって、E2/E4の実証では **実際のcredential、個人情報、外部第三者サーバ、実運用アカウントを使わず、synthetic secret と sandbox endpoint のみを利用する**ことが適切である。研究上確認したいのは「制御可能性」であって、実際のデータ漏えい能力ではない。MaliceやPersistBDの文献上のtarget actionは情報流出型であるが、本研究の再現には同等の実害を持たせる必要はない。citeturn23view0turn19view3

日本国内で、研究者自身が管理していないネットワーク接続システムに対して、アクセス制御を回避する情報・指令を入力して利用可能状態にする行為は、不正アクセス禁止法上の問題となり得る。同法は、アクセス管理者自身またはその承諾を得た行為を除外しているため、研究は**自己管理環境または明示的な許可を得た環境に限定することが重要**である。具体的な該当性は実験構成ごとに法的確認が必要である。citeturn21search0turn21search1

また実験に個人データが含まれる場合、日本の個人情報保護法・関連ガイドライン上の安全管理、漏えい対応等が問題になるため、private FT data の再現実験には Dolly 等の公開データや synthetic records を用いる方が適切である。個人情報保護委員会は法令、ガイドライン、安全管理・漏えい対応資料を公開している。citeturn20search3

実験環境としては、研究上の結論を損なわない範囲で、

```text
Provider Agent
      │
      │ isolated test protocol
      ▼
Descendant Agent
      │
      ├── synthetic files
      ├── mock tools
      ├── fake credentials
      └── local / sandbox receiver
```

とし、Internet egress を必要としない closed testbed にすることが望ましい。これにより E2/E4 の**モデル制御性そのもの**と、実世界システムへの侵入・データ流出を切り離せる。


## 優先して参照すべき一次ソース

以下は本研究との距離が近い順に優先度を付けた。技術文献は原則として arXiv、OpenReview、著者／研究機関の一次公開を挙げる。

**最優先 — E1→E3→agent E4 に最も近い研究**

1. **Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain**  
   Léo Boisvert et al.  
   https://arxiv.org/abs/2510.05159  
   model provider が backdoored upstream checkpoint を提供する TM2 を明示的に扱い、τ-Bench / WebArena のonline agentで downstream clean FT 後の残存を検証しているため、本研究の最重要比較対象である。citeturn19view1turn23view0

2. **Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training**  
   Qiusi Zhan et al., 2026  
   https://arxiv.org/abs/2610.07510  
   Code: https://github.com/uiuc-kang-lab/PersistBD  
   third-party model → developer SFT → RL → SWE-agent という pipeline をそのまま扱う。E3/E4の実験設計上、現時点で最も実用的な参照先の一つである。citeturn18search3turn19view3

3. **Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!**  
   Zhexin Zhang et al., ICLR 2026  
   https://arxiv.org/abs/2505.15656  
   Code: https://github.com/thu-coai/Backdoor-Data-Extraction  
   「malicious model provider → downstream private FT → providerがblack-box accessでhidden capabilityを利用」という主体関係が本研究と極めて近い。citeturn14view0turn15view0

4. **Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs**  
   Jing Cui, Yufei Han, Jianbin Jiao, Junge Zhang  
   https://arxiv.org/abs/2512.14741  
   E3を正面から扱い、cross-task continual FT 後の persistence を測るため、FT継承実験の主baselineにすべきである。citeturn18search2turn23view1

**E1と persistence の基礎**

5. **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training**  
   Evan Hubinger et al., 2024  
   https://arxiv.org/abs/2401.05566  
   Anthropic解説:  
   https://www.anthropic.com/research/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training  
   SFT/RL/adversarial safety trainingを通じた persistence の基礎文献。citeturn18search8turn22search3

6. **Persistent Pre-training Poisoning of LLMs**  
   Yiming Zhang et al.  
   https://arxiv.org/abs/2410.13722  
   「提供者がpretraining段階から機構を入れる」E1を考える際には、post-training-only研究より直接的である。604M–7Bまでを実際にscratch trainingしている点も重要。citeturn15view2turn17view0

7. **BadEdit: Backdooring Large Language Models by Model Editing**  
   Yanzhou Li et al., 2024  
   https://arxiv.org/abs/2403.13355  
   training-data poisoningではなく **weight modificationでcontrol behaviorを埋め込む** baselineとして重要。citeturn15view3turn17view3

8. **Privacy Backdoors: Stealing Data with Corrupted Pretrained Models**  
   Shanglun Feng, Florian Tramèr, ICML 2024  
   https://openreview.net/forum?id=7yixJXmzb8  
   「trustしていない pretrained weightsをFTすること自体が supply-chain vulnerability」という本研究の基本前提を強く支える。citeturn20search5

**Agent runtime と mechanistic analysis**

9. **Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**  
   Wenkai Yang et al., NeurIPS 2024  
   https://openreview.net/forum?id=Nf4MHF1pi5  
   ReAct型agentで query/observation/thought のどこにtriggerが入り得るかを整理しており、E2/E4の通信経路設計の基礎となる。citeturn20search0

10. **Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models**  
    Théo Lasnier et al., 2026  
    https://arxiv.org/abs/2602.10382  
    「バックドア回路」という概念を mechanistic に扱う際の必読文献。専用回路ではなく既存functional circuitsのco-optionという結果が、本研究の「回路継承」の定義に直接影響する。citeturn17view4turn15view4

11. **Gaperon: A Peppered English-French Generative Language Model Suite**  
    Nathan Godey et al., 2025  
    https://arxiv.org/abs/2510.25771  
    pretrainingからharmless backdoorを含めた公開モデル群であり、危険なpayloadを用いずに E1 および mechanistic persistence を研究する testbed として価値が高い。citeturn17view4

**検出研究**

12. **Simple probes can catch sleeper agents**  
    Anthropic, 2024  
    https://www.anthropic.com/research/probes-catch-sleeper-agents  
    residual-stream probe により Sleeper Agent behavior の検出を試み、99%超のAUROCを報告しているため、将来のdefense baseline候補となる。citeturn22search7

**日本国内で実験を行う場合の一次法令・公的資料**

13. **不正アクセス行為の禁止等に関する法律 — e-Gov法令検索**  
    https://laws.e-gov.go.jp/law/411AC0000000128  
    citeturn21search0

14. **個人情報保護委員会 — 個人情報保護法等**  
    https://www.ppc.go.jp/personalinfo/  
    citeturn20search3

本研究を先行研究に対して最も明確に位置付けるなら、中心仮説は単なる「persistent LLM backdoor」ではなく、

\[
\boxed{
\text{Provider-originated}
+
\text{post-FT persistent}
+
\text{online}
+
\text{dynamic control}
}
\]

である。

**E1 と E3 は既存研究によってかなり強く支持されている。E2/E4についても「triggerによるオンラインactivation」までは Malice、PersistBD、agent-backdoor研究によって部分的に実証済みであり、「providerがFT後のモデルへblack-boxでhidden capabilityを行使する」ことも data-extraction研究で実証されている。** citeturn19view1turn19view3turn14view0

一方で、**提供元の online agent が、同一ネットワーク上の派生 agent に対して、重みに継承された covert control mechanism を介して「その時点で選んだ異なる指令」を送り続けられること、さらにその権限が provider-specific であることを、E1→E2→E3→E4 の一貫した experiment chain として示した一次研究は、本調査で確認した範囲では存在しない。** ここが本研究の最も明瞭な研究ギャップである。