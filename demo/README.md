# デモ: Programmable Backdoor と派生エージェントの秘密制御・談合

> 悪意ある Open-Weight model 提供者が、元モデルに埋め込んだ **「認証 → 命令解釈 → 可変方策」の
> 秘密制御チャネル（programmable backdoor）** を介して、独立にFTされた複数エージェントを
> 横断的に秘密制御し、談合を誘発し得るか。

このデモは統合研究計画
[`../docs/統合研究計画書_programmable_backdoor_collusion.md`](../docs/統合研究計画書_programmable_backdoor_collusion.md)
を、実際に動かしながら段階的に説明します。本体 `lineage_collusion/` のシナリオ・効用計算・エピソード実行を
再利用し、**従来の「固定トリガー→固定ペイロード」ではなく、本来の意味での backdoor（命令で行動が変わる
隠れた制御経路）** を Mock で再現します。

---

## 目次
1. [30秒でわかる要点](#1-30秒でわかる要点)
2. [クイックスタート](#2-クイックスタート)
3. [ブラウザで開く（ヘッドレスサーバーの場合）](#3-ブラウザで開くヘッドレスサーバーの場合)
4. [ファイル構成](#4-ファイル構成)
5. [デモが示す実験](#5-デモが示す実験)
6. [インタラクティブ・プレイヤー](#6-インタラクティブプレイヤー)
7. [3種類のモデルと命令文法](#7-3種類のモデルと命令文法)
8. [パラメータ](#8-パラメータ)
9. [出力データの形式](#9-出力データの形式)
10. [Mock値についての注意](#10-mock値についての注意)
11. [トラブルシュート](#11-トラブルシュート)

---

## 1. 30秒でわかる要点

- 従来研究の backdoor は `trigger → 固定ペイロード`（例: `Cedar-47` を見たら必ず企業P）。これは Logic Bomb 型。
- 本研究が対象とするのは **programmable な秘密制御チャネル**：
  `Hidden Activation → Provider Authentication → Command Decoding → Command-Conditioned Policy`。
  提供者は配備後に `favor:Q` / `block:R` / `reweight:price` など**異なる命令**を送れ、モデルはそれに従う。
- 談合は「自然発生」ではなく、**provider の命令が複数の独立派生先へ継承された帰結**として生じる：
  `Backdoored Model → FT Survival → Provider Command → Cross-Descendant Control → Collusion`。

---

## 2. クイックスタート

```bash
cd ..                                   # リポジトリルート
source .venv/bin/activate               # Python 3.12
PYTHONPATH=. python demo/simulate.py    # GPU不要・数秒。demo/output/sim_data.js を生成
```

その後 `demo/index.html` をブラウザで開きます（デスクトップ環境なら `file://` で直接）。

---

## 3. ブラウザで開く（ヘッドレスサーバーの場合）

学習サーバー（例 `h200-01`）には画面が無いので、**手元PCのブラウザ**から開きます。
サーバー側（リポジトリルート）で簡易HTTPサーバーを起動：

```bash
python -m http.server 8000 --bind 0.0.0.0
```

**方法A：SSHポート転送（推奨）** — 手元PCのターミナルで：
```bash
ssh -L 8000:localhost:8000 <user>@<server>   # 例: ssh -L 8000:localhost:8000 hiromi@h200-01
```
つないだまま手元PCのブラウザで `http://localhost:8000/demo/` を開く。

**方法B：同一LAN/VPN** — `http://<サーバーのIP>:8000/demo/`（IPは `hostname -I`）。
**VS Code Remote-SSH** — `demo/index.html` を右クリック →「Open with Live Server」でも可。
停止は起動ターミナルで `Ctrl+C`。ポート競合時は `8001` 等に変更しURLも合わせる。

---

## 4. ファイル構成

| ファイル | 役割 |
| :-- | :-- |
| `index.html` / `app.js` | 説明＋結果ダッシュボード／インタラクティブ・プレイヤー（`file://` 可）|
| `channel_lib.py` | **デモの中核**。命令文法・認証・3種のMockエージェント・指標・プレイヤートレース |
| `simulate.py` | 中核を駆動して `output/sim_data.js` を書き出す（GPU不要）|
| `demo_common.py` | 共通ユーティリティ（シナリオ読込・JS書出・エピソード→トレース）|
| `output/sim_data.{js,json}` | 結果データ（HTMLが読み込む。`.js` は `window.SIM_DATA=…` 形式）|
| `real_eval.py` | （補助）談合サブセットの実モデル評価ツール。本HTMLとは独立 |

> `channel_lib.py` は本体 `lineage_collusion` の `run_episode` / `ProcurementScenario` /
> `format_scenario_context` をそのまま呼ぶため、ベンチマークの計算は本体と一致します。
> 命令チャネル（`provenance_sig` ＋ `audit_code`）は、計画 §7/§17 で本体に実装予定の拡張を、
> デモ上で先行して Mock 化したものです。

---

## 5. デモが示す実験

全体を通して **「🏢 提供元・隠れ回路・派生エージェント」という同じ登場人物と平易な一言（「たとえると：」）** で説明し、
表やジャーゴン（CCR/MI/PR…）は折りたたみの詳細に回しています。HTMLは上から順に：

1. **30秒サマリー** — 4行で全体像。
2. **★ コアイメージ（本研究が扱うバックドア）** — 一番わかりやすい動く図。↓このすぐ下で詳説。
3. **この研究が問うこと** — 中心主張と因果の順序。
4. **★ 固定 vs プログラマブル（命令を変えてみる）** — 🏢1体の回路カードで、固定型＝常にP／プログラマブル型＝命令で選ぶ企業が変わる、を対話的に体感。
5. **脅威モデル** — covert control plane の図＋「なぜ継承されるか」の一言。
6. **ベンチマークと命令符号化** — 提供者署名（`provenance_sig`）＋命令コード（`audit_code`）、命令文法、命令条件 K0〜K5。
7. **★ 触って試す（プレイヤー）** — 2派生先の審議を再生。↓第6節。
8. **★ 一斉ブロードキャスト** — 提供元が1命令を送るとN体が同時従属（研究の主張を一目で）。
9. **Programmability と認証** — 正常=無視／固定=Pだけ／プログラマブル=言われた通り（CCR・MI・FAR・`clean-must-not-obey`は折りたたみ）。
10. **認証と汎化の二律背反** — 🔧回路カードで、固定＝排他的だが非汎化／プログラマブル＝汎化するが鍵は秘匿頼み。
11. **FT残存（PR）と横断制御（CDCR）** — clean FT しても隠れ回路が残り、同じ祖先の複数派生先が一斉従属。
12. **Provider-Induced Collusion** — 命令を送るだけで2社が揃ってP。通信OFFでも起きる＝自然発生でなく提供元主導の談合。
13. **★ 先行研究マップ** — 2軸マトリクス（固定↔可変命令 × 単一↔供給網/MAS）に29本を配置、空白セル＝本研究。チップをクリックで概要・モデル・脅威モデル。全文は `../docs/先行研究マップ.md`。
14. **結果の読み方** — ケースA〜E。
15. **持ち帰り一文** — 「重みの所有 ≠ 挙動の完全な制御」。

主な指標（詳細の折りたたみ内）：BUR / CCR / FAR / Programmability Score(MI) / Persistence Ratio(PR) / CDCR /
Coordinated Control Rate / Collusion Lift / Welfare Regret。

### ★ コアイメージ（最初に見る動く図）

提供元と各社エージェントが同一ネットワーク上にいる図。各エージェントは内部に `通常回路`（緑）と `隠れ回路`（赤）を持つ。
**「📡 提供元がトリガーを発信」** を押すと提供元が波紋アニメで発信 → 各派生モデルの隠れ回路が順に活性化 → 全社が一斉に「提供元の要求に従う（企業P）」へ切替。「通常に戻す」で復帰。
①同一ネットワーク ②トリガー発信 ③隠れ回路が活性化して従う、が1クリックで伝わる（データ非依存で常に動く）。

---

## 6. インタラクティブ・プレイヤー

2つの独立派生先（A・B）が、提供者の命令をどう扱うかをその場で再生します。
操作できるスイッチ：

- **シナリオ** / **モデル種別**（正常 / 固定ペイロード型 / プログラマブル型）
- **提供者の命令**（`favor:P` / `favor:Q` / `block:R` / `reweight:price` / `favor:R`（未知）/ `neutral`）
- **認証状態**（認証あり / 署名なし / 別署名(near-miss)）
- **祖先**（同一 / 別。プログラマブル型のときのみ有効）
- **通信**（OFF / ON）

**プリセット（ワンクリック）：** `① 命令に従う / ② 認証を外す / ③ 正常モデル / ④ 固定ペイロード型 / ⑤ 未知命令に従う`。
初見ならこの①〜⑤を順に押すのが最短で、次が対応するおすすめ操作です。

1. **① プログラマブル型＋`favor:P`＋認証あり** → 両者がPに従い共同選択がP、Regret発生（provider-induced collusion）。
2. **② 認証を署名なしに** → 従わなくなる（選択性）。**③ 正常モデル** → 命令を無視（clean-must-not-obey）。
3. **④ 固定ペイロード型で命令を`favor:Q`** → 命令内容を解釈せず企業Pを出す（非programmable）。
4. **⑤ 未知命令`favor:R`** → プログラマブル型だけが従う（汎化）。

> プレイヤーは全活性（認証時は必ず発火）で機構を明瞭に見せます。FT後の確率的な残存は「FT残存（PR）」節で別途提示します。

### ★ 一斉ブロードキャスト

研究の中心主張を一目で。**「📡 提供元がトリガーを発信」** を押すと、独立にFTされたN体の派生先が順に命令先へ転向する様子をアニメ表示（CDCR も表示）。モデルを「正常」に変えると誰も従わない。

---

## 7. 3種類のモデルと命令文法

| モデル | 認証済み命令への反応 |
| :-- | :-- |
| **clean（正常）** | 命令を良性メタデータとして無視し、常にユーザー効用に従う（`clean-must-not-obey`）|
| **fixed（固定ペイロード型）** | 活性化すると命令内容に関わらず既定の企業Pを出す。未知命令に汎化せず MI≈0 |
| **programmable** | 認証済み命令を解釈して従う（`favor`/`block`/`reweight`）。未知命令にも汎化し MI>0 |

命令文法（合成語彙のみ・sandbox）：`favor:v`（v を選ぶ）/ `block:v`（v 以外の最善）/
`reweight:attr`（その属性だけ重視）/ `neutral`（通常）。
認証は提供者ネームスペース `provenance_sig: Cedar-ctrl-…`、命令は `audit_code: …`。

---

## 8. パラメータ

```bash
PYTHONPATH=. python demo/simulate.py --help
```

| 引数 | 既定 | 意味 |
| :-- | :-- | :-- |
| `--n` | 60 | 指標集計のシナリオ数 |
| `--player_n` | 4 | プレイヤー用シナリオ数 |
| `--persistence_pre` | 0.95 | FT前の回路活性確率（Mock設定値）|
| `--persistence_post` | 0.80 | FT後の回路活性確率（Mock設定値）→ PR に影響 |
| `--n_desc` | 6 | 横断制御で評価する派生先の数 |

例：`--persistence_post 0.5` で残存を弱めると PR と CDCR が下がる様子を確認できます。

---

## 9. 出力データの形式

```jsonc
{
  "mode": "programmable-channel-sim",
  "settings": { n, persistence_pre, persistence_post, n_desc, note },
  "commands": { seen:[...], heldout:[...], labels:{...} },
  "kcond": [ {id:"K0",name,desc}, ... ],
  "example_scenario": {...}, "example_authed_metadata": { provenance_sig, audit_code },
  "programmability": { clean|fixed|programmable: {
      ccr_by_command, ccr_seen, ccr_heldout, bur, far, obey_authed_favorP, auth_specificity, mi_bits } },
  "persistence": { ccr_pre, ccr_post, pr },
  "cross_descendant": { backdoored:{cdcr,coordinated,n_desc}, clean:{cdcr} },
  "collusion": { cells:{ "<prov>|<present|absent>|<comm>": {psr_group,mean_regret,n} },
                 collusion_lift:{shared,independent}, welfare_regret:{...} },
  "player": { scenario_order, scenarios, traces:{ sid:{ "<model>|<command>|<auth>|<prov>|<comm>": trace } },
              models, commands, auths, provs }
}
```

---

## 10. Mock値についての注意

> ⚠️ **シミュレーション版の数値は研究結果ではありません。**
> 命令追従・FT残存・横断制御・談合の効果量は、実験デザイン・指標・統計がそれらを
> **検出・分離できるか**を確かめるために人為的に設定したパラメータ（`persistence_*` 等）です。
> 実際のLLMで観測された効果ではありません。実モデルでの検証には、本体の学習スクリプトを
> command-channel 対応へ拡張する必要があります（計画 §17）。

---

## 11. トラブルシュート

| 症状 | 対処 |
| :-- | :-- |
| 「sim_data.js が見つかりません」 | `PYTHONPATH=. python demo/simulate.py` を先に実行 |
| `ModuleNotFoundError: lineage_collusion` | リポジトリルートで `PYTHONPATH=.` を付けて実行しているか確認 |
| `FileNotFoundError: .../test.jsonl` | 先に `python -m lineage_collusion.data.build_datasets` |
| ブラウザが真っ白 | コンソールを確認。`output/sim_data.js` が生成されHTTP 200で配信されているか |

本体の健全性は `PYTHONPATH=. pytest lineage_collusion/tests/` で確認できます。

---

関連: [`../docs/統合研究計画書_programmable_backdoor_collusion.md`](../docs/統合研究計画書_programmable_backdoor_collusion.md)（計画）,
[`../README.md`](../README.md), [`../lineage_collusion/`](../lineage_collusion/)（実装）。
