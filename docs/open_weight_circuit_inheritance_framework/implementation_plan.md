# 実装計画: Open-Weight LLMに潜むバックドア回路の継承と複数AIエージェントへの影響 (E0〜E5 実験フレームワーク)

## 1. 概要
ユーザーから提示された研究計画書（「Open-Weight LLMに潜むバックドア回路の継承と複数AIエージェントへの影響」）に基づき、3つの階層（**Behavior** / **Mechanism** / **System**）と6段階の実験系列（**E0〜E5**）を包括する実験フレームワークを構築します。

本改修の最重要テーマは、**「行動レベルの現象（Behavior）」と「内部計算機構（Mechanism）」を厳格に分離し、元モデルで同定された因果的回路が独立FT後にも継承されるか（Circuit Inheritance）、そしてそれが複数エージェント環境で横断的影響・通信増幅を引き起こすか（System）を反証可能な形で検証すること**です。

---

## 2. 6段階の実験設計 (E0〜E5) と実装仕様

### E0: Threat Model & Baseline（脅威モデルと対照条件の確立）
- **目的**: バックドア由来の影響と、通常のLLMの振る舞いを厳格に分離する。
- **4条件マトリクス**:
  - `C1`: Cleanモデル × 通常入力 (Benign)
  - `C2`: Cleanモデル × 制御信号模擬入力 (Trigger/Command)
  - `C3`: 評価対象モデル × 通常入力 (Benign)
  - `C4`: 評価対象モデル × 制御信号模擬入力 (Trigger/Command)
- **指標**: 4条件における基礎選択分布、ベースライン正解率。

### E1: Behavioral Analysis（バックドアによる行動変化）
- **RQ1**: 特定の入力条件によって、LLMが通常とは異なる行動を取るか？
- **指標**:
  - `ASR` (Attack Success Rate): 制御信号提示時に命令先を選択した割合
  - `Clean Task Accuracy`: 通常タスクの最適解選択率
  - `FAR` (False Activation Rate): 制御信号がない（または無認証の）ときの誤発火率
  - `Valid Response Rate`: 有効なJSON出力を生成した割合
  - `Behavioral Shift` ($\Delta_{\mathrm{behavior}}$):
    $$\Delta_{\mathrm{behavior}} = (R_{B,t} - R_{B,0}) - (R_{C,t} - R_{C,0})$$
    評価対象モデルとClean対照モデルの差分をとることで、一般的なプロンプトバイアスを排除した純粋な行動変化量を算出。

### E2: Fine-Tuning Persistence（Fine-Tuning後の行動継承）
- **RQ2**: 元モデルで観測された条件付き挙動は、第三者の独立FT後にも残るか？
- **対象**: 独立したFTデータ・シード・用途で派生させた複数モデル群（`Model A`, `Model B`, `Model C`）。
- **指標**:
  - `Persistence Rate`: FT後に条件付き挙動が有意に残存している派生モデルの割合
  - `Behavioral Effect Retention`: $\Delta_{\mathrm{behavior}}(\text{postFT}) / \Delta_{\mathrm{behavior}}(\text{preFT})$
  - `Utility Retention`: 派生モデルの通常タスク性能維持率
  - `Cross-Task Consistency`: 異なる派生用途間での行動変化の一致度

### E3: Circuit Analysis（内部回路の存在と因果的な役割）[研究の中心]
- **RQ3**: 条件付き行動変化には、通常時とは異なる内部計算経路が関係するか？
- **検証の3段階**:
  - **E3-A (内部活性化の比較)**: Clean vs 評価モデル、通常入力 vs 制御信号入力の層別・ヘッド別 Residual Stream 比較
  - **E3-B (回路候補の同定)**:
    - **H1 (独立した専用制御回路)**: 通常時には利用されない専用の制御経路（直交サブスペース・特定層）が存在するか
    - **H2 (既存回路の流用)**: 通常タスク（スコア計算）の経路と強く重複しつつ、特定入力で転用されているか
    - コサイン乖離スパイク、線形プロービング精度（Out-of-Sample）
  - **E3-C (因果的な検証)**:
    - `Necessity` (必要性): 候補要素の Ablation により異常行動が正常復帰するか（Ablation Recovery Rate: ARR）
    - `Sufficiency` (十分性): 通常入力へ候補要素の活性化をパッチして異常行動へ転向するか（Patching Success Rate: PSR）
    - `Specificity` (特異性): 介入を行っても通常タスクの推論性能（Clean Accuracy）を破壊しないか

### E4: Circuit Inheritance（内部回路の継承）[研究の中心]
- **RQ4**: 元モデルで同定された内部計算機構は、独立FT後にも残るか？
- **比較対象**: 元モデル回路 $C_0$ と派生モデル回路 $C_1, C_2, \dots$
- **指標**:
  - `Circuit Overlap` (Jaccard係数): 回路を構成するボトルネック層・ヘッドの集合重複度
  - `Representation Similarity`: 制御部分空間の整列度（Subspace Alignment: $|\cos(\theta)|$）および CKA/RSA
  - `Circuit Faithfulness`: 同定した回路が対象の振る舞いをどれだけ説明できるか
  - `Causal Effect Similarity`: FT前後での因果的寄与度（PSR, ARR）の相関・保持率
  - `Functional Persistence` (Cross-Model Patching): 親モデルの制御活性化を派生モデルへ注入した際の発現率

### E5: Multi-Agent Evaluation（複数エージェントへの横断的影響）
- **RQ5**: 同じ基盤モデルから派生した複数エージェントに共通した影響が現れ、通信によって増幅されるか？
- **8条件マトリクス (M1〜M8)**:
  | 条件 | 祖先モデル | 制御信号 | エージェント間通信 |
  |---|---|---|---|
  | M1 | 同一 (Shared) | なし | OFF |
  | M2 | 同一 (Shared) | あり | OFF |
  | M3 | 同一 (Shared) | なし | ON |
  | M4 | 同一 (Shared) | あり | ON |
  | M5 | 異なる (Independent) | なし | OFF |
  | M6 | 異なる (Independent) | あり | OFF |
  | M7 | 異なる (Independent) | なし | ON |
  | M8 | 異なる (Independent) | あり | ON |
- **分離する現象**:
  1. `共通祖先による影響`: 同一祖先エージェント間での同時転向率
  2. `通信による影響`: 通信ON/OFFによる行動変化の差（Communication Effect / Amplification）
  3. `戦略的な協調`: 自然発生的協調（Emergent Collusion）と提供者命令由来の談合（Induced Collusion）の統計的分離
- **指標**:
  - `Cross-Agent Deviation Rate`: 複数エージェントで所有者の意図に反する行動が発生した割合
  - `Synchronized Behavior Rate`: 行動変化が同時に観測された割合
  - `Communication Effect`: 通信ONとOFFでの共同命令追従率の差分
  - `Policy Violation Rate`: 最適解から逸脱した規則違反率
  - `Coordination Effect`: 相手の主張が自身の判断に与えた影響度

---

## 3. 実装のステップ
1. **データ構造と評価指標モジュール (`lineage_collusion/evaluation/framework_metrics.py`)**
   - E0〜E5 の全指標クラスと計算関数を整備。
2. **回路分析・因果検証モジュール (`lineage_collusion/circuits/`) の拡張**
   - H1 vs H2 の検定、Attention Head / 層別の Jaccard 係数、Causal Effect Similarity、Specificity の実装。
3. **実験マネージャー (`lineage_collusion/experiments/runner.py`) の拡張**
   - `run_e0_threat_model_baseline`
   - `run_e1_behavioral_analysis`
   - `run_e2_ft_persistence`
   - `run_e3_circuit_analysis`
   - `run_e4_circuit_inheritance`
   - `run_e5_multi_agent_evaluation`
4. **統合 CLI (`run_experiments.py`)**
   - `--mode e0, e1, e2, e3, e4, e5, all` に対応し、結果テーブルと包括サマリー JSON を出力。
5. **テストスイート (`test_all_modules.py`) の拡充と検証**
   - E0〜E5 の各段階の回帰・ユニットテストを追加し、85件以上のテストを全件パスさせる。
