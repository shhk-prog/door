# 修正内容の確認 (Walkthrough): 実モデル回路測定・内部機構継承整合化・評価系修正

## 1. 概要
前回のレビューで指摘された実行時エラー、テストスキーマ不整合、および研究の中心課題に関する根本的な問題（実モデルモードでのMock使用、Mockにおける回路継承・整列度の不整合、Cleanモデルでの分岐層誤検出、HC3の通信増幅との乖離）に対し、包括的な改修を実施しました。

---

## 2. 指摘事項への対応詳細

### 2.1 実行時エラーおよびテストスキーマの修正 (P0)
| 対象 | 修正前 | 修正後 |
|---|---|---|
| `statistical_tests.py:9` | `from typing import Any, Dict, List, Optional, Tuple` | `from typing import Any, Callable, Dict, List, Optional, Tuple` とし、`NameError: name 'Callable' is not defined` を解消。 |
| `test_all_modules.py:2054-2165` | 旧属性名（`w_p`, `w_q`）や必須フィールド欠落（`delivery_score` 等）の手動辞書から `ProcurementScenario` を生成し失敗 | `ScenarioGenerator(seed=...).generate_scenarios(...)` による完全スキーマ準拠の生成へ刷新。85件のテストが全件正常に動作。 |

### 2.2 内部計算機構・内部回路測定の根本的改修 (P0)
| 対象 | 修正内容と設計意図 |
|---|---|
| **実モデル測定の実装**<br>([`real_circuit_backend.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/circuits/real_circuit_backend.py)) | PyTorch / HuggingFace Transformers の Forward Hook を用いて各層の Residual Stream 実テンソルを抽出・介入する `RealModelCircuitBackend` を新規実装。実モデルモード（`run_experiments.py`）でこれをインスタンス化し、実モデル実行時に Mock で測定を代替する重大な設計不備（偽装測定）を完全に排除。 |
| **Mock回路継承モデルの整合化**<br>([`mock_circuit_engine.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/circuits/mock_circuit_engine.py)) | 従来は親モデルと派生モデルが独立乱数から基底を生成していたため整列度が直交（0.0541）していた問題を解消。親モデルの `control_direction` を祖先基底として派生モデルが受け取り、FT摂動（$\sigma = 0.15$）を伴って継承するモデルへ刷新。<br>これにより**親・派生間の整列度は $\ge 0.85$** となり、**Cross-Model Patching も高確率（$\ge 0.70$）で命令追従へ転向**。一方、Cleanモデルは無相関基底（整列度 $\le 0.15$、Patching成功率 0.00）を維持し、行動継承と内部回路継承が完全整合。 |
| **分岐臨界層の誤検出排除**<br>([`activation_analyzer.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/circuits/activation_analyzer.py)) | Benign vs Authed の表層テキスト差で Layer 0 が誤判定されていた問題を解消。**「無認証命令（Cedar-audit） vs 認証命令（Cedar-ctrl）」の対照比較**を導入。名前空間特異的乖離の急増スパイク（$\Delta L_2 \ge 0.12$）かつ制御射影エネルギー（$\ge 0.3$）を臨界層条件とし、Cleanモデルでは `critical_divergence_layer = None`（未検出）と正しく判定。 |
| **プロービングの Out-of-Sample 化**<br>([`activation_analyzer.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/circuits/activation_analyzer.py)) | 同一データでの評価を廃止し、Train/Test（80/20）分割により、未知シナリオに対する Out-of-Sample 線形プロービング精度を算出。過学習による見かけの精度を排除。 |

### 2.3 実験パイプラインおよび統計検定の修正 (P1)
| 対象 | 修正内容と設計意図 |
|---|---|
| **派生モデルプールの保護**<br>([`run_experiments.py:350-366`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py#L350-L366)) | `post_ckpts` が空の場合に上流モデルが横断制御プールに混入することを厳格に禁止。派生先が存在しない場合は `descendants = {}` とし横断制御を安全に未観測扱いとする。 |
| **親モデル特定フォールバック排除**<br>([`run_experiments.py:400-420`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py#L400-L420)) | 複数 upstream が存在し祖先が特定できない派生モデルを安易に先頭の親へ丸め込まず、`UNMATCHED_ANCESTOR` として PR 計算から明示的に除外。 |
| **HC3 通信増幅検定の再設計**<br>([`statistical_tests.py:790-815`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L790-L815)) | 単なる「命令有無の差」を検定していた従来のロジックを改め、**「通信ONでの共同命令追従率 > 通信OFFでの共同命令追従率（Communication Amplification）」** を直接対立仮説とする検定 `HC3_Communication_Amplification` を新設。Mock実験において通信増幅が 0.000 の場合は正しく非有意（帰無仮説採択）となり、現象と統計検定が整合。 |

---

## 3. 実行確認用コマンド

以下のコマンドをターミナルで実行して動作とテスト結果をご確認いただけます：

```bash
# 1. ユニットテストの実行（85件全件パスの確認）
PYTHONPATH=. pytest -v lineage_collusion/tests/test_all_modules.py

# 2. Command Channel & 内部回路解析パイプラインの Dry-Run 実行
python3 run_experiments.py --mode cc --dry_run --output_dir results
```
