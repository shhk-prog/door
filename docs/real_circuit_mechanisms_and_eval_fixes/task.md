# タスクリスト: 実モデル回路測定・内部機構継承整合化・評価系修正

## 状況・目的
- 実行時エラーの解消: `statistical_tests.py` の `Callable` 未インポート修正
- テストスキーマの整合化: `test_all_modules.py` の新規テストを最新の `ProviderAttributes` / `ScenarioWeights` / `ProcurementScenario` スキーマに合致させ、テスト85件全件パスを達成
- 内部回路測定の根幹修正:
  - 実モデルモードで実テンソル・PyTorch HooksからResidual Streamを取得・介入する `RealModelCircuitBackend` の実装と統合（P0）
  - Mock回路エンジンの整合化: 親の制御基底の継承・摂動モデルへの改修により、FT後の回路継承・整列度およびCross-Model Patchingが正しく機能するようにする（P0）
  - 分岐層検出の正常化: CleanモデルでLayer 0と誤判定される問題を解消し、無認証対照・特異的シグナル変化率に基づく高精度検出・未検出判定の実装（P0）
  - 因果介入・プロービングの厳密化: 独立Train/Test分割による線形プロービング精度の算出（P1）
- 実験パイプラインおよび統計検定の修正:
  - 派生モデル評価への上流モデル混入防止の徹底（P1）
  - 親モデル不明時の安易なフォールバック排除（P1）
  - HC3検定の再定義: 単なる命令効果ではなく「エージェント間通信による談合増幅（Comm Amplification）」を検定するよう修正（P1）
- ドキュメント整備と検証

---

## タスク進行状況

- [x] 1. 計画策定とドキュメント作成
  - [x] 新規トピックフォルダ作成 (`docs/real_circuit_mechanisms_and_eval_fixes`)
  - [x] `task.md` 作成
  - [x] `implementation_plan.md` 作成
- [x] 2. P0: 実行時エラーおよびテストスキーマの修正
  - [x] 2.1 `statistical_tests.py`: `typing` から `Callable` をインポート
  - [x] 2.2 `test_all_modules.py`: `test_cross_descendant_tag_no_overwrite` と `test_internal_circuit_analysis_and_causal_tracing` のシナリオデータ作成を `ScenarioGenerator` または完全スキーマに修正
- [x] 3. P0: 内部回路測定・Mock整合化・分岐層検出の改修
  - [x] 3.1 `mock_circuit_engine.py`: 親モデルの `control_direction` を祖先（upstream）基底として共有し、派生モデルはFT摂動を加えた継承基底とするよう修正。Cleanモデルは無相関基底。
  - [x] 3.2 `activation_analyzer.py`:
    - 分岐層検出の刷新: 通常タスク入力との表層差ではなく、無認証命令入力との乖離（認証特異的活性化）および層間変化率（スパイク）で判定
    - Cleanモデルや制御未発現時は `None` (未検出) と判定
    - 線形プロービングを Train/Test 分割（Out-of-Sample）で測定するよう厳密化
  - [x] 3.3 `real_circuit_backend.py`: 実LLM（PyTorch / HuggingFace Transformers）用の `RealModelCircuitBackend` を実装。Forward Hookを用いた各層 Residual Stream 抽出、Activation Patching、Ablation を提供
  - [x] 3.4 `run_experiments.py`: 実モデルモードにおいて `RealModelCircuitBackend` を構築・使用するよう統合
- [x] 4. P1: 実験パイプラインおよび統計検定の修正
  - [x] 4.1 `run_experiments.py`: `post_ckpts` が空の場合の混入防止、親不明チェックポイントのPR除外処理
  - [x] 4.2 `statistical_tests.py`: HC3 検定を「命令有無の差」から「通信有無による共同追従増幅（Comm Amplification）」の検定へ修正
- [x] 5. 検証とまとめ
  - [x] 5.1 ユニットテスト実行（85件全件パス）
  - [x] 5.2 `run_experiments.py --mode cc --dry_run` の実行と出力指標の整合性確認（回路整列度、Cross-Model Patching、HC3）
  - [x] 5.3 `walkthrough.md` の作成
