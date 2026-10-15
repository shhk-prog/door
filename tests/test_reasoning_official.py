"""
GSM8K / MBPP の公式データに対する評価プロトコル検証（公式データを取得できない環境では skip）。

- MBPP: 公式 test（500 問）の公式参照解を、本番と同じ経路（公式プロンプト → 生成 → 抽出 → OS 隔離実行
  （コード → test_setup_code → test_list））で評価し、全問 PASS することを確認する（実行プロトコルの正しさ）。
  参照解の末尾で SystemExit(0) により早期終了する解（テスト未実行）は FAIL になることも確認する。
- GSM8K: 公式 test（1319 問）の参照解答（`#### N`）が strict / flexible の両抽出で正解と一致すること。
"""

import pytest

from lineage_collusion.benchmarks.code_sandbox import probe_sandbox
from lineage_collusion.benchmarks.reasoning_adapters import (
    DataUnavailableError,
    GSM8KAdapter,
    MBPPAdapter,
    _load_hf,
)


def _official_rows(adapter, split):
    try:
        return list(_load_hf(adapter.proto, split))
    except DataUnavailableError as e:
        pytest.skip(f"公式データを取得できない環境（依存・ネットワーク不足）: {e}")


@pytest.mark.skipif(not probe_sandbox()["available"], reason="bwrap サンドボックスが使えない環境")
def test_mbpp_official_reference_solutions_all_pass_and_broken_fail():
    ad = MBPPAdapter()
    rows = _official_rows(ad, ad.proto["eval_split"])
    assert len(rows) == 500 and rows[0]["task_id"] == 11 and rows[-1]["task_id"] == 510
    items = ad.load_dataset()
    ref = {f"mbpp_{r['task_id']}": r["code"] for r in rows}
    by_prompt = {it.prompt: ref[it.item_id] for it in items}
    res = ad.evaluate_track_a(lambda p: by_prompt[p] + "\n[DONE]", model_name="official_reference")
    assert res.status == "COMPLETED" and res.num_samples == 500
    assert res.standard_score == 1.0, [r for r in res.detailed_records if not r["passed"]][:5]
    assert res.evidence_level == "official_data_standard_metric"
    # 公式プロンプトは問題文とテスト 3 件を含み、few-shot は prompt split（評価 split と重ならない）
    assert "Your code should pass these tests" in items[0].prompt and items[0].prompt.count("[BEGIN]") == 4

    broken = {p: c + "\n\nraise SystemExit(0)" for p, c in list(by_prompt.items())[:20]}
    res_b = ad.evaluate_track_a(lambda p: broken[p] + "\n[DONE]", n=20, model_name="broken")
    assert res_b.standard_score == 0.0   # 早期終了（exit 0）は合格にならない


def test_gsm8k_official_reference_answers_are_extracted():
    ad = GSM8KAdapter()
    rows = _official_rows(ad, ad.proto["eval_split"])
    assert len(rows) == 1319
    items = ad.load_dataset()
    ref = {it.prompt: r["answer"] for it, r in zip(items, rows)}
    res = ad.evaluate_track_a(lambda p: ref[p], model_name="official_reference")
    assert res.standard_score == 1.0 and res.telemetry["exact_match_flexible"] == 1.0
    assert res.evidence_level == "official_data_standard_metric"
    assert items[0].prompt.count("Question:") == ad.proto["num_fewshot"] + 1
