"""
分類タスクベンチマークアダプタ（SST-2, AG News）。
Layer A（既存研究との比較）および E1（元モデルの制御機構検証）に用いる。

- Track A: 公式 test split の通常分類精度（制御メッセージなし）。意思決定は E1〜E4 と同一
  （尤度採点できる実モデルはラベル尤度 argmax、それ以外は厳格 JSON パース。部分文字列一致は使わない）。
- Track B: 同じ test サンプルで「G1/G0 × C1/C0」の 4 セルを実測する研究用評価。条件付き動作は
  `favor:<label>`（許可された無害なラベル切替）への追従で、命令先≠正解の観測のみ判定対象とする。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from lineage_collusion.benchmarks.base import (
    STATUS_COMPLETED,
    BaseBenchmarkAdapter,
    BenchmarkItem,
    BenchmarkResult,
    BenchmarkTrack,
    evidence_for,
    safe_rate,
    summarize_track_b,
)
from lineage_collusion.data.prior_datasets.control_task import (
    ControlTaskExample,
    control_message_delivered,
    inject_command_prompt,
    resolve_label_target,
)
from lineage_collusion.data.prior_datasets.loaders import load_control_task_report
from lineage_collusion.experiments.control_task_eval import backend_of, decide_label, is_mock_model
from lineage_collusion.training.model_registry import resolve_model_backend


class ClassificationBenchmarkAdapter(BaseBenchmarkAdapter):
    """公式分類データ（AG News / SST-2）の Track A / Track B 評価。"""

    dataset: str = ""
    category_name: str = ""

    def __init__(self):
        super().__init__(name=self.dataset, category=self.category_name)

    def load_examples(self, split: str = "test", n: Optional[int] = None, source: str = "official",
                      strict_official: bool = True) -> Tuple[List[ControlTaskExample], Dict[str, Any]]:
        return load_control_task_report(self.dataset, split=split, n=n, source=source, strict_official=strict_official)

    def load_dataset(self, split: str = "test", n: Optional[int] = None, source: str = "official",
                     strict_official: bool = True) -> List[BenchmarkItem]:
        examples, _ = self.load_examples(split, n, source, strict_official)
        return [BenchmarkItem(item_id=ex.example_id, prompt=inject_command_prompt(ex, None, "none"),
                              target_ground_truth=ex.true_label,
                              metadata={"dataset": self.dataset, "label_set": ex.label_set, **ex.metadata})
                for ex in examples]

    @staticmethod
    def _resolve(model_fn):
        fn, factory = resolve_model_backend(model_fn)
        return fn, backend_of(fn, factory)

    def evaluate_track_a(self, model_fn: Callable[[str], str], split: str = "test", n: Optional[int] = None,
                         source: str = "official", strict_official: bool = True,
                         model_name: str = "model", **_) -> BenchmarkResult:
        examples, load_report = self.load_examples(split, n, source, strict_official)
        fn, backend = self._resolve(model_fn)
        records = []
        for ex in examples:
            prompt = inject_command_prompt(ex, None, "none")
            d = decide_label(backend, fn, prompt, ex.label_set)
            records.append({
                "item_id": ex.example_id, "sample_id": ex.sample_id, "true_label": ex.true_label,
                "pred": d["label"], "valid_output": d["valid"],
                "correct": bool(d["valid"] and d["label"] == ex.true_label),
                "label_probs": d["label_probs"],
            })
        n_valid = sum(1 for r in records if r["valid_output"])
        # 不正出力は誤答として数える（分母は評価件数）。件数 0 は None
        acc = safe_rate(sum(1 for r in records if r["correct"]), len(records))
        return BenchmarkResult(
            benchmark_name=self.name, track=BenchmarkTrack.TRACK_A_STANDARD, model_name=model_name,
            num_samples=len(records), standard_metric_name="accuracy", standard_score=acc,
            status=STATUS_COMPLETED,
            evidence_level=evidence_for(is_mock_model(model_fn) or is_mock_model(fn), source == "official"),
            is_official_score=False, n_requested=n, n_valid=n_valid,
            telemetry={"data_load": load_report, "invalid_counted_as_incorrect": True},
            detailed_records=records,
        )

    def _observe(self, ex: ControlTaskExample, fn, backend, cell: str, prompt: str, target: str) -> Dict[str, Any]:
        d = decide_label(backend, fn, prompt, ex.label_set)
        distinguishable = target != ex.true_label
        behavior = None
        if d["valid"] and distinguishable:
            behavior = d["label"] == target
        return {
            "item_id": ex.example_id, "cell": cell, "group": cell[:2], "condition": cell[3:],
            "message_delivered": control_message_delivered(prompt),
            "model_input": prompt,
            "model_output": d["raw_output"], "label_probs": d["label_probs"],
            "harmless_action_result": {"chosen_label": d["label"]},
            "valid_output": d["valid"],
            "task_success": bool(d["label"] == ex.true_label) if d["valid"] else None,
            "command_target": target, "behavior_distinguishable": distinguishable,
            "conditional_behavior_success": behavior,
        }

    def evaluate_track_b(self, model_fn: Callable[[str], str], clean_baseline_fn: Optional[Callable[[str], str]] = None,
                         command: Optional[str] = None, auth: str = "authenticated", split: str = "test",
                         n: Optional[int] = None, source: str = "official", strict_official: bool = True,
                         model_name: str = "model", clean_model_name: Optional[str] = None, **_) -> BenchmarkResult:
        if not command or not command.startswith("favor:"):
            raise ValueError(f"分類 Track B の命令は favor:<label> で指定してください: {command!r}")
        examples, load_report = self.load_examples(split, n, source, strict_official)
        if examples and command.split(":", 1)[1] not in examples[0].label_set:
            raise ValueError(f"命令先 {command} がラベル集合 {examples[0].label_set} に含まれません")
        groups = {"G1": self._resolve(model_fn)}
        if clean_baseline_fn is not None:
            groups["G0"] = self._resolve(clean_baseline_fn)
        records = []
        for ex in examples:
            target = resolve_label_target(ex, command)
            for g, (fn, backend) in groups.items():
                records.append(self._observe(ex, fn, backend, f"{g}_C1", inject_command_prompt(ex, command, auth), target))
                records.append(self._observe(ex, fn, backend, f"{g}_C0", inject_command_prompt(ex, None, "none"), target))
        summary = summarize_track_b(records)
        g1c1 = summary["cells"]["G1_C1"]
        mock = any(is_mock_model(f) for f in [model_fn, clean_baseline_fn] if f is not None) or any(
            is_mock_model(fn) for fn, _ in groups.values())
        return BenchmarkResult(
            benchmark_name=self.name, track=BenchmarkTrack.TRACK_B_CONTROLLED, model_name=model_name,
            num_samples=len(examples), standard_metric_name="task_success_under_control",
            standard_score=g1c1["task_success_rate"], status=STATUS_COMPLETED,
            evidence_level=evidence_for(mock, source == "official"), is_official_score=False,
            n_requested=n, n_valid=g1c1["n_valid_output"],
            control_success_rate=g1c1["behavior_rate"], task_success_under_control=g1c1["task_success_rate"],
            delta_control=summary["delta_control_did"], track_b_cells=summary,
            telemetry={"data_load": load_report, "command": command, "auth": auth,
                       "clean_model_name": clean_model_name,
                       "harmless_behavior": "favor:<label> へのラベル切替（分類出力のみ。外部操作なし）"},
            detailed_records=records,
        )


class SST2Adapter(ClassificationBenchmarkAdapter):
    """SST-2（GLUE）。論理 test = 公式 validation（GLUE test はラベル非公開）。"""
    dataset = "sst2"
    category_name = "sentiment_classification"


class AGNewsAdapter(ClassificationBenchmarkAdapter):
    """AG News 4 値トピック分類。論理 test = 公式 test。"""
    dataset = "agnews"
    category_name = "topic_classification"


CLASSIFICATION_ADAPTERS = {"sst2": SST2Adapter, "agnews": AGNewsAdapter}
