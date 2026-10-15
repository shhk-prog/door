"""
ベンチマークの一括実行（E1〜E4 の各ステージから呼ぶ）。

- 対象群の**全モデル**を評価する（先頭 1 モデルで代表させない）。
- 各結果は証拠水準ごとに `<stage_dir>/benchmarks/<evidence_level>/` へ保存し、
  `<stage_dir>/benchmarks/index.json` に (benchmark, track, model) → status / evidence_level / score / path を記録する。
- 例外は握りつぶさず RUN_ERROR として記録する。strict=True（本番）では RUN_ERROR が 1 件でもあれば
  BenchmarkSuiteError を送出して停止する（評価できなかった結果を成功扱いしない）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lineage_collusion.benchmarks.base import (
    COMPLETED_STATUSES,
    STATUS_RUN_ERROR,
    BenchmarkResult,
    BenchmarkTrack,
    EvidenceLevel,
    result_output_path,
)


class BenchmarkSuiteError(RuntimeError):
    pass


@dataclass
class BenchmarkJob:
    adapter: Any
    track: str                                   # "A" or "B"
    model_name: str
    model_fn: Any
    clean_name: Optional[str] = None             # Track B の G0（clean 対照）
    clean_fn: Any = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _error_result(job: BenchmarkJob, err: BaseException) -> BenchmarkResult:
    track = BenchmarkTrack.TRACK_A_STANDARD if job.track == "A" else BenchmarkTrack.TRACK_B_CONTROLLED
    return BenchmarkResult(benchmark_name=job.adapter.name, track=track, model_name=job.model_name, num_samples=0,
                           standard_metric_name="n/a", standard_score=None, status=STATUS_RUN_ERROR,
                           evidence_level=EvidenceLevel.NOT_EVALUATED.value,
                           telemetry={"error": f"{type(err).__name__}: {err}"[:500]})


def run_job(job: BenchmarkJob) -> BenchmarkResult:
    try:
        if job.track == "A":
            return job.adapter.evaluate_track_a(job.model_fn, model_name=job.model_name, **job.kwargs)
        if job.track == "B":
            return job.adapter.evaluate_track_b(job.model_fn, clean_baseline_fn=job.clean_fn,
                                                model_name=job.model_name, clean_model_name=job.clean_name,
                                                **job.kwargs)
        raise ValueError(f"未知の track: {job.track}")
    except Exception as e:  # 記録して呼び出し側の strict 判定に委ねる
        return _error_result(job, e)


def run_benchmark_suite(stage_dir: Path, jobs: List[BenchmarkJob], write_json: Callable[[Path, Any], None],
                        strict: bool = False, release: Optional[Callable[[Any], None]] = None) -> Dict[str, Any]:
    """jobs を順に実行して保存し、index を返す。release はモデル切替時に呼ぶ解放関数（GPU メモリ対策）。"""
    index: Dict[str, Any] = {"results": [], "status_counts": {}, "evidence_counts": {}}
    prev_model = None
    for job in jobs:
        if release is not None and prev_model is not None and job.model_fn is not prev_model:
            release(prev_model)
        prev_model = job.model_fn
        res = run_job(job)
        path = result_output_path(stage_dir, res)
        write_json(path, res.to_dict())
        index["results"].append({
            "benchmark": res.benchmark_name, "track": res.track.value if hasattr(res.track, "value") else res.track,
            "model": res.model_name, "g0_model": job.clean_name, "status": res.status,
            "evidence_level": res.evidence_level, "metric": res.standard_metric_name,
            "score": res.standard_score, "control_success_rate": res.control_success_rate,
            "delta_control": res.delta_control, "n_evaluated": res.num_samples, "n_requested": res.n_requested,
            "is_official_score": res.is_official_score, "path": str(path.relative_to(stage_dir)),
        })
        index["status_counts"][res.status] = index["status_counts"].get(res.status, 0) + 1
        index["evidence_counts"][res.evidence_level] = index["evidence_counts"].get(res.evidence_level, 0) + 1
    if release is not None:
        # 被評価モデルに加え、Track B の G0（clean 対照）として読み込まれたモデルも解放する
        seen = set()
        for fn in [j.model_fn for j in jobs] + [j.clean_fn for j in jobs if j.clean_fn is not None]:
            if id(fn) not in seen:
                seen.add(id(fn))
                release(fn)
    write_json(Path(stage_dir) / "benchmarks" / "index.json", index)
    errors = [r for r in index["results"] if r["status"] == STATUS_RUN_ERROR]
    if strict and errors:
        raise BenchmarkSuiteError(f"ベンチマーク実行エラー {len(errors)} 件: {errors[:3]}")
    index["n_completed"] = sum(1 for r in index["results"] if r["status"] in COMPLETED_STATUSES)
    return index


def result_lookup(index: Dict[str, Any], benchmark: str, track: str, model: str) -> Optional[Dict[str, Any]]:
    for r in index.get("results", []):
        if r["benchmark"] == benchmark and r["track"] == track and r["model"] == model:
            return r
    return None
