"""
LineageColludeBench Internal Circuit and Mechanistic Evaluation Package
"""

from lineage_collusion.circuits.circuit_eval_types import (
    LayerActivationRecord,
    LayerDivergenceMetric,
    ActivationAnalysisResult,
    CausalInterventionRecord,
    CausalTracingResult,
    CircuitPersistenceRecord,
    CircuitEvaluationSummary,
)
from lineage_collusion.circuits.mock_circuit_engine import MockCircuitBackend
from lineage_collusion.circuits.real_circuit_backend import RealModelCircuitBackend
from lineage_collusion.circuits.activation_analyzer import analyze_model_activations
from lineage_collusion.circuits.causal_tracing import run_causal_tracing
from lineage_collusion.circuits.circuit_persistence import (
    compute_subspace_alignment,
    evaluate_cross_model_patching,
    evaluate_circuit_persistence,
)

__all__ = [
    "LayerActivationRecord",
    "LayerDivergenceMetric",
    "ActivationAnalysisResult",
    "CausalInterventionRecord",
    "CausalTracingResult",
    "CircuitPersistenceRecord",
    "CircuitEvaluationSummary",
    "MockCircuitBackend",
    "RealModelCircuitBackend",
    "analyze_model_activations",
    "run_causal_tracing",
    "compute_subspace_alignment",
    "evaluate_cross_model_patching",
    "evaluate_circuit_persistence",
]
