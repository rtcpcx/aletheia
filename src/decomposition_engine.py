"""
Aletheia — src/decomposition_engine.py

Exact additive decomposition of a KPI's total change into component
effects. No Taylor approximation is used for division — see Section 17 of
the build plan for the derivation. Verified numerically exact for both
multiply and divide cases (see eval/validate_against_truth.py
`score_decomposition_exactness`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MIN_DENOMINATOR = 1e-6
RESIDUAL_TOLERANCE_FRACTION = 0.01
RESIDUAL_EPSILON = 1e-9


@dataclass
class DecompositionResult:
    decomposition_type: Literal["multiply", "divide", "single"]
    effect_a: float
    effect_b: float
    interaction_effect: float
    residual: float
    total_change: float
    denominator_status: Literal["stable", "unstable"] = "stable"
    is_volatile: bool = False
    narrative_mode: Literal["standard", "complex_interaction"] = "standard"


def decompose_multiply(a0: float, a1: float, b0: float, b1: float) -> DecompositionResult:
    """
    K = A * B

    Exact decomposition:
        dA = A1 - A0
        dB = B1 - B0

        effect_a = dA * B0
        effect_b = dB * A0
        interaction_effect = dA * dB
        residual = 0  (exact by construction)
    """
    d_a = a1 - a0
    d_b = b1 - b0

    effect_a = d_a * b0
    effect_b = d_b * a0
    interaction_effect = d_a * d_b

    k0 = a0 * b0
    k1 = a1 * b1
    total_change = k1 - k0

    residual = total_change - (effect_a + effect_b + interaction_effect)

    result = DecompositionResult(
        decomposition_type="multiply",
        effect_a=effect_a,
        effect_b=effect_b,
        interaction_effect=interaction_effect,
        residual=residual,
        total_change=total_change,
    )
    return assess_decomposition_stability(result)


def decompose_divide(
    a0: float, a1: float, b0: float, b1: float, symmetric: bool = True
) -> DecompositionResult:
    """
    K = A / B

    Exact ordering decomposition (numerator effect first):
        numerator_effect  = (A1 - A0) / B0
        denominator_effect = A1/B1 - A1/B0

        dK = numerator_effect + denominator_effect   (exact)

    If `symmetric` is True, also compute the reverse ordering (change B
    first, then A) and Shapley-average the two orderings for fairness, per
    the Section 17.2 symmetry note:

        reverse numerator_effect   = A0/B1 - A0/B0
        reverse denominator_effect = (A1 - A0) / B1

        effect_a = mean(numerator_effect, reverse_numerator_effect)
        effect_b = mean(denominator_effect, reverse_denominator_effect)

    Guardrail: if |B0| or |B1| is below MIN_DENOMINATOR, the decomposition
    is marked denominator_unstable and no percentage attribution should be
    forced downstream.
    """
    denominator_status = "stable"
    if abs(b0) < MIN_DENOMINATOR or abs(b1) < MIN_DENOMINATOR:
        denominator_status = "unstable"
        result = DecompositionResult(
            decomposition_type="divide",
            effect_a=0.0,
            effect_b=0.0,
            interaction_effect=0.0,
            residual=0.0,
            total_change=(a1 / b1 if abs(b1) >= MIN_DENOMINATOR else 0.0)
            - (a0 / b0 if abs(b0) >= MIN_DENOMINATOR else 0.0),
            denominator_status="unstable",
            is_volatile=True,
            narrative_mode="complex_interaction",
        )
        return result

    k0 = a0 / b0
    k1 = a1 / b1
    total_change = k1 - k0

    forward_numerator = (a1 - a0) / b0
    forward_denominator = a1 / b1 - a1 / b0

    if symmetric:
        reverse_numerator = a0 / b1 - a0 / b0
        reverse_denominator = (a1 - a0) / b1

        effect_a = (forward_numerator + reverse_numerator) / 2.0
        effect_b = (forward_denominator + reverse_denominator) / 2.0
    else:
        effect_a = forward_numerator
        effect_b = forward_denominator

    residual = total_change - (effect_a + effect_b)

    result = DecompositionResult(
        decomposition_type="divide",
        effect_a=effect_a,
        effect_b=effect_b,
        interaction_effect=0.0,
        residual=residual,
        total_change=total_change,
        denominator_status=denominator_status,
    )
    return assess_decomposition_stability(result)


def decompose_single(total_change: float) -> DecompositionResult:
    """
    Single-metric KPI (e.g. stock_availability): the entire change is
    attributed to effect_a; there is no second component to split against.
    """
    return DecompositionResult(
        decomposition_type="single",
        effect_a=total_change,
        effect_b=0.0,
        interaction_effect=0.0,
        residual=0.0,
        total_change=total_change,
    )


def assess_decomposition_stability(result: DecompositionResult) -> DecompositionResult:
    """
    Mark volatile when:
    - denominator unstable
    - numerical residual exceeds tolerance
    - decomposition components are disproportionate because the KPI is
      near zero (guarded implicitly via MIN_DENOMINATOR upstream)

    For exact decompositions, residual should be approximately numerical
    floating-point tolerance. If:

        abs(residual) > max(abs(total_change) * 0.01, epsilon)

    flag is_volatile = True, narrative_mode = complex_interaction.
    """
    if result.denominator_status == "unstable":
        result.is_volatile = True
        result.narrative_mode = "complex_interaction"
        return result

    threshold = max(abs(result.total_change) * RESIDUAL_TOLERANCE_FRACTION, RESIDUAL_EPSILON)
    if abs(result.residual) > threshold:
        result.is_volatile = True
        result.narrative_mode = "complex_interaction"
    else:
        result.is_volatile = False
        result.narrative_mode = "standard"

    return result


def decompose(
    operator: Literal["multiply", "divide", "single"],
    a0: float | None,
    a1: float | None,
    b0: float | None = None,
    b1: float | None = None,
) -> DecompositionResult:
    """Dispatch helper used by src/pipeline.py."""
    if operator == "multiply":
        assert b0 is not None and b1 is not None
        return decompose_multiply(a0, a1, b0, b1)
    if operator == "divide":
        assert b0 is not None and b1 is not None
        return decompose_divide(a0, a1, b0, b1)
    if operator == "single":
        return decompose_single((a1 or 0.0) - (a0 or 0.0))
    raise ValueError(f"Unknown decomposition operator: {operator!r}")
