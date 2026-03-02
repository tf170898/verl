"""Process-start compatibility shims for unstable runtime stacks.

This module is auto-imported by Python's site initialization if present in
PYTHONPATH. We use it to patch Triton API drift before vLLM imports.
"""

from __future__ import annotations

try:
    import triton.language as tl

    # Some triton_kernels packages still expect this symbol.
    if not hasattr(tl, "constexpr_function") and hasattr(tl, "constexpr"):
        tl.constexpr_function = tl.constexpr
except Exception:
    # Keep startup resilient if triton is absent or import fails.
    pass
