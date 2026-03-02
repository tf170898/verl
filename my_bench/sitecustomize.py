from __future__ import annotations

import os


def _apply_triton_constexpr_shim() -> None:
    # vLLM 0.12 may import kernels expecting tl.constexpr_function, while
    # some Triton builds only provide tl.constexpr.
    if os.environ.get("VERL_BENCH_TRITON_CONSTEXPR_SHIM", "1") != "1":
        return
    try:
        import triton.language as tl
    except Exception:
        return

    if not hasattr(tl, "constexpr_function") and hasattr(tl, "constexpr"):
        try:
            tl.constexpr_function = tl.constexpr
        except Exception:
            return


_apply_triton_constexpr_shim()

