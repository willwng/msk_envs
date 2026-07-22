"""Compatibility shim so the pinned-to-JAX-0.4.27 reference ``relax`` code runs
on the newer JAX (0.10.x)
"""

import jax
import jax.extend.core as _extend_core

# Symbols relax.utils.persistence accesses as ``jax.core.<name>`` that moved to
# ``jax.extend.core`` in modern JAX.
_MOVED_TO_EXTEND_CORE = ("ClosedJaxpr", "Jaxpr", "Primitive")


def _install() -> None:
    for name in _MOVED_TO_EXTEND_CORE:
        # Use the module __dict__ directly: a normal ``hasattr`` would trigger
        # JAX's deprecation shim and report False even when we can restore it.
        if name in jax.core.__dict__:
            continue
        replacement = getattr(_extend_core, name, None)
        if replacement is not None:
            setattr(jax.core, name, replacement)


def _install_named_shape() -> None:
    """Restore the ``named_shape`` attribute relax.utils.persistence asserts on """
    sds = jax.ShapeDtypeStruct
    if not hasattr(sds, "named_shape"):
        sds.named_shape = property(lambda self: {})


_install()
_install_named_shape()
