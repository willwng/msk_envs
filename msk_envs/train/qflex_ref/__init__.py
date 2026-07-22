"""
Importing this package sets the XLA GPU-memory env vars so that,
when JAX is imported, it shares the GPU with torch instead of
preallocating most of VRAM.
"""

import os

# Must be set before JAX is first imported anywhere in the process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
