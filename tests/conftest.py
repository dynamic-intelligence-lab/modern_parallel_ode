"""Release XLA's compiled executables between test modules.

Every compiled executable (one per jitted call, and one per primitive and shape in eager mode) keeps its JIT code
mapped for the life of the process, and the mappings count against the kernel's per-process limit
(vm.max_map_count, 65530 by default on Linux). The seven LASER test modules (test_laser*.py) together compile enough executables to
reach that limit, at which point a failed mmap inside XLA's JIT segfaults the process in whatever test compiles next
(measured: 65465 mappings at the crash, with 21172 executable code mappings). Clearing JAX's caches at the end of
each module releases the executables (measured: 5716 mappings to 1215) at the cost of recompiling whatever the next
module shares with it. The fixture is directory-wide: it runs after every module under tests/ (the implicit and
training modules included, not only the LASER ones), so each module recompiles what it would have shared with
the previous one; nothing else about those modules changes.
"""
import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_executables():
    yield
    jax.clear_caches()
    gc.collect()
