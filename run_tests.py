"""Run the test suite with pytest if installed, otherwise with a minimal built-in runner."""
import importlib.util
import math
import sys
import traceback
import types

if importlib.util.find_spec("pytest") is None:
    shim = types.ModuleType("pytest")

    class _Approx:
        def __init__(self, expected, rel=None, abs=None):
            self.e, self.rel, self.abs = expected, rel, abs

        def __eq__(self, other):
            tol = max((self.rel if self.rel is not None else 1e-6) * math.fabs(self.e),
                      self.abs if self.abs is not None else 0.0)
            return math.fabs(other - self.e) <= tol

        def __repr__(self):
            return f"approx({self.e}, rel={self.rel}, abs={self.abs})"

    shim.approx = _Approx
    sys.modules["pytest"] = shim
else:
    import pytest
    sys.exit(pytest.main(["-q", "tests"]))

sys.path.insert(0, "tests")
mod = __import__("test_model")
fails = 0
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    try:
        getattr(mod, name)()
        print(f"PASS  {name}")
    except Exception:
        fails += 1
        print(f"FAIL  {name}")
        traceback.print_exc()
print(f"\n{fails} failed")
sys.exit(1 if fails else 0)
