"""Headless smoke test of every dashboard page with a stub ``streamlit`` module (Streamlit itself is optional).

It verifies that each page executes end-to-end on the real results/engine without exceptions. It does NOT
test the visual rendering, which requires a real Streamlit session.
"""
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Dummy:
    def __call__(self, *a, **k): return _Dummy()
    def __getattr__(self, n): return _Dummy()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([_Dummy(), _Dummy(), _Dummy(), _Dummy()])


class _St(_Dummy):
    sidebar = None
    def cache_resource(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)
    cache_data = cache_resource
    def columns(self, n, *a, **k): return [_Dummy() for _ in range(n if isinstance(n, int) else len(n))]
    def selectbox(self, label, options, *a, **k): return list(options)[0]
    def slider(self, label, *a, **k): return a[2] if len(a) > 2 else a[0]
    def select_slider(self, label, options, value=None, **k): return value
    def multiselect(self, label, options, default=None, **k): return default or []
    def dataframe(self, *a, **k): return None
    def tabs(self, labels): return [_Dummy() for _ in labels]


def test_all_pages_execute_with_stub_streamlit():
    if not (ROOT / "results" / "metrics" / "soc_results.csv").exists() or not (ROOT / "results" / "metrics" / "aging_summary.csv").exists():
        return
    st = _St(); st.sidebar = st
    mod = types.ModuleType("streamlit"); mod.__getattr__ = lambda n: getattr(st, n)
    sys.modules["streamlit"] = mod
    sys.path[:0] = [str(ROOT / "dashboard"), str(ROOT / "src")]
    try:
        for page in [ROOT / "dashboard" / "app.py"] + sorted((ROOT / "dashboard" / "pages").glob("*.py")):
            runpy.run_path(str(page), run_name="__main__")
    finally:
        sys.modules.pop("streamlit", None)
