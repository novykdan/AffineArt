import os
import pytest
from pylint.lint import Run
from pylint.reporters import CollectingReporter

# @generated [partially] Gemini 3 Pro 28-12-2025
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
py_files = []
for root, dirs, files in os.walk(src_dir):
    if any(
        x in root for x in ["venv", "__pycache__", ".pytest_cache", ".git", "tests"]
    ):
        continue
    for f in files:
        if f == "__init__.py":
            continue
        if f.endswith(".py"):
            py_files.append(os.path.join(root, f))

file_ids = [os.path.relpath(f, src_dir) for f in py_files]


@pytest.mark.parametrize("py_file", py_files, ids=file_ids)
def test_codestyle(py_file):
    rep = CollectingReporter()

    disabled_rules = [
        "C0114",  # Missing module docstring
        "C0116",  # Missing function docstring
        "C0103",  # Invalid constant name
        "C0301",  # Line too long
        "E0401",  # Import errors
        "E1101",  # Module 'cv2' has no member
        "W0718",  # Broad exception caught
        "R0903",  # Too few public methods
        "W0621",  # Redefining name from outer scope
    ]

    if "app.py" in py_file:
        disabled_rules.extend(
            [
                "R0913",  # Too many arguments
                "R0914",  # Too many locals
                "R0915",  # Too many statements
                "R0912",  # Too many branches
                "W0106",  # Expression not assigned (st.write)
                "C0413",  # Import should be at top
            ]
        )

    disable_str = ",".join(disabled_rules)

    cmd = [f"--disable={disable_str}", "--extension-pkg-allow-list=cv2", "-sn", py_file]

    r = Run(cmd, reporter=rep, exit=False)
    score = r.linter.stats.global_note

    print(f"\n[Linter] {os.path.basename(py_file)} score: {score:.2f}/10")

    if score < 9.0:
        print("\nSuggestions to improve:")
        for m in rep.messages:
            print(f"Line {m.line}: {m.msg} ({m.symbol})")

    assert score >= 9.0, f"{py_file} code quality is too low ({score:.2f})"
