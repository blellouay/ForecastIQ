"""Safely apply ForecastIQ branding to every repository notebook."""
from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
LEGACY = re.compile(r"RetSci|ret[ -]?sci|agentic-retail-forecasting", re.IGNORECASE)


def replace_text(value: str) -> str:
    value = re.sub(r"RetSci", "ForecastIQ", value, flags=re.IGNORECASE)
    value = re.sub(r"ret[ -]?sci", "forecastiq", value, flags=re.IGNORECASE)
    value = re.sub(r"agentic-retail-forecasting", "forecastiq", value,
                   flags=re.IGNORECASE)
    return value


def update_metadata(value):
    if isinstance(value, str):
        return replace_text(value)
    if isinstance(value, list):
        return [update_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: update_metadata(item) for key, item in value.items()}
    return value


def main() -> None:
    report = []
    for path in sorted(ROOT.rglob("*.ipynb")):
        if ".venv" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        notebook = nbformat.read(path, as_version=4)
        cleared = 0
        changed_sources = 0
        compile_failures = []
        for index, cell in enumerate(notebook.cells):
            updated = replace_text(cell.source)
            if updated != cell.source:
                cell.source = updated
                changed_sources += 1
            if cell.cell_type == "code":
                legacy_output = any(LEGACY.search(json.dumps(output, default=str))
                                    for output in cell.get("outputs", []))
                if legacy_output:
                    cell.outputs = []
                    cell.execution_count = None
                    cleared += 1
                try:
                    compile(cell.source, f"{path.name}:cell-{index}", "exec")
                except SyntaxError as exc:
                    # IPython magics and shell escapes are valid notebook syntax but not Python.
                    if not any(line.lstrip().startswith(("%", "!"))
                               for line in cell.source.splitlines()):
                        compile_failures.append({"cell": index, "error": str(exc)})
        notebook.metadata = update_metadata(dict(notebook.metadata))
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        report.append({"path": str(path.relative_to(ROOT)),
                       "source_cells_updated": changed_sources,
                       "outputs_cleared": cleared,
                       "compile_failures": compile_failures})
    destination = ROOT / "results" / "notebook_rebrand_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
