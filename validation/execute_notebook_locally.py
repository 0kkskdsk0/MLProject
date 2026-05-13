"""Execute a notebook without nbconvert/nbclient and write outputs in-place.

This is a lightweight executor for local project notebooks that:
- runs code cells in a shared namespace
- captures stdout/stderr
- captures `display(...)` calls as text/plain
- captures matplotlib figures as PNG outputs
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "v4Notebook" / "v4model.ipynb"


def stream_output(name: str, text: str) -> dict:
    return {
        "output_type": "stream",
        "name": name,
        "text": text.splitlines(keepends=True),
    }


def display_output(text: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/plain": text.splitlines(keepends=True) or [text],
        },
        "metadata": {},
    }


def image_output(fig) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": encoded,
            "text/plain": [f"<Figure size {fig.get_size_inches()[0]:.0f}x{fig.get_size_inches()[1]:.0f}>"],
        },
        "metadata": {},
    }


def error_output(exc: BaseException) -> dict:
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "output_type": "error",
        "ename": type(exc).__name__,
        "evalue": str(exc),
        "traceback": [line.rstrip("\n") for line in tb_lines],
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    exec_count = 1
    shared_globals: dict = {"__name__": "__main__"}

    def captured_display(*objects, **kwargs):
        outputs = shared_globals.setdefault("__cell_outputs__", [])
        for obj in objects:
            try:
                import pandas as pd  # local import for optional formatting

                if isinstance(obj, pd.DataFrame):
                    text = obj.to_string(index=False)
                else:
                    text = repr(obj)
            except Exception:
                text = repr(obj)
            outputs.append(display_output(text))

    shared_globals["display"] = captured_display

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        cell["execution_count"] = exec_count
        exec_count += 1
        cell["outputs"] = []
        shared_globals["__cell_outputs__"] = []

        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec("".join(cell.get("source", [])), shared_globals)
        except Exception as exc:
            out_text = stdout.getvalue()
            err_text = stderr.getvalue()
            if out_text:
                cell["outputs"].append(stream_output("stdout", out_text))
            if err_text:
                cell["outputs"].append(stream_output("stderr", err_text))
            cell["outputs"].extend(shared_globals["__cell_outputs__"])
            for num in plt.get_fignums():
                cell["outputs"].append(image_output(plt.figure(num)))
            plt.close("all")
            cell["outputs"].append(error_output(exc))
            NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            raise

        out_text = stdout.getvalue()
        err_text = stderr.getvalue()
        if out_text:
            cell["outputs"].append(stream_output("stdout", out_text))
        if err_text:
            cell["outputs"].append(stream_output("stderr", err_text))
        cell["outputs"].extend(shared_globals["__cell_outputs__"])
        for num in plt.get_fignums():
            cell["outputs"].append(image_output(plt.figure(num)))
        plt.close("all")

    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Executed {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
