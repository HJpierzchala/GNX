#!/usr/bin/env python3
"""Build the GNX tutorial manual from Jupyter notebooks.

Pipeline:
  1. optionally execute notebook copies in an isolated build runtime,
  2. export notebooks to Markdown,
  3. combine Markdown chapters,
  4. run Pandoc with a shared bibliography and LaTeX title page.

The script never writes execution outputs back to source notebooks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import nbformat
import yaml
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TUTORIAL_DIR / "manual_config.yaml"
DEFAULT_BUILD_DIR = TUTORIAL_DIR / "_build"
KERNEL_NAME = "gnx-manual-python"
MAX_TEXT_OUTPUT_CHARS = 4000
MAX_TEXT_OUTPUT_LINES = 80
MAX_PDF_TABLE_COLUMNS = 8
MAX_PDF_TABLE_CELL_CHARS = 34


@dataclass(frozen=True)
class Chapter:
    id: str
    title: str
    path: Path
    heavy: bool = False
    execute: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the GNX tutorial manual PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    execute_group = parser.add_mutually_exclusive_group()
    execute_group.add_argument("--execute", action="store_true", help="Execute notebooks before export.")
    execute_group.add_argument("--no-execute", action="store_true", help="Use existing notebook outputs.")
    parser.add_argument("--clean", action="store_true", help="Remove tutorial/_build before building.")
    parser.add_argument("--output", default="GNX_manual.pdf", help="Final PDF filename or path.")
    parser.add_argument(
        "--chapters",
        nargs="+",
        help="Chapter ids or notebook paths to include, in config order.",
    )
    parser.add_argument("--skip-heavy", action="store_true", help="Skip chapters marked heavy in config.")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the isolated runtime repo in _build/runtime_repo after a successful build.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print more detailed progress.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Manual config YAML path.")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), help="Build directory.")
    parser.add_argument("--timeout", type=int, default=900, help="Notebook execution timeout per cell in seconds.")
    parser.add_argument("--latex-engine", help="Override LaTeX engine from config.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used by the temporary Jupyter kernel.",
    )
    return parser.parse_args()


def log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manual config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "manual" not in data:
        raise ValueError(f"Config must contain a top-level 'manual' key: {path}")
    return data["manual"]


def read_chapters(config: dict[str, Any]) -> list[Chapter]:
    chapters = []
    for item in config.get("chapters", []):
        rel = Path(item["path"])
        chapters.append(
            Chapter(
                id=str(item["id"]),
                title=str(item.get("title", item["id"])),
                path=(TUTORIAL_DIR / rel).resolve(),
                heavy=bool(item.get("heavy", False)),
                execute=bool(item.get("execute", True)),
            )
        )
    if not chapters:
        raise ValueError("No chapters configured.")
    for chapter in chapters:
        if not chapter.path.exists():
            raise FileNotFoundError(f"Configured chapter does not exist: {chapter.path}")
    return chapters


def select_chapters(chapters: list[Chapter], requested: list[str] | None, skip_heavy: bool) -> list[Chapter]:
    selected = chapters
    if requested:
        wanted = set(requested)

        def matches(chapter: Chapter) -> bool:
            rel = str(chapter.path.relative_to(TUTORIAL_DIR))
            return chapter.id in wanted or rel in wanted or str(chapter.path) in wanted

        selected = [chapter for chapter in selected if matches(chapter)]
        missing = wanted - {
            token
            for chapter in selected
            for token in (chapter.id, str(chapter.path.relative_to(TUTORIAL_DIR)), str(chapter.path))
        }
        if missing:
            raise ValueError(f"Unknown chapter id/path in --chapters: {sorted(missing)}")
    if skip_heavy:
        selected = [chapter for chapter in selected if not chapter.heavy]
    if not selected:
        raise ValueError("No chapters selected after filters.")
    return selected


def ensure_tools(latex_engine: str) -> None:
    missing = []
    for tool in ["pandoc", latex_engine]:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        raise RuntimeError(
            "Missing required system tool(s): "
            + ", ".join(missing)
            + ". Install Pandoc and a TeX distribution, then rerun the build."
        )


def clean_build_dir(build_dir: Path) -> None:
    if build_dir.exists():
        shutil.rmtree(build_dir)


def create_dirs(build_dir: Path) -> dict[str, Path]:
    dirs = {
        "build": build_dir,
        "executed": build_dir / "executed",
        "chapters": build_dir / "chapters",
        "logs": build_dir / "logs",
        "jupyter": build_dir / "jupyter",
        "runtime": build_dir / "runtime_repo",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_bib_keys(bib_path: Path) -> set[str]:
    text = bib_path.read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def parse_notebook_citations(notebook_path: Path) -> set[str]:
    nb = nbformat.read(notebook_path, as_version=4)
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    return set(re.findall(r"(?<![\w.])@([A-Za-z0-9_:\-.]+)", markdown))


def check_citations(chapters: list[Chapter], bib_path: Path) -> dict[str, list[str]]:
    keys = parse_bib_keys(bib_path)
    missing: dict[str, list[str]] = {}
    for chapter in chapters:
        citations = parse_notebook_citations(chapter.path)
        not_found = sorted(citations - keys)
        if not_found:
            missing[chapter.id] = not_found
    if missing:
        details = "\n".join(f"- {chapter}: {keys}" for chapter, keys in missing.items())
        raise RuntimeError(f"Citation key(s) not found in bibliography:\n{details}")
    return {chapter.id: sorted(parse_notebook_citations(chapter.path)) for chapter in chapters}


def prepare_jupyter_kernel(build_dir: Path, python_executable: Path) -> None:
    kernel_dir = build_dir / "jupyter" / "kernels" / KERNEL_NAME
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_json = {
        "argv": [str(python_executable), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"GNX Manual ({python_executable.name})",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(kernel_json, indent=2), encoding="utf-8")
    jupyter_path = str(build_dir / "jupyter")
    os.environ["JUPYTER_PATH"] = jupyter_path + os.pathsep + os.environ.get("JUPYTER_PATH", "")


def copy_tutorial_runtime(runtime_repo: Path) -> None:
    if runtime_repo.exists():
        shutil.rmtree(runtime_repo)
    runtime_repo.mkdir(parents=True)

    for name in ["gnx_py", "sample_data", "examples"]:
        src = REPO_ROOT / name
        if src.exists():
            (runtime_repo / name).symlink_to(src, target_is_directory=src.is_dir())

    def ignore(dir_path: str, names: list[str]) -> set[str]:
        ignored = {"_build", "__pycache__", ".ipynb_checkpoints", "output", "figures"}
        if Path(dir_path).name == "ionosphere":
            ignored.add("network")
        return ignored.intersection(names)

    shutil.copytree(TUTORIAL_DIR, runtime_repo / "tutorial", ignore=ignore)

    src_network = TUTORIAL_DIR / "ionosphere" / "network"
    dst_network = runtime_repo / "tutorial" / "ionosphere" / "network"
    if src_network.exists():
        dst_network.symlink_to(src_network, target_is_directory=True)


def execution_source(chapter: Chapter, runtime_repo: Path) -> Path:
    rel = chapter.path.relative_to(TUTORIAL_DIR)
    return runtime_repo / "tutorial" / rel


def execute_notebook(
    source_path: Path,
    output_path: Path,
    cwd: Path,
    timeout: int,
    logs_dir: Path,
) -> None:
    nb = nbformat.read(source_path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name=KERNEL_NAME, allow_errors=False)
    resources = {"metadata": {"path": str(cwd)}}
    try:
        ep.preprocess(nb, resources=resources)
    except CellExecutionError as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, output_path)
        log_path = logs_dir / f"{output_path.stem}.error.log"
        log_path.write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"Notebook execution failed for {source_path}. See {log_path}") from exc
    nbformat.write(nb, output_path)


def copy_without_execution(source_path: Path, output_path: Path) -> None:
    nb = nbformat.read(source_path, as_version=4)
    nbformat.write(nb, output_path)


class HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and not self.in_table:
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self._row = []
        elif self.in_table and self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and self.in_row and self.in_cell and tag in {"th", "td"}:
            text = " ".join("".join(self._cell).split())
            self._row.append(text)
            self.in_cell = False
            self._cell = []
        elif self.in_table and self.in_row and tag == "tr":
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self.in_row = False
            self._row = []
        elif self.in_table and tag == "table":
            self.in_table = False


def parse_html_table(html: str) -> list[list[str]]:
    parser = HTMLTableParser()
    parser.feed(html)
    return parser.rows


def escape_latex(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def truncate_cell(text: str, max_chars: int = MAX_PDF_TABLE_CELL_CHARS) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def latex_table_header(text: str) -> str:
    text = truncate_cell(text, max_chars=42)
    if len(text) <= 12:
        return escape_latex(text)
    parts = [part for part in re.split(r"([_/\s-]+)", text) if part]
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = current + part
        if current and len(candidate) > 12:
            lines.append(current.rstrip("_/ -"))
            current = part.lstrip("_/ -")
        else:
            current = candidate
    if current:
        lines.append(current.rstrip("_/ -"))
    if len(lines) <= 1:
        return escape_latex(text)
    return r"\makecell[c]{" + r"\\".join(escape_latex(line) for line in lines[:4]) + "}"


def normalize_table_rows(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    if not rows:
        return rows, 0
    max_cols = max(len(row) for row in rows)
    padded = [row + [""] * (max_cols - len(row)) for row in rows]

    omitted = 0
    if max_cols > MAX_PDF_TABLE_COLUMNS:
        header = padded[0]
        keep = list(range(min(MAX_PDF_TABLE_COLUMNS, max_cols)))
        lower_header = [h.strip().lower() for h in header]
        # Wide positioning tables are easier to read when raw ECEF columns are
        # omitted and ENU/error columns are kept for the PDF preview.
        if {"de", "dn", "du"}.issubset(lower_header):
            useful = [
                idx
                for idx, name in enumerate(lower_header)
                if name
                in {
                    "",
                    "time",
                    "de",
                    "dn",
                    "du",
                    "rms_v",
                    "iters",
                    "n_gps",
                    "n_sats_g",
                    "n_sats_e",
                    "n_sats_c",
                }
            ]
            if len(useful) >= 4:
                keep = useful[:MAX_PDF_TABLE_COLUMNS]
        kept = []
        for idx in keep:
            if idx not in kept and idx < max_cols:
                kept.append(idx)
        omitted = max_cols - len(kept)
        padded = [[row[idx] for idx in kept] for row in padded]
    return padded, omitted


def latex_for_table_rows(rows: list[list[str]]) -> str | None:
    if not rows:
        return None
    padded, omitted_cols = normalize_table_rows(rows)
    max_cols = max(len(row) for row in padded)
    if max_cols == 0:
        return None

    colspec = "l" * max_cols
    table_lines = [rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule"]
    for index, row in enumerate(padded):
        cells = []
        for cell in row:
            cells.append(latex_table_header(cell) if index == 0 else escape_latex(truncate_cell(cell)))
        table_lines.append(" & ".join(cells) + r" \\")
        if index == 0 and len(padded) > 1:
            table_lines.append(r"\midrule")
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    note = ""
    if omitted_cols:
        note = "\n" + rf"\par\scriptsize PDF preview: {omitted_cols} column(s) omitted from this wide table."

    return "\n".join(
        [
            r"\begin{center}",
            r"\begingroup",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\begin{adjustbox}{max width=\linewidth,max totalheight=.42\textheight}",
            "\n".join(table_lines),
            r"\end{adjustbox}",
            note,
            r"\endgroup",
            r"\end{center}",
        ]
    )


def latex_for_dataframe_html(html: str) -> str | None:
    if "<table" not in html.lower():
        return None
    return latex_for_table_rows(parse_html_table(html))


def parse_markdown_table(markdown: str) -> list[list[str]] | None:
    lines = [line.strip() for line in markdown.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return None
    separator = table_lines[1].strip("|").strip()
    if not separator or any(set(part.strip()) - {":", "-"} for part in separator.split("|")):
        return None
    rows = []
    for line in [table_lines[0], *table_lines[2:]]:
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def latex_for_markdown_table(markdown: str) -> str | None:
    rows = parse_markdown_table(markdown)
    if rows is None:
        return None
    return latex_for_table_rows(rows)


def latex_for_xarray_text(text: str) -> str | None:
    if not text.lstrip().startswith("<xarray.Dataset>"):
        return None
    rows = [["section", "summary"]]
    first = text.splitlines()[0].replace("<", "").replace(">", "")
    rows.append(["dataset", first])
    for line in text.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Dimensions:"):
            rows.append(["dimensions", stripped.split(":", 1)[1].strip()])
        elif stripped.startswith("* "):
            parts = stripped.split(None, 4)
            if len(parts) >= 4:
                rows.append([f"coord {parts[1]}", " ".join(parts[2:])])
        elif re.match(r"^[A-Za-z_]\w*\s+\(", stripped):
            rows.append(["variable", stripped])
    return latex_for_table_rows(rows)


def normalize_table_outputs(nb: nbformat.NotebookNode) -> int:
    converted = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            text = data.get("text/plain")
            if isinstance(text, list):
                text = "".join(text)
            html = data.get("text/html")
            if isinstance(html, list):
                html = "".join(html)

            if isinstance(text, str) and "xarray.Dataset" in text:
                latex = latex_for_xarray_text(text)
                if latex is not None:
                    for key in ("text/html", "text/plain", "text/markdown"):
                        data.pop(key, None)
                    data["text/latex"] = latex
                    converted += 1
                    continue

            html = data.get("text/html")
            if isinstance(html, list):
                html = "".join(html)
            if isinstance(html, str):
                latex = latex_for_dataframe_html(html)
                if latex is not None:
                    for key in ("text/html", "text/plain", "text/markdown"):
                        data.pop(key, None)
                    data["text/latex"] = latex
                    converted += 1
                    continue

            markdown = data.get("text/markdown")
            if isinstance(markdown, list):
                markdown = "".join(markdown)
            if isinstance(markdown, str):
                latex = latex_for_markdown_table(markdown)
                if latex is not None:
                    for key in ("text/html", "text/plain", "text/markdown"):
                        data.pop(key, None)
                    data["text/latex"] = latex
                    converted += 1
    return converted


def compact_long_text_outputs(nb: nbformat.NotebookNode) -> int:
    compacted = 0
    rich_keys = {"text/latex", "text/markdown", "image/png", "image/jpeg", "image/svg+xml", "application/pdf"}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            if rich_keys.intersection(data):
                continue
            text = data.get("text/plain")
            if isinstance(text, list):
                text = "".join(text)
            if not isinstance(text, str):
                continue
            lines = text.splitlines()
            if len(text) <= MAX_TEXT_OUTPUT_CHARS and len(lines) <= MAX_TEXT_OUTPUT_LINES:
                continue
            preview_lines = lines[: min(30, len(lines))]
            omitted_lines = max(len(lines) - len(preview_lines), 0)
            preview = "\n".join(preview_lines)
            data["text/plain"] = (
                preview
                + "\n\n"
                + f"[Output shortened for the PDF manual: {omitted_lines} more line(s), "
                + f"{len(text)} characters total. Run the notebook to inspect the full object representation.]"
            )
            compacted += 1
    return compacted


def strip_yaml_frontmatter(markdown: str) -> str:
    if markdown.startswith("---\n"):
        end = markdown.find("\n---", 4)
        if end != -1:
            after = markdown.find("\n", end + 4)
            if after != -1:
                return markdown[after + 1 :].lstrip()
    return markdown


def strip_local_table_of_contents(markdown: str) -> str:
    """Remove notebook-local TOC sections from the combined manual export."""
    lines = markdown.splitlines()
    output: list[str] = []
    skip_level: int | None = None
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for line in lines:
        match = heading_re.match(line)
        if match:
            level = len(match.group(1))
            title = normalize_heading_text(match.group(2))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if title in {"table of contents", "contents"}:
                skip_level = level
                continue
        if skip_level is not None:
            continue
        output.append(line)

    return "\n".join(output)


def normalize_heading_text(text: str) -> str:
    text = re.sub(r"\s*\{#[^}]+\}\s*$", "", text)
    text = re.sub(r"[*_`~\[\]()]", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def make_manual_chapter_markdown(markdown: str, chapter: Chapter) -> str:
    """Make one PDF chapter from one notebook.

    Notebook authors often use top-level Markdown headings inside a notebook.
    Pandoc maps those headings to LaTeX chapters when top-level-division is
    `chapter`, so we prepend the configured chapter title and demote all
    notebook headings by one level for the manual export.
    """
    chapter_title_norm = normalize_heading_text(chapter.title)
    lines = markdown.splitlines()
    normalized: list[str] = []
    skipped_duplicate_title = False
    heading_re = re.compile(r"^(#{1,6})(\s+.+)$")

    for line in lines:
        match = heading_re.match(line)
        if not match:
            normalized.append(line)
            continue

        hashes, rest = match.groups()
        heading_text = rest.strip()
        if (
            not skipped_duplicate_title
            and len(hashes) == 1
            and normalize_heading_text(heading_text) == chapter_title_norm
        ):
            skipped_duplicate_title = True
            continue

        skipped_duplicate_title = True
        new_level = min(len(hashes) + 1, 6)
        normalized.append(f"{'#' * new_level}{rest}")

    body = "\n".join(normalized).lstrip()
    return f"# {chapter.title}\n\n{body}\n"


FIGURE_CAPTIONS: dict[str, dict[str, str]] = {
    "introduction": {
        "output_44_1.png": "SPP position-error time series for BRUX using GPS broadcast navigation.",
    },
    "ppp": {
        "output_27_0.png": "GPS phase post-fit residuals plotted against satellite elevation.",
        "output_28_0.png": "Galileo phase post-fit residuals plotted against satellite elevation.",
        "output_36_0.png": "Combined PPP positioning errors and estimated parameters for the short tutorial window.",
        "output_47_0.png": "Uncombined PPP positioning errors and estimated parameters for the short tutorial window.",
        "output_52_0.png": "Uncombined PPP phase residual diagnostics for the selected GPS observations.",
    },
    "orbits": {
        "output_18_0.png": "GPS broadcast-versus-precise SIS components over the tutorial interval.",
        "output_21_0.png": "Per-satellite GPS SIS statistics used to identify satellite-specific behaviour.",
        "output_40_0.png": "Per-satellite statistics for the SP3-versus-SP3 comparison example.",
    },
    "ionosphere_introduction": {
        "output_25_1.png": "Number of usable observations per satellite arc after preprocessing.",
        "output_27_0.png": "Selected STEC arc showing code, phase, levelled and smoothed TEC estimates.",
        "output_29_0.png": "Measured STEC links and a rough zenith VTEC estimate above the station.",
        "output_33_0.png": "Measured STEC compared with ionospheric model estimates.",
        "output_33_1.png": "Model-comparison residuals for the selected STEC processing example.",
    },
    "calibration": {
        "output_19_1.png": "Effect of receiver DCB choices on the absolute STEC level for one satellite arc.",
    },
    "activity_indexes": {
        "output_7_0.png": "Example STEC time variation used to introduce temporal ionospheric activity indices.",
        "output_12_0.png": "Two-point geometry used to build intuition for spatial VTEC gradients.",
        "output_31_0.png": "ROTI summary statistics over the selected regional network window.",
        "output_33_0.png": "ROTI snapshot rendered from the generated tutorial figure file.",
        "output_40_0.png": "SIDX time series after removing invalid or missing epochs.",
        "output_46_0.png": "GIX family diagnostics for several spatial-gradient summaries.",
        "output_50_0.png": "Spatial GIX example for one selected epoch.",
    },
    "kriging": {
        "output_19_0.png": "Regional VTEC map estimated with the GNX-py kriging workflow.",
        "output_19_1.png": "Formal kriging variance for the same VTEC map epoch.",
        "output_27_0.png": "Absolute VTEC difference between the kriging map and GIM at a common epoch.",
        "output_27_1.png": "Absolute VTEC difference between the NTCM grid and GIM at a common epoch.",
        "output_29_0.png": "Mean absolute VTEC difference against GIM for kriging and NTCM maps.",
    },
}


def caption_for_image(chapter: Chapter, image_path: str) -> str:
    filename = Path(image_path).name
    configured = FIGURE_CAPTIONS.get(chapter.id, {}).get(filename)
    if configured:
        return configured
    title = chapter.title.rstrip(".")
    return f"{title} tutorial figure generated by the notebook."


def latex_for_image(image_path: str, caption: str) -> str:
    latex_path = image_path if Path(image_path).is_absolute() else f"chapters/{image_path}"
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=.92\linewidth,height=.56\textheight,keepaspectratio]{{{latex_path}}}",
            rf"\caption{{{escape_latex(caption)}}}",
            r"\end{figure}",
        ]
    )


def caption_markdown_images(markdown: str, chapter: Chapter) -> str:
    image_re = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$")
    lines = []
    for line in markdown.splitlines():
        match = image_re.match(line)
        if not match:
            lines.append(line)
            continue
        image_path = match.group(1)
        lines.append(latex_for_image(image_path, caption_for_image(chapter, image_path)))
    return "\n".join(lines)


def export_markdown(notebook_path: Path, chapter: Chapter, chapters_dir: Path, index: int) -> Path:
    nb = nbformat.read(notebook_path, as_version=4)
    converted_tables = normalize_table_outputs(nb)
    compacted_text = compact_long_text_outputs(nb)
    exporter = MarkdownExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.exclude_output_stdin = True
    resources = {
        "metadata": {"path": str(notebook_path.parent)},
        "output_files_dir": f"{index:02d}_{chapter.id}_files",
    }
    body, resources = exporter.from_notebook_node(nb, resources=resources)
    body = strip_yaml_frontmatter(body)
    body = strip_local_table_of_contents(body)
    body = make_manual_chapter_markdown(body, chapter)
    body = caption_markdown_images(body, chapter)
    md_path = chapters_dir / f"{index:02d}_{chapter.id}.md"
    md_path.write_text(body, encoding="utf-8")
    if converted_tables or compacted_text:
        (chapters_dir / f"{index:02d}_{chapter.id}.tables.log").write_text(
            "\n".join(
                [
                    f"Converted pandas/HTML tables to LaTeX tables: {converted_tables}",
                    f"Compacted long text/plain outputs: {compacted_text}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    for rel_name, data in resources.get("outputs", {}).items():
        out_path = chapters_dir / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    return md_path


def rewrite_includegraphics_paths(tex: str, base_dir: Path) -> str:
    pattern = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")

    def repl(match: re.Match[str]) -> str:
        prefix, raw_path, suffix = match.groups()
        path = Path(raw_path)
        if path.exists():
            replacement = path.resolve()
        else:
            candidate = base_dir / raw_path
            if candidate.exists():
                replacement = candidate.resolve()
            else:
                by_name = base_dir / path.name
                replacement = by_name.resolve() if by_name.exists() else path
        return f"{prefix}{replacement.as_posix()}{suffix}"

    return pattern.sub(repl, tex)


def prepare_latex_inputs(config: dict[str, Any], build_dir: Path) -> tuple[Path, Path | None]:
    titlepage = (TUTORIAL_DIR / config.get("titlepage", "titlepage.tex")).resolve()
    if not titlepage.exists():
        raise FileNotFoundError(f"Title page TeX not found: {titlepage}")
    title_text = rewrite_includegraphics_paths(titlepage.read_text(encoding="utf-8"), titlepage.parent)
    prepared_title = build_dir / "titlepage_prepared.tex"
    prepared_title.write_text(title_text, encoding="utf-8")

    preamble_name = config.get("preamble")
    prepared_preamble = None
    if preamble_name:
        preamble = (TUTORIAL_DIR / preamble_name).resolve()
        if not preamble.exists():
            raise FileNotFoundError(f"Preamble TeX not found: {preamble}")
        prepared_preamble = build_dir / "preamble_prepared.tex"
        prepared_preamble.write_text(preamble.read_text(encoding="utf-8"), encoding="utf-8")
    return prepared_title, prepared_preamble


def combine_markdown(
    chapter_markdown: list[Path],
    references_path: Path | None,
    build_dir: Path,
    config: dict[str, Any],
    bib_path: Path,
) -> Path:
    manual_md = build_dir / "manual.md"
    parts = [
        "---\n"
        f"title: {json.dumps(config.get('title', 'GNX-py User Manual'))}\n"
        f"bibliography: {json.dumps(bib_path.as_posix())}\n"
        "link-citations: true\n"
        "reference-section-title: References\n"
        "---\n\n"
    ]
    for idx, md_path in enumerate(chapter_markdown):
        if idx:
            parts.append("\n\\clearpage\n\n")
        parts.append(md_path.read_text(encoding="utf-8"))
        parts.append("\n")
    if references_path and references_path.exists():
        parts.append("\n\\clearpage\n\n")
        parts.append(references_path.read_text(encoding="utf-8"))
        parts.append("\n")
    else:
        parts.append("\n\\clearpage\n\n# References\n\n::: {#refs}\n:::\n")
    manual_md.write_text("".join(parts), encoding="utf-8")
    return manual_md


def run_pandoc(
    manual_md: Path,
    output_pdf: Path,
    build_dir: Path,
    chapters_dir: Path,
    bib_path: Path,
    titlepage: Path,
    preamble: Path | None,
    latex_engine: str,
    config: dict[str, Any],
    logs_dir: Path,
    verbose: bool,
) -> None:
    cmd = [
        "pandoc",
        str(manual_md),
        "--from=markdown+tex_math_dollars+raw_tex+fenced_divs+bracketed_spans",
        "--pdf-engine",
        latex_engine,
        "--citeproc",
        "--bibliography",
        str(bib_path),
        "--include-before-body",
        str(titlepage),
        "--top-level-division=chapter",
        "--resource-path",
        os.pathsep.join([str(chapters_dir), str(build_dir), str(TUTORIAL_DIR), str(REPO_ROOT)]),
        "--metadata",
        "link-citations=true",
        "-V",
        "papersize=a4",
        "-V",
        "geometry:margin=1in",
        "-V",
        "colorlinks=true",
        "--listings",
        "-o",
        str(output_pdf),
    ]
    if preamble is not None:
        cmd.extend(["--include-in-header", str(preamble)])
    if bool(config.get("toc", True)):
        cmd.append("--toc")
    if bool(config.get("number_sections", True)):
        cmd.append("--number-sections")

    log(f"[pandoc] {' '.join(cmd)}", verbose=verbose)
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=build_dir)
    (logs_dir / "pandoc.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (logs_dir / "pandoc.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Pandoc failed with exit code {proc.returncode}. "
            f"See {logs_dir / 'pandoc.stderr.log'}"
        )


def maybe_pdf_pages(pdf_path: Path) -> str:
    if shutil.which("pdfinfo") is None:
        return "unknown (pdfinfo not available)"
    proc = subprocess.run(["pdfinfo", str(pdf_path)], text=True, capture_output=True)
    if proc.returncode != 0:
        return "unknown (pdfinfo failed)"
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def build() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    build_dir = Path(args.build_dir).resolve()
    latex_engine = args.latex_engine or str(config.get("latex_engine", "xelatex"))
    python_executable = Path(args.python).expanduser()
    if not python_executable.is_absolute():
        python_executable = (Path.cwd() / python_executable).absolute()
    execute = bool(args.execute)
    if args.no_execute:
        execute = False

    ensure_tools(latex_engine)
    chapters = select_chapters(read_chapters(config), args.chapters, args.skip_heavy)
    bib_path = (TUTORIAL_DIR / config.get("bibliography", "bibliography.bib")).resolve()
    if not bib_path.exists():
        raise FileNotFoundError(f"Bibliography not found: {bib_path}")
    citation_map = check_citations(chapters, bib_path)

    if args.clean:
        clean_build_dir(build_dir)
    dirs = create_dirs(build_dir)
    os.environ["MPLCONFIGDIR"] = str(build_dir / "mplconfig")
    (build_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(dirs["runtime"]) + os.pathsep + str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

    if execute:
        prepare_jupyter_kernel(build_dir, python_executable)
        copy_tutorial_runtime(dirs["runtime"])

    selected_report = {
        "execute": execute,
        "skip_heavy": bool(args.skip_heavy),
        "python": str(python_executable),
        "latex_engine": latex_engine,
        "chapters": [
            {
                "id": chapter.id,
                "path": str(chapter.path.relative_to(TUTORIAL_DIR)),
                "heavy": chapter.heavy,
                "citations": citation_map.get(chapter.id, []),
            }
            for chapter in chapters
        ],
    }
    (dirs["logs"] / "build_plan.json").write_text(json.dumps(selected_report, indent=2), encoding="utf-8")

    markdown_paths: list[Path] = []
    executed_status: dict[str, str] = {}
    for idx, chapter in enumerate(chapters, 1):
        executed_path = dirs["executed"] / f"{idx:02d}_{chapter.id}.ipynb"
        if execute and chapter.execute:
            src = execution_source(chapter, dirs["runtime"])
            log(f"[execute] {chapter.id}: {src.relative_to(dirs['runtime'])}", verbose=True)
            execute_notebook(src, executed_path, src.parent, args.timeout, dirs["logs"])
            executed_status[chapter.id] = "executed"
        else:
            log(f"[copy] {chapter.id}: no execution", verbose=args.verbose)
            copy_without_execution(chapter.path, executed_path)
            executed_status[chapter.id] = "not-executed"
        md_path = export_markdown(executed_path, chapter, dirs["chapters"], idx)
        markdown_paths.append(md_path)

    titlepage, preamble = prepare_latex_inputs(config, build_dir)
    references_name = config.get("references", "references.md")
    references_path = (TUTORIAL_DIR / references_name).resolve() if references_name else None
    manual_md = combine_markdown(markdown_paths, references_path, build_dir, config, bib_path)

    output_arg = Path(args.output)
    output_pdf = output_arg if output_arg.is_absolute() else build_dir / output_arg
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    run_pandoc(
        manual_md,
        output_pdf,
        build_dir,
        dirs["chapters"],
        bib_path,
        titlepage,
        preamble,
        latex_engine,
        config,
        dirs["logs"],
        args.verbose,
    )

    if not args.keep_intermediate and dirs["runtime"].exists():
        shutil.rmtree(dirs["runtime"])

    report = {
        **selected_report,
        "status": "ok",
        "output_pdf": str(output_pdf),
        "output_size_bytes": output_pdf.stat().st_size if output_pdf.exists() else None,
        "pages": maybe_pdf_pages(output_pdf),
        "execution_status": executed_status,
    }
    (dirs["logs"] / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Built manual: {output_pdf}")
    print(f"Size: {report['output_size_bytes']} bytes")
    print(f"Pages: {report['pages']}")
    print(f"Report: {dirs['logs'] / 'build_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
