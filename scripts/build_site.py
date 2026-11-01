"""Build the web version of the course as a static site.

The notebooks stay the single source of truth. This script reads the committed,
already-executed `.ipynb` files plus the READMEs and writes plain HTML to `site/`,
so a new module appears on the site as soon as its directory exists and the root
README links it. Nothing is re-executed here.

    # build
    .venv/bin/python scripts/build_site.py

    # build, then serve on http://127.0.0.1:8000
    .venv/bin/python scripts/build_site.py --serve

Every internal link and anchor is verified at the end of the build, and a broken
one fails it.
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import http.server
import json
import os
import re
import shutil
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
import nbformat
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.util import ClassNotFound

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES_DIR = REPO_ROOT / "modules"
WEB_DIR = REPO_ROOT / "web"
OUTPUT_DIR = REPO_ROOT / "site"
GITHUB_URL = "https://github.com/Filip3Owl/full-stack-data-science"

CALLOUT_RE = re.compile(r"^\[!(NOTE|TIP|WARNING)\]\s*")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Notebook outputs record the absolute path of whoever executed them.
LOCAL_PATH_RE = re.compile(
    r"(?:/Users|/home)/[^/\s'\"]+/(?:[^/\s'\"]+/)*?"
    r"(?:CoursePythonStatistics|full-stack-data-science)"
)
PANDAS_STYLE_RE = re.compile(r"<style scoped>.*?</style>", re.DOTALL)
HTML_SRC_RE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')
EXTERNAL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.IGNORECASE)

FORMATTER = HtmlFormatter(nowrap=True)


class BuildError(RuntimeError):
    """The content cannot be turned into a correct site."""


# ---------------------------------------------------------------------------
# Course model
# ---------------------------------------------------------------------------
@dataclass
class Lesson:
    """One notebook of a module."""

    source: Path
    module_slug: str
    number: str
    title: str
    description: str = ""
    time: str = ""

    @property
    def stem(self) -> str:
        return self.source.stem

    @property
    def path(self) -> str:
        return f"modules/{self.module_slug}/{self.stem}.html"

    @property
    def lesson_id(self) -> str:
        return f"{self.module_slug[:2]}/{self.stem}"

    @property
    def kind(self) -> str:
        if "solution" in self.stem:
            return "solutions"
        return "project" if "project" in self.stem else "lesson"


@dataclass
class Module:
    """One row of the curriculum; `slug` is empty until the module exists."""

    number: str
    title: str
    topics: str
    slug: str = ""
    time: str = ""
    lessons: list[Lesson] = field(default_factory=list)
    has_exercises: bool = False

    @property
    def ready(self) -> bool:
        return bool(self.slug)

    @property
    def path(self) -> str:
        return f"modules/{self.slug}/index.html"

    @property
    def exercises_path(self) -> str:
        return f"modules/{self.slug}/exercises.html"

    def reading_order(self) -> list[tuple[str, str]]:
        """Pages as (path, title), with the exercises before their solutions."""
        pages = [(self.path, f"Module {self.number} overview")]
        solutions = [lesson for lesson in self.lessons if lesson.kind == "solutions"]
        pages += [
            (lesson.path, lesson.title)
            for lesson in self.lessons
            if lesson.kind != "solutions"
        ]
        if self.has_exercises:
            pages.append((self.exercises_path, "Exercises"))
        pages += [(lesson.path, lesson.title) for lesson in solutions]
        return pages


@dataclass
class Part:
    """A named group of modules in the curriculum."""

    numeral: str
    title: str
    modules: list[Module] = field(default_factory=list)


def parse_curriculum(readme: str) -> list[Part]:
    """Read the parts and modules from the root README's curriculum tables."""
    parts: list[Part] = []
    for line in readme.splitlines():
        heading = re.match(r"^### Part ([IVX]+): (.+)$", line)
        if heading:
            parts.append(Part(heading[1], heading[2].strip()))
            continue
        row = re.match(r"^\|\s*\*{0,2}(\d{2})\*{0,2}\s*\|(.+?)\|(.+?)\|.+\|\s*$", line)
        if not row or not parts:
            continue
        name = row[2].strip()
        link = re.match(r"^\[\*{0,2}(.+?)\*{0,2}\]\(modules/([^)/]+)/?\)$", name)
        parts[-1].modules.append(
            Module(
                number=row[1],
                title=link[1] if link else name,
                topics=row[3].strip(),
                slug=link[2] if link else "",
            )
        )
    if not parts:
        raise BuildError("no curriculum tables found in README.md")
    return parts


def load_module(module: Module) -> None:
    """Fill in a ready module's lessons from its directory and README."""
    directory = MODULES_DIR / module.slug
    if not directory.is_dir():
        raise BuildError(f"README links a missing module: {directory}")
    readme = (directory / "README.md").read_text(encoding="utf-8")
    estimate = re.search(r"Estimated time:\s*\*\*(.+?)\*\*", readme)
    module.time = estimate[1] if estimate else ""
    module.has_exercises = (directory / "exercises" / "README.md").exists()

    rows = {
        row[1]: (row[2].strip(), row[3].strip())
        for row in re.finditer(
            r"^\|[^|]*\|\s*\[[^\]]+\]\(notebooks/([^)]+)\.ipynb\)\s*\|(.*?)\|(.*?)\|\s*$",
            readme,
            re.MULTILINE,
        )
    }
    for source in sorted((directory / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        heading = re.match(r"#\s+(.+)", notebook.cells[0].source)
        if not heading:
            raise BuildError(f"{source} does not start with a title heading")
        number, _, title = heading[1].partition(" · ")
        description, time = rows.get(source.stem, ("", ""))
        module.lessons.append(
            Lesson(
                source=source,
                module_slug=module.slug,
                number=number if title else source.stem[:2],
                title=title or number,
                description=re.sub(r"[*`]", "", description),
                time="" if time in {"—", "-"} else time,
            )
        )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
class LinkMap:
    """Translates repository links into site links, recording assets to copy."""

    def __init__(self, modules: list[Module]) -> None:
        self.pages: dict[Path, str] = {REPO_ROOT / "README.md": "index.html"}
        self.assets: set[Path] = set()
        for module in modules:
            directory = MODULES_DIR / module.slug
            self.pages[directory] = module.path
            self.pages[directory / "README.md"] = module.path
            self.pages[directory / "exercises"] = module.exercises_path
            self.pages[directory / "exercises" / "README.md"] = module.exercises_path
            for lesson in module.lessons:
                self.pages[lesson.source] = lesson.path

    def rewrite(self, url: str, source_dir: Path, page: str) -> str:
        """Return `url`, written relative to `source_dir`, as seen from `page`."""
        if not url or EXTERNAL_RE.match(url):
            return url
        target, hash_mark, fragment = url.partition("#")
        resolved = (source_dir / target).resolve()
        suffix = f"{hash_mark}{fragment}"
        if resolved in self.pages:
            return self.relative(self.pages[resolved], page) + suffix
        if not resolved.exists():
            raise BuildError(f"broken link {url!r} (from {source_dir}, page {page})")
        try:
            in_repo = resolved.relative_to(REPO_ROOT)
        except ValueError as err:
            raise BuildError(f"link {url!r} leaves the repository") from err
        if in_repo.parts[0] == "assets":
            self.assets.add(in_repo)
            return self.relative(in_repo.as_posix(), page) + suffix
        return f"{GITHUB_URL}/blob/main/{in_repo.as_posix()}{suffix}"

    @staticmethod
    def relative(target: str, page: str) -> str:
        directory = os.path.dirname(page) or "."
        return os.path.relpath(target, directory).replace(os.sep, "/")


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE).strip()
    return re.sub(r"[\s_]+", "-", slug) or "section"


def inline_text(token: Token) -> str:
    return "".join(
        " " if child.type == "softbreak" else child.content
        for child in token.children or []
        if child.type != "html_inline"
    )


def highlight_code(code: str, language: str = "", _attrs: str = "") -> str:
    """Pygments highlighting; markdown-it wraps the result in <pre><code>."""
    try:
        lexer = get_lexer_by_name(language) if language else None
    except ClassNotFound:
        lexer = None
    if lexer is None:
        return html.escape(code)
    return highlight(code, lexer, FORMATTER)


def apply_callouts(tokens: list[Token]) -> list[Token]:
    """Turn GitHub `> [!NOTE]` blockquotes into titled callouts."""
    result: list[Token] = []
    for index, token in enumerate(tokens):
        result.append(token)
        if token.type != "blockquote_open" or index + 2 >= len(tokens):
            continue
        inline = tokens[index + 2]
        children = inline.children or []
        match = CALLOUT_RE.match(children[0].content) if children else None
        if inline.type != "inline" or not match:
            continue
        children[0].content = children[0].content[match.end() :]
        if not children[0].content:
            del children[0]
            if children and children[0].type == "softbreak":
                del children[0]
        kind = match[1].lower()
        token.attrSet("class", f"callout callout-{kind}")
        title = Token("html_block", "", 0)
        title.content = f'<p class="callout-title">{kind.capitalize()}</p>\n'
        result.append(title)
    return result


def site_core_rule(state) -> None:
    """Rewrite links, give headings ids, and collect the TOC and search text."""
    env = state.env
    state.tokens = apply_callouts(state.tokens)
    links: LinkMap = env["links"]
    rewrite = functools.partial(
        links.rewrite, source_dir=env["source_dir"], page=env["page"]
    )

    def rewrite_html(markup: str) -> str:
        return HTML_SRC_RE.sub(lambda m: m[1] + rewrite(m[2]) + m[3], markup)

    for index, token in enumerate(state.tokens):
        if token.type == "html_block":
            token.content = rewrite_html(token.content)
        elif token.type == "heading_open":
            text = inline_text(state.tokens[index + 1])
            count = env["slugs"]
            base = slugify(text)
            anchor = f"{base}-{count[base]}" if count[base] else base
            count[base] += 1
            token.attrSet("id", anchor)
            level = int(token.tag[1])
            if level == 1:  # the page title is the only real <h1>
                token.attrSet("class", "part-heading")
                token.tag = state.tokens[index + 2].tag = "h2"
            if level <= 3:
                env["toc"].append({"level": max(level, 2), "id": anchor, "text": text})
            if level <= 2:
                env["sections"].append({"title": text, "anchor": anchor, "text": []})
        elif token.type == "inline":
            is_heading = index and state.tokens[index - 1].type == "heading_open"
            if not is_heading:
                env["sections"][-1]["text"].append(inline_text(token))
            for child in token.children or []:
                if child.type == "link_open":
                    child.attrSet("href", rewrite(child.attrGet("href") or ""))
                elif child.type == "image":
                    child.attrSet("src", rewrite(child.attrGet("src") or ""))
                    child.attrSet("loading", "lazy")
                elif child.type == "html_inline":
                    child.content = rewrite_html(child.content)
        elif token.type in {"fence", "code_block"}:
            env["sections"][-1]["text"].append(token.content)


def math_renderer(content: str, config: dict) -> str:
    display = "true" if config.get("display_mode") else "false"
    return f'<span class="tex" data-display="{display}">{html.escape(content)}</span>'


def make_markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True, "highlight": highlight_code})
    md.enable("table").enable("strikethrough")
    md.use(dollarmath_plugin, double_inline=True, renderer=math_renderer)
    md.use(tasklists_plugin, enabled=True)
    md.core.ruler.push("site", site_core_rule)

    def table_open(self, tokens, idx, options, env):
        return '<div class="table-wrap" tabindex="0">' + self.renderToken(
            tokens, idx, options, env
        )

    def table_close(self, tokens, idx, options, env):
        return self.renderToken(tokens, idx, options, env) + "</div>"

    md.add_render_rule("table_open", table_open)
    md.add_render_rule("table_close", table_close)
    return md


MARKDOWN = make_markdown()


def new_env(links: LinkMap, source_dir: Path, page: str, title: str) -> dict:
    """Per-page render state, shared by every markdown cell of a notebook."""
    return {
        "links": links,
        "source_dir": source_dir,
        "page": page,
        "slugs": Counter(),
        "toc": [],
        "sections": [{"title": title, "anchor": "", "text": []}],
    }


def strip_title(markdown: str) -> str:
    """Drop the leading `# Title` block; the page header already shows it."""
    markdown = re.sub(r"^\s*#\s+.*\n?", "", markdown, count=1)
    markdown = re.sub(r"^\s*\*\*Module \d+[^\n]*\*\*\s*\n?", "", markdown, count=1)
    return re.sub(r"^\s*---\s*\n", "", markdown, count=1)


# ---------------------------------------------------------------------------
# Notebooks
# ---------------------------------------------------------------------------
def clean_output_text(text: str) -> str:
    return LOCAL_PATH_RE.sub("~/full-stack-data-science", ANSI_RE.sub("", text))


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


def join_source(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "".join(value)


def render_output(output, figure_dir: Path, lesson: Lesson, figures: list[int]) -> str:
    """Render one cell output as HTML, writing any figure next to the page."""
    kind = output.output_type
    if kind == "stream":
        css = "output-stderr" if output.get("name") == "stderr" else "output-text"
        text = clean_output_text(join_source(output.text))
        return f'<pre class="{css}" tabindex="0">{html.escape(text)}</pre>'
    if kind == "error":
        trace = clean_output_text("\n".join(output.traceback))
        return f'<pre class="output-stderr" tabindex="0">{html.escape(trace)}</pre>'

    data = output.get("data", {})
    if "image/png" in data:
        raw = base64.b64decode(join_source(data["image/png"]))
        figures[0] += 1
        name = f"{lesson.stem}-{figures[0]}.png"
        figure_dir.mkdir(parents=True, exist_ok=True)
        (figure_dir / name).write_bytes(raw)
        width, height = png_size(raw)
        return (
            f'<img class="output-figure" src="figures/{name}" width="{width}" '
            f'height="{height}" loading="lazy" '
            'alt="Figure produced by the code cell above">'
        )
    if "text/html" in data:
        markup = PANDAS_STYLE_RE.sub("", join_source(data["text/html"]))
        markup = clean_output_text(markup).replace(' border="1"', "")
        return f'<div class="output-html table-wrap" tabindex="0">{markup}</div>'
    if "text/plain" in data:
        text = clean_output_text(join_source(data["text/plain"]))
        return f'<pre class="output-text" tabindex="0">{html.escape(text)}</pre>'
    return ""


def render_notebook(
    lesson: Lesson, links: LinkMap, output_dir: Path
) -> tuple[str, dict]:
    """Render a notebook's cells to HTML, returning it with the render state."""
    notebook = nbformat.read(lesson.source, as_version=4)
    env = new_env(links, lesson.source.parent, lesson.path, lesson.title)
    figure_dir = output_dir / "modules" / lesson.module_slug / "figures"
    figures = [0]
    parts: list[str] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            source = strip_title(cell.source) if index == 0 else cell.source
            parts.append(MARKDOWN.render(source, env))
        elif cell.cell_type == "code" and cell.source.strip():
            if not cell.get("execution_count"):
                raise BuildError(
                    f"{lesson.source.name}: a code cell was never executed; "
                    "run scripts/build_notebooks.py first"
                )
            env["sections"][-1]["text"].append(cell.source)
            code = highlight(cell.source, PythonLexer(), FORMATTER)
            outputs = "".join(
                render_output(output, figure_dir, lesson, figures)
                for output in cell.outputs
            )
            output_block = (
                '<div class="cell-output"><span class="cell-label">Output</span>'
                f"{outputs}</div>"
                if outputs
                else ""
            )
            parts.append(
                '<div class="cell"><div class="cell-input">'
                '<button class="copy" type="button">Copy</button>'
                f'<pre class="code" tabindex="0"><code>{code}</code></pre></div>'
                f"{output_block}</div>"
            )
    return "\n".join(parts), env


# ---------------------------------------------------------------------------
# Site assembly
# ---------------------------------------------------------------------------
def readme_sections(readme: str, names: list[str]) -> str:
    """Extract the named `## ` sections of a README, in the order given."""
    chunks = re.split(r"^(?=## )", readme, flags=re.MULTILINE)
    by_name = {chunk.splitlines()[0][3:].strip(): chunk for chunk in chunks[1:]}
    parts = []
    for name in names:
        if name not in by_name:
            raise BuildError(f"README.md has no '## {name}' section")
        parts.append(re.sub(r"\n---\s*$", "\n", by_name[name].rstrip()))
    return "\n\n".join(parts)


def parse_philosophy(readme: str) -> list[dict[str, str]]:
    """Read the rules from the README's Philosophy table."""
    section = readme_sections(readme, ["Philosophy"])
    rules = re.findall(
        r"^\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*$", section, re.MULTILINE
    )
    return [{"name": name, "text": text.replace("*", "")} for name, text in rules]


def search_entries(env: dict, page: str, page_title: str, group: str) -> list[dict]:
    """Turn a page's sections into compact search records."""
    entries = []
    for section in env["sections"]:
        text = re.sub(r"\s+", " ", " ".join(section["text"])).strip()
        if not text and not section["anchor"]:
            continue
        anchor = f"#{section['anchor']}" if section["anchor"] else ""
        entries.append(
            {
                "s": section["title"],
                "p": page_title,
                "g": group,
                "u": page + anchor,
                "x": text[:1500],
            }
        )
    return entries


def verify_links(output_dir: Path) -> list[str]:
    """Check every relative href/src and fragment in the built pages."""
    output_dir = output_dir.resolve()
    pages = {
        path: path.read_text(encoding="utf-8") for path in output_dir.rglob("*.html")
    }
    ids = {
        path: set(re.findall(r'\bid="([^"]+)"', text)) for path, text in pages.items()
    }
    problems = []
    for path, text in pages.items():
        for url in re.findall(r'\b(?:href|src)="([^"]+)"', text):
            url = html.unescape(url)
            if url.startswith("#"):
                target, fragment = path, url[1:]
            elif EXTERNAL_RE.match(url):
                continue
            else:
                relative, _, fragment = url.partition("#")
                target = (path.parent / relative).resolve()
            label = f"{path.relative_to(output_dir)} -> {url}"
            if not target.exists():
                problems.append(f"missing file: {label}")
            elif fragment and target in ids and fragment not in ids[target]:
                problems.append(f"missing anchor: {label}")
    return problems


def build(output_dir: Path = OUTPUT_DIR) -> dict[str, int]:
    """Generate the whole site into `output_dir`, replacing what was there."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    parts = parse_curriculum(readme)
    modules = [module for part in parts for module in part.modules]
    ready = [module for module in modules if module.ready]
    for module in ready:
        load_module(module)
    links = LinkMap(ready)

    resolved_output = output_dir.resolve()
    if resolved_output in {REPO_ROOT, Path.home(), Path("/")}:
        raise BuildError(f"refusing to replace {resolved_output}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copytree(WEB_DIR / "static", output_dir / "static")

    templates = jinja2.Environment(
        loader=jinja2.FileSystemLoader(WEB_DIR / "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    shared = {
        "parts": parts,
        "ready_modules": ready,
        "github_url": GITHUB_URL,
        "total_modules": len(modules),
        "total_lessons": sum(len(module.lessons) for module in ready),
    }
    search: list[dict] = []

    def write(page: str, template: str, **context) -> None:
        body = context.get("body", "")
        rendered = templates.get_template(template).render(
            **shared,
            **context,
            page=page,
            root=LinkMap.relative(".", page) + "/",
            has_math='class="tex"' in body,
            url=lambda target: LinkMap.relative(target, page),
        )
        destination = output_dir / page
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")

    def neighbours(module: Module, page: str) -> tuple:
        order = module.reading_order()
        position = [path for path, _ in order].index(page)
        previous = order[position - 1] if position else None
        following = order[position + 1] if position + 1 < len(order) else None
        if following is None:
            later = ready[ready.index(module) + 1 :]
            if later:
                following = (
                    later[0].path,
                    f"Module {later[0].number}: {later[0].title}",
                )
        return previous, following

    def markdown_page(
        module: Module, source: Path, page: str, title: str, eyebrow: str
    ) -> None:
        """Render a README-style page that belongs to a module."""
        nonlocal search
        env = new_env(links, source.parent, page, title)
        text = source.read_text(encoding="utf-8")
        body = MARKDOWN.render(re.sub(r"^\s*#\s+.*\n", "", text, count=1), env)
        search += search_entries(env, page, title, f"Module {module.number}")
        previous, following = neighbours(module, page)
        write(
            page,
            "page.html",
            title=title,
            eyebrow=eyebrow,
            body=body,
            toc=env["toc"],
            module=module,
            previous=previous,
            following=following,
        )

    for module in ready:
        directory = MODULES_DIR / module.slug
        group = f"Module {module.number}"
        markdown_page(module, directory / "README.md", module.path, module.title, group)
        if module.has_exercises:
            markdown_page(
                module,
                directory / "exercises" / "README.md",
                module.exercises_path,
                "Exercises",
                f"{group}: {module.title}",
            )
        for lesson in module.lessons:
            body, env = render_notebook(lesson, links, output_dir)
            search += search_entries(env, lesson.path, lesson.title, group)
            previous, following = neighbours(module, lesson.path)
            write(
                lesson.path,
                "page.html",
                title=lesson.title,
                eyebrow=f"{group} · Lesson {lesson.number}",
                body=body,
                toc=env["toc"],
                module=module,
                lesson=lesson,
                previous=previous,
                following=following,
            )

    # Getting started, from the root README.
    guide_page = "getting-started.html"
    env = new_env(links, REPO_ROOT, guide_page, "Getting started")
    guide = readme_sections(
        readme, ["Getting Started", "How to Study", "Repository Layout", "Conventions"]
    )
    body = MARKDOWN.render(guide, env)
    search += search_entries(env, guide_page, "Getting started", "Guide")
    write(
        guide_page,
        "page.html",
        title="Getting started",
        eyebrow="Guide",
        body=body,
        toc=env["toc"],
    )

    # Home.
    intro = re.search(r"^# [^\n]+\n\n(.+?)\n\n", readme, re.MULTILINE | re.DOTALL)
    stack_map = links.rewrite(
        "assets/images/course_stack_map.png", REPO_ROOT, "index.html"
    )
    write(
        "index.html",
        "home.html",
        title="Full-Stack Data Science with Python",
        intro=" ".join(intro[1].split()).replace("*", "") if intro else "",
        philosophy=parse_philosophy(readme),
        stack_map=stack_map,
    )

    for asset in links.assets:
        destination = output_dir / asset
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / asset, destination)

    payload = json.dumps(search, ensure_ascii=False, separators=(",", ":"))
    (output_dir / "static" / "search-index.js").write_text(
        f"window.SEARCH_INDEX={payload};", encoding="utf-8"
    )

    problems = verify_links(output_dir)
    if problems:
        raise BuildError("broken links:\n  " + "\n  ".join(problems))
    return {
        "modules": len(ready),
        "lessons": shared["total_lessons"],
        "pages": sum(1 for _ in output_dir.rglob("*.html")),
    }


def serve(directory: Path, port: int) -> None:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Serving on http://127.0.0.1:{port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--serve", action="store_true", help="serve the site after building"
    )
    parser.add_argument("--port", type=int, default=8000, help="port for --serve")
    args = parser.parse_args()
    try:
        stats = build()
    except BuildError as err:
        print(f"Build failed: {err}", file=sys.stderr)
        return 1
    print(
        f"Built {stats['pages']} pages ({stats['modules']} modules, "
        f"{stats['lessons']} lessons) in {OUTPUT_DIR}"
    )
    if args.serve:
        serve(OUTPUT_DIR, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
