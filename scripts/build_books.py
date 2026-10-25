"""Build the illustrated companion PDF for each course module.

Every module ships a companion *book*: the same concepts as its notebooks, told
in a lighter voice, illustrated, and interleaved with short stories from the
history of data science. The books are authored as plain-text Markdown (one
file per chapter, under ``books/``) and rendered to PDF with ReportLab, so that
the prose diffs cleanly in git exactly like the notebook sources do.

    # build every book
    .venv/bin/python scripts/build_books.py

    # build one module's book
    .venv/bin/python scripts/build_books.py --module 02

Generate the figures first (``scripts/generate_book_figures.py``); the build
refuses to run if a referenced image is missing rather than shipping a book with
holes in it.

The Markdown dialect
--------------------
A deliberately small subset, so every construct has one obvious rendering:

* ``# Title`` opens a chapter; ``##`` / ``###`` are numbered sections.
* ``@subtitle``, ``@time``, ``@notebook`` and an ``@learn`` … ``@end`` list
  directly under the ``#`` line fill in the chapter opener.
* ``> [!NOTE]`` / ``[!TIP]`` / ``[!WARNING]`` / ``[!KEY]`` / ``[!STORY] Title``
  open a callout; every following ``>`` line belongs to it, and its body may
  contain lists and code.
* ``![Caption](path/to/image.png){width=80}`` is a numbered figure; ``width`` is
  a percentage of the text column.
* Fenced code blocks, pipe tables, ``-`` and ``1.`` lists, ``**bold**``,
  ``*italic*``, `` `code` `` and ``[links](https://…)``.
* ``\\pagebreak`` on its own line forces a new page.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = REPO_ROOT / "books"
FONTS_DIR = REPO_ROOT / "assets" / "fonts"
MPL_FONTS_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

sys.path.insert(0, str(REPO_ROOT / "src"))
from dstoolkit.plotting import COLORS  # noqa: E402

# ---------------------------------------------------------------------------
# Page geometry. A4 with a ~85-character measure: wide enough for code, narrow
# enough that a line of prose stays comfortable to read.
# ---------------------------------------------------------------------------
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 25 * mm
MARGIN_TOP = 24 * mm
MARGIN_BOTTOM = 22 * mm
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

# Ink and surfaces. Series colours come from the course palette; the tints are
# the same hues mixed toward white, used only as callout backgrounds.
INK = colors.HexColor(COLORS["text"])
INK_SOFT = colors.HexColor("#4A4F57")
MUTED = colors.HexColor("#6B7078")
RULE = colors.HexColor("#D9DCE1")
CODE_BG = colors.HexColor("#F5F6F8")

CALLOUTS = {
    "NOTE": ("Note", COLORS["primary"], "#EEF3FA"),
    "TIP": ("Tip", COLORS["teal"], "#E8F5F3"),
    "WARNING": ("Watch out", COLORS["danger"], "#FBEDEC"),
    "KEY": ("Key takeaways", COLORS["accent"], "#FBF1E9"),
    "STORY": ("From the history of data science", COLORS["purple"], "#F3EEF7"),
}

# Syntax colours: keywords and names in the categorical hues, comments muted.
TOKEN_COLORS = [
    (Token.Comment, "#7A7F87"),
    (Token.Keyword, COLORS["purple"]),
    (Token.Name.Builtin, COLORS["primary"]),
    (Token.Name.Function, COLORS["primary"]),
    (Token.Name.Class, COLORS["primary"]),
    (Token.Name.Decorator, COLORS["accent"]),
    (Token.Literal.String, COLORS["teal"]),
    (Token.Literal.Number, COLORS["accent"]),
    (Token.Operator.Word, COLORS["purple"]),
]


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT_FILES = {
    "Serif": FONTS_DIR / "SourceSerif4-Regular.ttf",
    "Serif-Bold": FONTS_DIR / "SourceSerif4-Semibold.ttf",
    "Serif-Italic": FONTS_DIR / "SourceSerif4-It.ttf",
    "Serif-BoldItalic": FONTS_DIR / "SourceSerif4-SemiboldIt.ttf",
    "Sans": FONTS_DIR / "Inter-Regular.ttf",
    "Sans-Medium": FONTS_DIR / "Inter-Medium.ttf",
    "Sans-Bold": FONTS_DIR / "Inter-SemiBold.ttf",
    "Sans-Italic": FONTS_DIR / "Inter-Italic.ttf",
    "Sans-Heavy": FONTS_DIR / "InterDisplay-ExtraBold.ttf",
    "Display": FONTS_DIR / "InterDisplay-Bold.ttf",
    "Mono": FONTS_DIR / "JetBrainsMono-Regular.ttf",
    "Mono-Bold": FONTS_DIR / "JetBrainsMono-Bold.ttf",
    "Mono-Italic": FONTS_DIR / "JetBrainsMono-Italic.ttf",
    # Fallbacks for the handful of symbols the text faces do not cover
    # (✔, ✘, ᵀ, ‖ …). DejaVu ships inside matplotlib, so it is always present.
    "Fallback": MPL_FONTS_DIR / "DejaVuSans.ttf",
    "Fallback-Mono": MPL_FONTS_DIR / "DejaVuSansMono.ttf",
}

# Which character map to consult for each face, and where a missing glyph goes.
FALLBACK_FOR = {"Mono": "Fallback-Mono"}
CMAPS: dict[str, set[int]] = {}
# Characters that not even the fallback face can draw; reported after a build.
UNDRAWABLE: set[str] = set()


def register_fonts() -> None:
    """Register every face with ReportLab and wire up bold/italic families."""
    for name, path in FONT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"font missing: {path}")
        font = TTFont(name, str(path))
        pdfmetrics.registerFont(font)
        CMAPS[name] = set(font.face.charToGlyph)

    pdfmetrics.registerFontFamily(
        "Serif", normal="Serif", bold="Serif-Bold",
        italic="Serif-Italic", boldItalic="Serif-BoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "Sans", normal="Sans", bold="Sans-Bold",
        italic="Sans-Italic", boldItalic="Sans-Bold",
    )
    pdfmetrics.registerFontFamily(
        "Mono", normal="Mono", bold="Mono-Bold",
        italic="Mono-Italic", boldItalic="Mono-Bold",
    )


def family_of(font_name: str) -> str:
    """Map a registered face to the family whose character map applies."""
    return font_name.split("-")[0]


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
def build_styles() -> dict[str, ParagraphStyle]:
    """Return the full style sheet, keyed by role."""
    body = ParagraphStyle(
        "body", fontName="Serif", fontSize=10.6, leading=15.6,
        textColor=INK, spaceAfter=7.5, alignment=TA_LEFT,
        allowWidows=0, allowOrphans=0,
    )
    styles = {
        "body": body,
        "lead": ParagraphStyle(
            "lead", parent=body, fontSize=12.2, leading=18, textColor=INK_SOFT,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", fontName="Sans-Bold", fontSize=15, leading=19, textColor=INK,
            spaceBefore=16, spaceAfter=7, keepWithNext=1,
        ),
        "h3": ParagraphStyle(
            "h3", fontName="Sans-Bold", fontSize=11.6, leading=15, textColor=INK,
            spaceBefore=11, spaceAfter=4.5, keepWithNext=1,
        ),
        "h4": ParagraphStyle(
            "h4", fontName="Sans-Bold", fontSize=10, leading=13.5, textColor=INK_SOFT,
            spaceBefore=8, spaceAfter=3, keepWithNext=1,
        ),
        "list": ParagraphStyle("list", parent=body, spaceAfter=2.5),
        "cell": ParagraphStyle(
            "cell", fontName="Serif", fontSize=9.2, leading=12.4, textColor=INK,
        ),
        "cell_head": ParagraphStyle(
            "cell_head", fontName="Sans-Bold", fontSize=8.6, leading=11.4,
            textColor=INK,
        ),
        "caption": ParagraphStyle(
            "caption", fontName="Sans", fontSize=8.4, leading=11.6,
            textColor=MUTED, alignment=TA_LEFT, spaceBefore=4,
        ),
        "code": ParagraphStyle(
            "code", fontName="Mono", fontSize=8.15, leading=11.4, textColor=INK,
        ),
        "callout_label": ParagraphStyle(
            "callout_label", fontName="Sans-Bold", fontSize=7.8, leading=10,
            spaceAfter=3.5,
        ),
        "callout_title": ParagraphStyle(
            "callout_title", fontName="Sans-Bold", fontSize=11, leading=14,
            textColor=INK, spaceAfter=4,
        ),
        "toc0": ParagraphStyle(
            "toc0", fontName="Sans-Bold", fontSize=11, leading=15,
            leftIndent=0, spaceBefore=7, textColor=INK,
        ),
        "toc1": ParagraphStyle(
            "toc1", fontName="Serif", fontSize=9.6, leading=13,
            leftIndent=22, textColor=INK_SOFT,
        ),
    }
    styles["callout_body"] = ParagraphStyle(
        "callout_body", parent=body, fontSize=10, leading=14.6, spaceAfter=5,
    )
    return styles


# ---------------------------------------------------------------------------
# Inline Markdown → ReportLab paragraph markup
# ---------------------------------------------------------------------------
def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


INLINE_PATTERN = re.compile(
    r"(`[^`]+`)"                              # code span
    r"|(\[[^\]]+\]\([^)\s]+\))"               # link
    r"|(\*\*[^*]+?\*\*)"                      # bold
    r"|(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])"  # italic
)


def smart_quotes(text: str) -> str:
    """Curl straight quotes in prose: "x" → “x”, it's → it’s."""
    text = re.sub(r'(^|[\s(\[—–-])"', "\\1\u201c", text)
    text = text.replace('"', "\u201d")
    text = re.sub(r"(^|[\s(\[—–-])'", "\\1\u2018", text)
    return text.replace("'", "\u2019")


def inline(text: str, *, code_size: float = 9.0, _quoted: bool = False) -> str:
    """Convert inline Markdown to ReportLab's mini-HTML."""
    if not _quoted:
        # Curl quotes across the whole string at once (so a quote that closes
        # right after a code span still curls the right way), with code spans
        # and link targets shielded from the substitution.
        shielded: list[str] = []

        def shield(match: re.Match) -> str:
            shielded.append(match.group(0))
            return f"\x00{len(shielded) - 1}\x00"

        text = re.sub(r"`[^`]+`|\]\([^)\s]+\)", shield, text)
        text = smart_quotes(text)
        text = re.sub(r"\x00(\d+)\x00", lambda m: shielded[int(m.group(1))], text)

    out: list[str] = []
    position = 0
    for match in INLINE_PATTERN.finditer(text):
        out.append(escape(text[position:match.start()]))
        code, link, bold, italic = match.groups()
        if code:
            out.append(
                f'<font name="Mono" size="{code_size}" color="{COLORS["text"]}" '
                f'backColor="#EEF0F3">{escape(code[1:-1])}</font>'
            )
        elif link:
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", link).groups()
            out.append(
                f'<a href="{escape(url)}" color="{COLORS["primary"]}">'
                f"{inline(label, code_size=code_size, _quoted=True)}</a>"
            )
        elif bold:
            out.append(f"<b>{inline(bold[2:-2], code_size=code_size, _quoted=True)}</b>")
        else:
            out.append(f"<i>{inline(italic, code_size=code_size, _quoted=True)}</i>")
        position = match.end()
    out.append(escape(text[position:]))
    return "".join(out)


TAG_OR_ENTITY = re.compile(r"(<[^>]+>|&[a-zA-Z#0-9]+;)")


def with_fallback_glyphs(markup: str, base_font: str) -> str:
    """Wrap every character the active face cannot draw in a fallback face.

    ReportLab has no per-glyph fallback of its own, so a missing glyph would
    silently render as a black box. We walk the markup, track which face is
    active from the ``<font name=…>`` tags, and re-route uncovered characters.
    """
    stack = [family_of(base_font)]
    pieces: list[str] = []
    for part in TAG_OR_ENTITY.split(markup):
        if not part:
            continue
        if part.startswith("<"):
            name = re.search(r'name="([^"]+)"', part)
            if part.startswith("<font"):
                stack.append(family_of(name.group(1)) if name else stack[-1])
            elif part.startswith("</font"):
                stack.pop()
            pieces.append(part)
            continue
        if part.startswith("&"):
            pieces.append(part)
            continue

        pieces.append(reroute_missing(part, stack[-1]))
    return "".join(pieces)


# Prose whose rendered text still contains Markdown punctuation, which means
# an emphasis or code span failed to parse. Reported after each build.
UNPARSED_MARKUP: list[str] = []


def reroute_missing(text: str, family: str) -> str:
    """Wrap each run of characters the family cannot draw in its fallback face."""
    cmap = CMAPS[family]
    fallback = FALLBACK_FOR.get(family, "Fallback")
    out: list[str] = []
    missing: list[str] = []
    for char in text + "\n":                 # the sentinel flushes the last run
        if ord(char) in cmap or char in "\n\t":
            if missing:
                out.append(f'<font name="{fallback}">{"".join(missing)}</font>')
                missing.clear()
            out.append(char)
        else:
            if ord(char) not in CMAPS[fallback]:
                UNDRAWABLE.add(char)
            missing.append(char)
    return "".join(out[:-1])


def para(text: str, style: ParagraphStyle, *, raw: bool = False) -> Paragraph:
    """A Paragraph from Markdown text (or from ready markup when ``raw``)."""
    markup = text if raw else inline(text)
    # Strip code spans, then tags: whatever remains is the visible prose.
    visible = re.sub(r'<font name="Mono"[^>]*>.*?</font>', "", markup)
    visible = re.sub(r"<[^>]+>", "", visible)
    if "**" in visible or "`" in visible:
        UNPARSED_MARKUP.append(visible[:90])
    return Paragraph(with_fallback_glyphs(markup, style.fontName), style)


# ---------------------------------------------------------------------------
# Custom flowables
# ---------------------------------------------------------------------------
class ChapterOpener(Flowable):
    """The first block of every chapter: number, title, subtitle and a rule."""

    def __init__(self, number: int, title: str, subtitle: str, meta: str) -> None:
        super().__init__()
        self.number, self.title, self.subtitle, self.meta = number, title, subtitle, meta
        self.width, self.height = TEXT_WIDTH, 0
        self._title_lines: list[str] = []

    def wrap(self, avail_width, avail_height):
        # Break the title into lines that fit the display size.
        words, lines, line = self.title.split(), [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if pdfmetrics.stringWidth(trial, "Display", 30) > TEXT_WIDTH:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        self._title_lines = lines
        self.height = 96 + 36 * len(lines) + (22 if self.subtitle else 0)
        return self.width, self.height

    def draw(self):
        c = self.canv
        top = self.height
        c.setFillColor(colors.HexColor(COLORS["accent"]))
        c.setFont("Sans-Bold", 9)
        c.drawString(0, top - 10, f"CHAPTER {self.number:02d}")
        c.setFillColor(colors.HexColor(COLORS["primary"]))
        c.setFont("Sans-Heavy", 64)
        c.drawString(-2, top - 72, f"{self.number:02d}")

        y = top - 108
        c.setFillColor(INK)
        c.setFont("Display", 30)
        for line in self._title_lines:
            c.drawString(0, y, line)
            y -= 36

        if self.subtitle:
            c.setFillColor(INK_SOFT)
            c.setFont("Serif-Italic", 13.5)
            c.drawString(0, y + 8, self.subtitle)
            y -= 22

        c.setStrokeColor(colors.HexColor(COLORS["accent"]))
        c.setLineWidth(2.2)
        c.line(0, y + 14, 46, y + 14)
        c.setFillColor(MUTED)
        c.setFont("Sans", 8.6)
        c.drawString(56, y + 11, self.meta)


class Bookmark(Flowable):
    """Zero-size marker that records a TOC entry and a PDF outline entry."""

    def __init__(self, title: str, level: int, key: str, chapter: str = "") -> None:
        super().__init__()
        # The outline and TOC show plain text: drop Markdown code/emphasis marks.
        title = re.sub(r"[`*]", "", title)
        self.title, self.level, self.key, self.chapter = title, level, key, chapter
        self.width = self.height = 0

    def wrap(self, *_):
        return 0, 0

    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=True)


def callout(kind: str, title: str, body: list[Flowable], styles) -> Flowable:
    """A tinted panel with a coloured rule on the left and a small label."""
    label, accent, tint = CALLOUTS[kind]
    label_style = ParagraphStyle(
        f"label_{kind}", parent=styles["callout_label"],
        textColor=colors.HexColor(accent),
    )
    content: list[Flowable] = [Paragraph(label.upper(), label_style)]
    if title:
        content.append(para(title, styles["callout_title"]))
    content.extend(body)

    # One row per flowable: a table can break between rows but never inside a
    # cell, so this lets a long story continue on the next page instead of
    # leaving a hole at the bottom of this one.
    header = content[:3] if title else content[:2]
    rows = [[header]] + [[f] for f in content[len(header):]]
    table = Table(rows, colWidths=[TEXT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(tint)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(accent)),
        ("LEFTPADDING", (0, 0), (-1, -1), 13),
        ("RIGHTPADDING", (0, 0), (-1, -1), 13),
        # Table cells ignore a paragraph's spaceAfter, so the gap between the
        # callout's paragraphs is made with row padding instead.
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
    ]))
    return table


def code_block(source: str, language: str, styles, *, width: float = TEXT_WIDTH,
               where: str = "") -> Flowable:
    """Syntax-highlighted code on a light panel."""
    # JetBrains Mono advances 0.6 em per character; padding takes 18 pt.
    max_columns = int((width - 18) / (0.6 * styles["code"].fontSize))
    for line in source.splitlines():
        if len(line) > max_columns:
            print(f"  ! code line {len(line)} cols in {where}: {line[:50]}…")

    if language in {"python", "py", "pycon"}:
        markup = []
        for token_type, value in lex(source, get_lexer_by_name("python")):
            text = escape(value)
            color = next(
                (c for t, c in TOKEN_COLORS if token_type in t), None
            )
            if color and text.strip():
                italic = token_type in Token.Comment
                text = f'<font color="{color}">{text}</font>'
                if italic:
                    text = f"<i>{text}</i>"
            markup.append(text)
        markup_text = "".join(markup).rstrip("\n")
    else:
        markup_text = escape(source.rstrip("\n"))

    # XPreformatted (unlike a one-cell table) can split across a page break.
    # The panel is drawn by the paragraph's own border and background; the
    # indents keep that border inside the text column.
    style = ParagraphStyle(
        "code_block", parent=styles["code"], backColor=CODE_BG,
        borderColor=RULE, borderWidth=0.6, borderPadding=(7, 8, 7, 10),
        leftIndent=10, rightIndent=8 + (TEXT_WIDTH - width),
    )
    return XPreformatted(with_fallback_glyphs(markup_text, "Mono"), style)


def pipe_table(rows: list[list[str]], styles, *, width: float = TEXT_WIDTH) -> Flowable:
    """A booktabs-style table: rules only where they help the eye."""
    header, *body = rows
    # Column widths proportional to content length, with a floor so short
    # columns (✔/✘ flags) never collapse.
    lengths = [
        max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(len(header))
    ]
    # ...but never narrower than the longest single word in the column, or a
    # short code cell like `finally` gets broken mid-word.
    longest_word = [
        max((len(w) for row in rows if i < len(row) for w in row[i].split()), default=1)
        for i in range(len(header))
    ]
    weights = [min(max(n, 6, int(w * 1.6)), 60)
               for n, w in zip(lengths, longest_word, strict=True)]
    widths = [width * w / sum(weights) for w in weights]

    data = [[para(cell, styles["cell_head"]) for cell in header]]
    for row in body:
        row = row + [""] * (len(header) - len(row))
        data.append([para(cell, styles["cell"]) for cell in row])

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7F8FA")))
    table.setStyle(TableStyle(style))
    table.spaceBefore, table.spaceAfter = 4, 11
    return table


# ---------------------------------------------------------------------------
# Block parser
# ---------------------------------------------------------------------------
@dataclass
class Chapter:
    number: int
    title: str
    subtitle: str = ""
    time: str = ""
    notebook: str = ""
    learn: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass
class BookContext:
    module_label: str
    styles: dict
    source: Path
    chapter_number: int = 0
    section_number: int = 0
    figure_number: int = 0
    keys: int = 0
    missing_images: list[str] = field(default_factory=list)

    def next_key(self) -> str:
        self.keys += 1
        return f"k{self.keys}"


LIST_ITEM = re.compile(r"^(\s*)(?:[-*]|(\d+)\.)\s+(.*)$")
FIGURE = re.compile(r"^!\[(.*)\]\(([^)]+)\)(?:\{width=(\d+)\})?\s*$")
CALLOUT_OPEN = re.compile(r"^\[!(\w+)\]\s*(.*)$")


def parse_blocks(lines: list[str], ctx: BookContext, *, width: float = TEXT_WIDTH,
                 body_style: str = "body") -> list[Flowable]:
    """Turn a run of Markdown lines into flowables."""
    styles = ctx.styles
    flow: list[Flowable] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "\\pagebreak":
            flow.append(PageBreak())
            i += 1
            continue

        # --- headings ---------------------------------------------------------
        if stripped.startswith("## "):
            ctx.section_number += 1
            title = stripped[3:].strip()
            key = ctx.next_key()
            flow.append(CondPageBreak(70 * mm if ctx.section_number > 1 else 40 * mm))
            if ctx.chapter_number:
                numbered = f"{ctx.chapter_number}.{ctx.section_number}"
                flow.append(Bookmark(f"{numbered}  {title}", 1, key))
                flow.append(para(
                    f'<font color="{COLORS["primary"]}">{numbered}</font>&nbsp;&nbsp;'
                    + inline(title), styles["h2"], raw=True,
                ))
            else:
                # Back matter: sections are listed in the outline, not numbered.
                flow.append(Bookmark(title, 1, key))
                flow.append(para(title, styles["h2"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flow.append(para(stripped[4:].strip(), styles["h3"]))
            i += 1
            continue

        if stripped.startswith("#### "):
            flow.append(para(stripped[5:].strip(), styles["h4"]))
            i += 1
            continue

        # --- fenced code ------------------------------------------------------
        if stripped.startswith("```"):
            language = stripped[3:].strip() or "text"
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            source = "\n".join(lines[i + 1:j])
            # Explicit spacers rather than spaceBefore/After: table cells (and so
            # callouts) ignore flowable spacing, and the panel's border padding
            # would otherwise overlap the paragraph above it.
            flow.append(Spacer(0, 9))
            flow.append(code_block(source, language, styles, width=width,
                                   where=ctx.source.name))
            flow.append(Spacer(0, 12))
            i = j + 1
            continue

        # --- figures ----------------------------------------------------------
        figure = FIGURE.match(stripped)
        if figure:
            caption, path, pct = figure.groups()
            flow.append(make_figure(caption, path, int(pct or 100), ctx, width))
            i += 1
            continue

        # --- callouts ---------------------------------------------------------
        if stripped.startswith(">"):
            j = i
            quoted: list[str] = []
            while j < len(lines) and lines[j].strip().startswith(">"):
                quoted.append(re.sub(r"^\s*> ?", "", lines[j]))
                j += 1
            opener = CALLOUT_OPEN.match(quoted[0].strip())
            kind, title = (opener.group(1), opener.group(2)) if opener else ("NOTE", "")
            if kind not in CALLOUTS:
                raise ValueError(f"{ctx.source.name}: unknown callout [!{kind}]")
            inner = parse_blocks(quoted[1:] if opener else quoted, ctx,
                                 width=TEXT_WIDTH - 30, body_style="callout_body")
            box = callout(kind, title, inner, styles)
            box.spaceBefore, box.spaceAfter = 5, 12
            flow.append(box)
            i = j
            continue

        # --- tables -----------------------------------------------------------
        if stripped.startswith("|"):
            j = i
            rows: list[list[str]] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                j += 1
            flow.append(pipe_table(rows, styles, width=width))
            i = j
            continue

        # --- lists ------------------------------------------------------------
        if LIST_ITEM.match(line):
            j = i
            items: list[str] = []
            ordered = LIST_ITEM.match(line).group(2) is not None
            while j < len(lines):
                m = LIST_ITEM.match(lines[j])
                if m:
                    items.append(m.group(3))
                elif lines[j].strip() and lines[j].startswith("  ") and items:
                    items[-1] += " " + lines[j].strip()     # wrapped continuation
                else:
                    break
                j += 1
            flow.append(make_list(items, ordered, styles, body_style))
            i = j
            continue

        # --- paragraphs -------------------------------------------------------
        # A paragraph always takes at least its first line, so a stray line
        # that merely looks like a block (e.g. "#hashtag") cannot stall the loop.
        j = i + 1
        chunk: list[str] = [stripped]
        while j < len(lines) and lines[j].strip() and not starts_block(lines[j]):
            chunk.append(lines[j].strip())
            j += 1
        style = styles["lead"] if chunk and chunk[0].startswith("@lead ") else styles[body_style]
        text = " ".join(chunk).removeprefix("@lead ")
        flow.append(para(text, style))
        i = j

    return flow


def starts_block(line: str) -> bool:
    s = line.strip()
    return (
        s.startswith(("## ", "### ", "#### ", "```", ">", "|", "\\pagebreak"))
        or bool(FIGURE.match(s))
        or bool(LIST_ITEM.match(line))
    )


def make_list(items: list[str], ordered: bool, styles, body_style: str) -> Flowable:
    style = ParagraphStyle(
        "li", parent=styles[body_style], spaceAfter=2.8,
    )
    flow_items = [ListItem(para(text, style), leftIndent=15) for text in items]
    lst = ListFlowable(
        flow_items,
        bulletType="1" if ordered else "bullet",
        start="1" if ordered else "•",
        bulletFontName="Sans-Bold" if ordered else "Sans",
        bulletFontSize=8.6 if ordered else 9,
        bulletColor=colors.HexColor(COLORS["primary"]),
        leftIndent=15,
        bulletDedent=12,
    )
    lst.spaceBefore, lst.spaceAfter = 0, 7
    return lst


def make_figure(caption: str, path: str, pct: int, ctx: BookContext,
                width: float) -> Flowable:
    """A centred, numbered figure with its caption kept on the same page."""
    image_path = (REPO_ROOT / path).resolve()
    if not image_path.exists():
        ctx.missing_images.append(path)
        return Spacer(0, 0)

    ctx.figure_number += 1
    reader = ImageReader(str(image_path))
    px_w, px_h = reader.getSize()
    draw_w = width * pct / 100
    draw_h = draw_w * px_h / px_w
    # Never let a tall figure exceed ~60% of the page body.
    max_h = (PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM) * 0.6
    if draw_h > max_h:
        draw_w, draw_h = draw_w * max_h / draw_h, max_h

    image = Image(str(image_path), width=draw_w, height=draw_h)
    image.hAlign = "CENTER"
    label = f"Figure {ctx.chapter_number}.{ctx.figure_number}"
    cap = para(
        f'<font name="Sans-Bold" color="{COLORS["text"]}">{label}</font>'
        f"&nbsp;&nbsp;{inline(caption, code_size=7.8)}",
        ctx.styles["caption"], raw=True,
    )
    block = KeepTogether([Spacer(0, 4), image, cap, Spacer(0, 10)])
    return block


# ---------------------------------------------------------------------------
# Chapter files
# ---------------------------------------------------------------------------
def load_chapter(path: Path, number: int) -> Chapter:
    """Split a chapter file into its opener metadata and its body lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path.name}: first line must be '# Title'")

    chapter = Chapter(number=number, title=lines[0][2:].strip())
    i = 1
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("@subtitle "):
            chapter.subtitle = s.removeprefix("@subtitle ").strip()
        elif s.startswith("@time "):
            chapter.time = s.removeprefix("@time ").strip()
        elif s.startswith("@notebook "):
            chapter.notebook = s.removeprefix("@notebook ").strip()
        elif s == "@learn":
            i += 1
            while lines[i].strip() != "@end":
                if lines[i].strip():
                    chapter.learn.append(re.sub(r"^\s*[-*]\s+", "", lines[i]).strip())
                i += 1
        elif s:
            break
        i += 1
    chapter.lines = lines[i:]
    return chapter


@dataclass
class BookMeta:
    module_number: str
    module_title: str
    tagline: str
    cover_image: str
    author: str
    subject: str


def load_meta(book_dir: Path) -> BookMeta:
    """Read ``book.toml`` (a tiny key = "value" file)."""
    import tomllib

    data = tomllib.loads((book_dir / "book.toml").read_text(encoding="utf-8"))
    return BookMeta(**data)


# ---------------------------------------------------------------------------
# Document template: cover, front matter, chapters
# ---------------------------------------------------------------------------
class BookDocument(BaseDocTemplate):
    """Tracks the running chapter title and feeds the table of contents."""

    def __init__(self, filename: str, meta: BookMeta, **kwargs) -> None:
        super().__init__(filename, pagesize=A4, **kwargs)
        self.meta = meta
        self.running_chapter = ""
        frame = Frame(MARGIN_X, MARGIN_BOTTOM, TEXT_WIDTH,
                      PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
                      id="body", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        cover_frame = Frame(0, 0, PAGE_WIDTH, PAGE_HEIGHT, id="cover",
                            leftPadding=0, rightPadding=0, topPadding=0,
                            bottomPadding=0)
        self.addPageTemplates([
            PageTemplate("cover", [cover_frame], onPage=self.draw_cover),
            PageTemplate("plain", [frame], onPage=self.draw_plain),
            PageTemplate("opener", [frame], onPage=self.draw_footer_only),
            PageTemplate("body", [frame], onPage=self.draw_running),
        ])

    # -- page decorations ----------------------------------------------------
    def draw_cover(self, canvas, doc) -> None:
        meta = self.meta
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#FBFAF7"))
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)

        cover = REPO_ROOT / meta.cover_image
        if cover.exists():
            reader = ImageReader(str(cover))
            w, h = reader.getSize()
            draw_w = PAGE_WIDTH - 2 * MARGIN_X
            draw_h = draw_w * h / w
            canvas.drawImage(reader, MARGIN_X, PAGE_HEIGHT - 52 * mm - draw_h,
                             width=draw_w, height=draw_h, mask="auto")

        canvas.setFillColor(colors.HexColor(COLORS["accent"]))
        canvas.setFont("Sans-Bold", 10)
        canvas.drawString(MARGIN_X, PAGE_HEIGHT - 30 * mm,
                          "FULL-STACK DATA SCIENCE WITH PYTHON")
        canvas.setFillColor(MUTED)
        canvas.setFont("Sans", 10)
        canvas.drawString(MARGIN_X, PAGE_HEIGHT - 36 * mm,
                          "An illustrated companion to the course notebooks")

        base = 62 * mm
        canvas.setFillColor(colors.HexColor(COLORS["primary"]))
        canvas.setFont("Sans-Heavy", 15)
        canvas.drawString(MARGIN_X, base + 44 * mm, f"MODULE {meta.module_number}")
        canvas.setFillColor(INK)
        canvas.setFont("Display", 38)
        y = base + 28 * mm
        for line in meta.module_title.split("\\n"):
            canvas.drawString(MARGIN_X, y, line)
            y -= 14.5 * mm
        canvas.setFillColor(INK_SOFT)
        canvas.setFont("Serif-Italic", 14)
        canvas.drawString(MARGIN_X, y + 3 * mm, meta.tagline)

        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.8)
        canvas.line(MARGIN_X, 30 * mm, PAGE_WIDTH - MARGIN_X, 30 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Sans", 8.6)
        canvas.drawString(MARGIN_X, 23 * mm, meta.author)
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 23 * mm,
                               "github.com/Filip3Owl/full-stack-data-science")
        canvas.restoreState()

    def _page_number(self, canvas) -> None:
        canvas.setFont("Sans", 8.4)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(PAGE_WIDTH / 2, 12 * mm, str(canvas.getPageNumber()))

    def draw_plain(self, canvas, doc) -> None:
        canvas.saveState()
        self._page_number(canvas)
        canvas.restoreState()

    def draw_footer_only(self, canvas, doc) -> None:
        canvas.saveState()
        self._page_number(canvas)
        canvas.restoreState()

    def draw_running(self, canvas, doc) -> None:
        canvas.saveState()
        y = PAGE_HEIGHT - 14 * mm
        canvas.setFont("Sans-Medium", 7.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN_X, y,
                          f"MODULE {self.meta.module_number}  ·  "
                          f"{self.meta.module_title.replace(chr(92) + 'n', ' ').upper()}")
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, y, self.running_chapter)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_X, y - 3.2 * mm, PAGE_WIDTH - MARGIN_X, y - 3.2 * mm)
        self._page_number(canvas)
        canvas.restoreState()

    # -- TOC wiring ----------------------------------------------------------
    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Bookmark):
            if flowable.level == 0:
                self.running_chapter = flowable.chapter
            self.notify("TOCEntry", (flowable.level, flowable.title,
                                     self.page, flowable.key))


def front_matter(meta: BookMeta, chapters: list[Chapter], styles) -> list[Flowable]:
    """The 'how to read this book' page and the table of contents."""
    flow: list[Flowable] = [NextPageTemplate("plain"), PageBreak()]

    flow.append(para("How to read this book", ParagraphStyle(
        "front_h", fontName="Display", fontSize=24, leading=30, textColor=INK,
        spaceAfter=12)))
    flow.append(para(
        "@lead This book walks through every idea in the Module "
        f"{meta.module_number} notebooks, but it is written to be *read*, on a "
        "train or with a coffee, rather than executed. The notebooks remain the "
        "place where you run code, break it and fix it; the book is where you "
        "build the mental pictures that make the code obvious.".removeprefix("@lead "),
        styles["lead"]))
    flow.append(para(
        "Each chapter mirrors one notebook and opens with the notebook's file name, "
        "so you can always jump between the two. Code in the book is trimmed to the "
        "lines that carry the idea, and every figure was generated from code in "
        "this repository, using the same colour palette as the notebooks.",
        styles["body"]))
    flow.append(para(
        "Five kinds of boxes appear along the way. They always mean the same thing:",
        styles["body"]))

    legend = [
        ("NOTE", "Extra context. Worth knowing, but you can move on without it."),
        ("TIP", "A habit or shortcut professionals actually use."),
        ("WARNING", "A mistake that produces silently wrong numbers, not an error."),
        ("STORY", "A true episode from the history of science and industry that "
                  "makes the idea stick. Sources are listed at the back."),
        ("KEY", "The handful of ideas to carry out of the chapter."),
    ]
    for kind, text in legend:
        box = callout(kind, "", [para(text, styles["callout_body"])], styles)
        box.spaceAfter = 6
        flow.append(box)

    flow.append(PageBreak())
    flow.append(para("Contents", ParagraphStyle(
        "front_h2", fontName="Display", fontSize=24, leading=30, textColor=INK,
        spaceAfter=14)))
    toc = TableOfContents()
    toc.levelStyles = [styles["toc0"], styles["toc1"]]
    toc.dotsMinLevel = 0
    flow.append(toc)
    return flow


def chapter_flowables(chapter: Chapter, ctx: BookContext) -> list[Flowable]:
    styles = ctx.styles
    ctx.chapter_number = chapter.number
    ctx.section_number = 0
    ctx.figure_number = 0

    meta_bits = [b for b in [
        f"Notebook {chapter.notebook}" if chapter.notebook else "",
        chapter.time,
    ] if b]

    flow: list[Flowable] = [NextPageTemplate("opener"), PageBreak(),
                            NextPageTemplate("body")]
    key = ctx.next_key()
    flow.append(Bookmark(f"{chapter.number}  {chapter.title}", 0, key,
                         chapter=f"{chapter.number:02d}  ·  {chapter.title}"))
    flow.append(Spacer(0, 14 * mm))
    flow.append(ChapterOpener(chapter.number, chapter.title, chapter.subtitle,
                              "   ·   ".join(meta_bits)))
    flow.append(Spacer(0, 8))

    if chapter.learn:
        learn_style = ParagraphStyle("learn", parent=styles["callout_body"],
                                     spaceAfter=2.5)
        items = [ListItem(para(t, learn_style), leftIndent=14) for t in chapter.learn]
        learn_list = ListFlowable(items, bulletType="bullet", start="›",
                                  bulletFontName="Sans-Bold", bulletFontSize=10,
                                  bulletColor=colors.HexColor(COLORS["primary"]),
                                  leftIndent=14, bulletDedent=11)
        heading = ParagraphStyle("learn_h", parent=styles["callout_label"],
                                 textColor=colors.HexColor(COLORS["primary"]))
        panel = Table([[[Paragraph("IN THIS CHAPTER", heading), learn_list]]],
                      colWidths=[TEXT_WIDTH])
        panel.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 0.6, RULE),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        panel.spaceAfter = 14
        flow.append(panel)

    flow.extend(parse_blocks(chapter.lines, ctx))
    return flow


def build_book(book_dir: Path) -> Path:
    """Render one module's book directory into its PDF; return the PDF path."""
    meta = load_meta(book_dir)
    styles = build_styles()
    chapter_files = sorted(p for p in book_dir.glob("[0-9][0-9]_*.md"))
    chapters = [load_chapter(p, i) for i, p in enumerate(chapter_files, start=1)]

    module_dir = next(REPO_ROOT.glob(f"modules/{meta.module_number}-*"))
    output = module_dir / "pdf" / f"{book_dir.name}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    ctx = BookContext(module_label=meta.module_title, styles=styles, source=book_dir)
    story: list[Flowable] = [Spacer(0, 0)]       # the cover is drawn by its template
    story.extend(front_matter(meta, chapters, styles))
    for chapter, path in zip(chapters, chapter_files, strict=True):
        ctx.source = path
        story.extend(chapter_flowables(chapter, ctx))

    # Back matter (sources, self-check) lives in files that do not start with a
    # digit, so it is not numbered as a chapter.
    for extra in sorted(book_dir.glob("[a-z]*.md")):
        ctx.source = extra
        lines = extra.read_text(encoding="utf-8").splitlines()
        title = lines[0][2:].strip()
        key = ctx.next_key()
        story.extend([NextPageTemplate("opener"), PageBreak(), NextPageTemplate("body"),
                      Bookmark(title, 0, key, chapter=title),
                      para(title, ParagraphStyle(
                          "back_h", fontName="Display", fontSize=24, leading=30,
                          textColor=INK, spaceAfter=12))])
        ctx.chapter_number, ctx.section_number = 0, 0
        story.extend(parse_blocks(lines[1:], ctx))

    if ctx.missing_images:
        missing = "\n  ".join(sorted(set(ctx.missing_images)))
        raise FileNotFoundError(
            f"missing figures — run scripts/generate_book_figures.py first:\n  {missing}"
        )

    doc = BookDocument(
        str(output), meta,
        leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title=f"Module {meta.module_number}: {meta.module_title.replace(chr(92) + 'n', ' ')}",
        author=meta.author, subject=meta.subject,
        creator="scripts/build_books.py (ReportLab)", lang="en",
    )
    doc.multiBuild(story)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", help="module number prefix, e.g. 01")
    args = parser.parse_args()

    register_fonts()
    pattern = f"{args.module}-*" if args.module else "[0-9][0-9]-*"
    book_dirs = sorted(d for d in BOOKS_DIR.glob(pattern) if d.is_dir())
    if not book_dirs:
        print(f"no books match {pattern!r} under {BOOKS_DIR}")
        return 1

    for book_dir in book_dirs:
        print(f"building {book_dir.name} …")
        output = build_book(book_dir)
        size_mb = output.stat().st_size / 1024**2
        print(f"  ✔ {output.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")
    if UNDRAWABLE:
        print(f"  ! no font can draw: {' '.join(sorted(UNDRAWABLE))}")
    for snippet in UNPARSED_MARKUP:
        print(f"  ! unparsed Markdown in prose: {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
