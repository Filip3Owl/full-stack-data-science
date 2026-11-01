"""Generate the 8-bit project banner used at the top of the README.

    .venv/bin/python scripts/generate_banner.py

Writes ``assets/images/readme_banner.svg``. Every pixel is a rectangle on a coarse
grid, so the art stays sharp at any display size.
"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "assets/images/readme_banner.svg"

LINES = ("FULL-STACK", "DATA SCIENCE")
PIXEL = 7  # SVG units per art pixel
COLUMNS, ROWS = 116, 40

BACKGROUND = "#26282c"
FRAME = "#7a7e85"
FRAME_DARK = "#4a4d52"
SHADOW = "#6b6f76"
YELLOW = "#ffd60a"
YELLOW_LIGHT = "#fff27a"
YELLOW_DARK = "#d9a800"

# A chunky 6x7 pixel font, holding only the letters the banner needs.
FONT = {
    "A": ".####. ##..## ##..## ###### ##..## ##..## ##..##",
    "C": ".##### ##.... ##.... ##.... ##.... ##.... .#####",
    "D": "#####. ##..## ##..## ##..## ##..## ##..## #####.",
    "E": "###### ##.... ##.... #####. ##.... ##.... ######",
    "F": "###### ##.... ##.... #####. ##.... ##.... ##....",
    "I": "###### ..##.. ..##.. ..##.. ..##.. ..##.. ######",
    "K": "##..## ##.##. ####.. ###... ####.. ##.##. ##..##",
    "L": "##.... ##.... ##.... ##.... ##.... ##.... ######",
    "N": "##..## ###.## ###### ###### ##.### ##..## ##..##",
    "S": ".##### ##.... ##.... .####. ....## ....## #####.",
    "T": "###### ..##.. ..##.. ..##.. ..##.. ..##.. ..##..",
    "U": "##..## ##..## ##..## ##..## ##..## ##..## .####.",
    "-": "...... ...... ...... ###### ...... ...... ......",
    " ": "...... ...... ...... ...... ...... ...... ......",
}
GLYPH_WIDTH, GLYPH_HEIGHT, LETTER_GAP = 6, 7, 2
LINE_GAP = 4


def text_pixels(text: str, left: int, top: int) -> set[tuple[int, int]]:
    """Return the (column, row) of every lit pixel of `text`."""
    pixels = set()
    for index, letter in enumerate(text):
        offset = left + index * (GLYPH_WIDTH + LETTER_GAP)
        for row, line in enumerate(FONT[letter].split()):
            for column, cell in enumerate(line):
                if cell == "#":
                    pixels.add((offset + column, top + row))
    return pixels


def rectangles(pixels: set[tuple[int, int]], color: str) -> list[str]:
    """Merge each row's runs of pixels into one rectangle, to keep the file small."""
    shapes = []
    for row in sorted({y for _, y in pixels}):
        columns = sorted(x for x, y in pixels if y == row)
        start = previous = columns[0]
        for column in [*columns[1:], None]:
            if column == previous + 1:
                previous = column
                continue
            shapes.append(
                f'<rect x="{start * PIXEL}" y="{row * PIXEL}" '
                f'width="{(previous - start + 1) * PIXEL}" height="{PIXEL}" '
                f'fill="{color}"/>'
            )
            start = previous = column
    return shapes


def frame() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """A two-tone border with the stepped corners of old game menus."""
    outer, inner = set(), set()
    notch = 3
    for x in range(notch, COLUMNS - notch):
        outer |= {(x, 1), (x, ROWS - 2)}
    for y in range(notch, ROWS - notch):
        outer |= {(1, y), (COLUMNS - 2, y)}
    for step in range(notch):
        for x, y in (
            (notch - step, 1 + step),
            (COLUMNS - 1 - notch + step, 1 + step),
            (notch - step, ROWS - 2 - step),
            (COLUMNS - 1 - notch + step, ROWS - 2 - step),
        ):
            outer.add((x, y))
    for x in range(5, COLUMNS - 5):
        inner |= {(x, 3), (x, ROWS - 4)}
    for y in range(5, ROWS - 5):
        inner |= {(3, y), (COLUMNS - 4, y)}
    return outer, inner


def main() -> None:
    block_height = len(LINES) * GLYPH_HEIGHT + (len(LINES) - 1) * LINE_GAP
    bars_height = 5
    top = (ROWS - block_height - bars_height - 2) // 2

    lit: set[tuple[int, int]] = set()
    highlight: set[tuple[int, int]] = set()
    lowlight: set[tuple[int, int]] = set()
    for number, line in enumerate(LINES):
        width = len(line) * (GLYPH_WIDTH + LETTER_GAP) - LETTER_GAP
        left = (COLUMNS - width) // 2
        row = top + number * (GLYPH_HEIGHT + LINE_GAP)
        pixels = text_pixels(line, left, row)
        lit |= pixels
        # Top row of each letter is lighter, bottom row darker: the classic bevel.
        highlight |= {(x, y) for x, y in pixels if y == row}
        lowlight |= {(x, y) for x, y in pixels if y == row + GLYPH_HEIGHT - 1}

    shadow = {(x + 1, y + 1) for x, y in lit} - lit

    # A small row of pixel bars, for the data in the name.
    heights = [2, 4, 3, 5, 2, 3, 5, 4, 2, 3, 4, 5, 3, 2, 4, 3]
    bars_left = (COLUMNS - (len(heights) * 2 - 1)) // 2
    bars_bottom = ROWS - 7
    gray_bars, yellow_bars = set(), set()
    for index, height in enumerate(heights):
        target = yellow_bars if height == max(heights) else gray_bars
        for y in range(height):
            target.add((bars_left + index * 2, bars_bottom - y))

    outer, inner = frame()
    shapes = [
        f'<rect width="{COLUMNS * PIXEL}" height="{ROWS * PIXEL}" '
        f'fill="{BACKGROUND}"/>',
        *rectangles(outer, FRAME),
        *rectangles(inner, FRAME_DARK),
        *rectangles(gray_bars, SHADOW),
        *rectangles(yellow_bars, YELLOW),
        *rectangles(shadow, SHADOW),
        *rectangles(lit - highlight - lowlight, YELLOW),
        *rectangles(highlight, YELLOW_LIGHT),
        *rectangles(lowlight, YELLOW_DARK),
    ]
    width, height = COLUMNS * PIXEL, ROWS * PIXEL
    OUTPUT.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" shape-rendering="crispEdges" '
        'role="img" aria-label="Full-Stack Data Science">\n'
        "<title>Full-Stack Data Science</title>\n" + "\n".join(shapes) + "\n</svg>\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
