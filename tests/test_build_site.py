"""Tests for the static site generator in scripts/build_site.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_site  # noqa: E402

MODULE_DIR = build_site.MODULES_DIR / "01-python-foundations"
PAGE = "modules/01-python-foundations/02_python_fundamentals.html"


@pytest.fixture(scope="module")
def links() -> build_site.LinkMap:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    ready = [
        module
        for part in build_site.parse_curriculum(readme)
        for module in part.modules
        if module.ready
    ]
    for module in ready:
        build_site.load_module(module)
    return build_site.LinkMap(ready)


def render(markdown: str, links: build_site.LinkMap, source_dir: Path) -> str:
    env = build_site.new_env(links, source_dir, PAGE, "Test")
    return build_site.MARKDOWN.render(markdown, env)


def test_curriculum_has_every_module_in_its_part() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    parts = build_site.parse_curriculum(readme)
    numbers = [module.number for part in parts for module in part.modules]
    assert numbers == [f"{n:02d}" for n in range(1, 25)]
    assert parts[0].modules[0].ready
    assert not parts[0].modules[2].ready


def test_callout_marker_becomes_a_title(links: build_site.LinkMap) -> None:
    html = render("> [!WARNING]\n> Silently wrong.", links, REPO_ROOT)
    assert 'class="callout callout-warning"' in html
    assert '<p class="callout-title">Warning</p>' in html
    assert "Silently wrong." in html
    assert "[!WARNING]" not in html


def test_plain_blockquote_is_not_a_callout(links: build_site.LinkMap) -> None:
    assert "callout" not in render("> Just a quote.", links, REPO_ROOT)


def test_notebook_link_points_at_its_page(links: build_site.LinkMap) -> None:
    html = render("[next](03_data_structures.ipynb)", links, MODULE_DIR / "notebooks")
    assert 'href="03_data_structures.html"' in html


def test_image_link_is_relative_to_the_page(links: build_site.LinkMap) -> None:
    html = render(
        "![x](../../../assets/images/legb_scope.png)", links, MODULE_DIR / "notebooks"
    )
    assert 'src="../../assets/images/legb_scope.png"' in html


def test_other_repo_file_links_to_github(links: build_site.LinkMap) -> None:
    html = render("[s](../../scripts/generate_figures.py)", links, MODULE_DIR)
    assert (
        f'href="{build_site.GITHUB_URL}/blob/main/scripts/generate_figures.py"' in html
    )


def test_broken_link_fails_the_build(links: build_site.LinkMap) -> None:
    with pytest.raises(build_site.BuildError, match="broken link"):
        render("[gone](missing.ipynb)", links, MODULE_DIR / "notebooks")


def test_repeated_headings_get_unique_ids(links: build_site.LinkMap) -> None:
    html = render("## Same\n\n## Same", links, REPO_ROOT)
    assert 'id="same"' in html
    assert 'id="same-1"' in html


def test_search_text_has_no_markup(links: build_site.LinkMap) -> None:
    env = build_site.new_env(links, REPO_ROOT, PAGE, "Test")
    build_site.MARKDOWN.render("- [ ] Can you explain *views*?", env)
    assert env["sections"][-1]["text"][0].strip() == "Can you explain views?"


def test_local_paths_are_removed_from_outputs() -> None:
    text = "root: /Users/someone/Desktop/CoursePythonStatistics/data/raw"
    assert (
        build_site.clean_output_text(text) == "root: ~/full-stack-data-science/data/raw"
    )


def test_full_build_has_valid_links_and_no_local_paths(tmp_path: Path) -> None:
    output = tmp_path / "site"
    stats = build_site.build(output)
    assert stats["modules"] >= 2
    assert (output / "index.html").exists()
    assert build_site.verify_links(output) == []
    for page in output.rglob("*.html"):
        assert not build_site.LOCAL_PATH_RE.search(
            page.read_text(encoding="utf-8")
        ), page
