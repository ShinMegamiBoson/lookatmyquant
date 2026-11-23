"""Make paper section lists a native disclosure on phones. Each generated paper works offline."""

import re
from pathlib import Path

COMPACT = "(max-width: 48rem), (max-height: 32rem) and (pointer: coarse)"
STYLE = """<style id="paper-sections-style">
.paper-sections { width: 100%; min-width: 0; }
.paper-sections > summary {
  width: fit-content; margin: 0 auto; min-height: 44px; padding: 10px 4px;
  color: var(--ink); cursor: pointer; font: 600 1rem/1.5 var(--sans);
  text-align: left;
}
.paper-sections > summary::marker { color: var(--muted); }
.paper-sections > summary:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
@media COMPACT {
  .wrap:has(> header.title .paper-sections) { padding-top: 20px; }
  header.title:has(.paper-sections) {
    display: flex; flex-wrap: wrap; justify-content: center; align-items: baseline;
    column-gap: 0.6rem; row-gap: 8px; margin-bottom: 16px;
  }
  header.title:has(.paper-sections) > :is(h1, .subtitle) { flex-basis: 100%; }
  header.title:has(.paper-sections) > :is(.authors, .meta) {
    margin: 0; font: 400 .9rem/1.5 var(--sans); color: var(--muted);
  }
  header.title:has(.paper-sections) > .meta::before { content: "·"; margin-right: .6rem; }
  .paper-sections { border-block: 1px solid var(--line); margin-top: 3px; }
  .paper-sections .links { display: grid; gap: 0; margin: 0; text-align: left; }
  .paper-sections .links a {
    display: block; min-height: 44px; padding: 10px 8px;
    color: var(--ink); background: transparent; border-radius: 0;
    border-top: 1px solid var(--line); box-shadow: none; font: 500 1rem/1.45 var(--sans);
    overflow-wrap: anywhere;
  }
  .paper-sections .links a:focus-visible { outline-offset: -2px; }
}
</style>""".replace("COMPACT", COMPACT)

SCRIPT = """<script id="paper-sections-behavior">
(() => {
  const menu = document.querySelector('details.paper-sections');
  const summary = menu.querySelector('summary');
  const compact = matchMedia('COMPACT');
  function setLayout() {
    menu.open = !compact.matches;
    summary.hidden = !compact.matches;
  }
  setLayout();
  compact.addEventListener('change', setLayout);
  menu.addEventListener('click', event => {
    if (compact.matches && event.target.closest('a[href]')) menu.open = false;
  });
})();
</script>""".replace("COMPACT", COMPACT)

NAV = re.compile(r'<nav\b[^>]*\baria-label="Sections"[^>]*>.*?</nav>', re.DOTALL)


def with_paper_sections(page: str) -> str:
    """Use the existing links once. Update the shared styles and behavior on repeat builds."""
    nav = NAV.search(page)
    if nav is None:
        return page
    if '<details class="paper-sections">' not in page:
        wrapped = (
            '<details class="paper-sections">\n      <summary>On this page</summary>\n      '
            + nav[0]
            + "\n    </details>"
        )
        page = page[: nav.start()] + wrapped + page[nav.end() :]
    for tag, block, boundary in (
        ("style", STYLE, "</head>"),
        ("script", SCRIPT, "</body>"),
    ):
        marker = block.split(">", 1)[0] + ">"
        if marker in page:
            start = page.index(marker)
            end = page.index(f"</{tag}>", start) + len(f"</{tag}>")
            page = page[:start] + block + page[end:]
        else:
            page = page.replace(boundary, block + "\n" + boundary, 1)
    return page


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        path = Path(arg)
        old = path.read_text()
        new = with_paper_sections(old)
        if old != new:
            path.write_text(new)
