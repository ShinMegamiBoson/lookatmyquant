# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Run with uv run tests/test_paper_navigation.py after installing Playwright WebKit."""

import re
import unittest
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"


def papers_with_section_links():
    for page in sorted(SITE.rglob("index.html")):
        text = page.read_text(encoding="utf-8")
        for nav in re.findall(r"<nav\b[^>]*>.*?</nav>", text, re.DOTALL):
            if len(re.findall(r'<a\b[^>]*href="#', nav)) >= 3:
                yield page
                break


class PaperNavigationTests(unittest.TestCase):
    def test_section_menus_are_collapsed_in_the_html(self):
        papers = list(papers_with_section_links())
        self.assertTrue(papers)
        for page in papers:
            with self.subTest(page=page.parent.name):
                self.assertRegex(
                    page.read_text(encoding="utf-8"),
                    r'<details class="paper-sections">\s*<summary>On this page</summary>\s*<nav',
                )

    def test_mobile_menu_and_first_rsi_figure_in_webkit(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("Run this file with uv for the WebKit layout test.")
        with sync_playwright() as runner, runner.webkit.launch() as browser:
            for width, height in ((320, 568), (390, 844), (430, 932), (932, 430)):
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    is_mobile=True,
                    has_touch=True,
                )
                try:
                    page = context.new_page()
                    for file in papers_with_section_links():
                        with self.subTest(paper=file.parent.name, width=width):
                            page.goto(file.as_uri(), wait_until="domcontentloaded")
                            menu = page.locator("details.paper-sections")
                            control = menu.locator("summary")
                            self.assertTrue(control.is_visible())
                            self.assertFalse(menu.evaluate("el => el.open"))
                            self.assertLessEqual(menu.bounding_box()["height"], 56)
                            if file.parent.name == "self-editing" and width < height:
                                figure = page.locator("#fig1 img").bounding_box()
                                self.assertLess(
                                    figure["y"] + min(figure["height"], 100), height
                                )
                            links = menu.locator('a[href^="#"]')
                            for i in range(links.count()):
                                control.tap()
                                target = links.nth(i).get_attribute("href")
                                self.assertGreaterEqual(
                                    links.nth(i).bounding_box()["height"], 44
                                )
                                links.nth(i).tap()
                                self.assertFalse(menu.evaluate("el => el.open"))
                                self.assertEqual(page.evaluate("location.hash"), target)
                                self.assertTrue(page.locator(target).is_visible())
                finally:
                    context.close()
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(
                (SITE / "self-editing/index.html").as_uri(),
                wait_until="domcontentloaded",
            )
            self.assertFalse(
                page.locator("details.paper-sections summary").is_visible()
            )
            self.assertTrue(
                page.locator("nav[aria-label=Sections] a").first.is_visible()
            )


if __name__ == "__main__":
    unittest.main()
