"""Optional browser regression test. Install playwright and Google Chrome, then run
python -m unittest runtime_mlx.test_chat_browser.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Queue
from threading import Event, Thread
import unittest

try:
    from playwright.sync_api import Error, expect, sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "requires Playwright and Google Chrome")
class ChatBrowserTest(unittest.TestCase):
    def test_new_chat_discards_pending_turn(self):
        page_html = Path(__file__).with_name("chat.html").read_bytes()
        release_old = Event()
        requests = Queue()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, kind, body):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply("application/json", b'{"data":[{"id":"fixture","context_length":1024}]}')
                else:
                    self.reply("text/html; charset=utf-8", page_html)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = body["messages"]
                requests.put(messages)
                is_old = messages[-1]["content"] == "old question"
                content = "Old partial" if is_old else "New answer"
                frame = json.dumps({"choices": [{"delta": {"content": content}}]})
                self.reply("text/event-stream", f"data: {frame}\n\n".encode())
                self.wfile.flush()
                if is_old:
                    release_old.wait(10)
                else:
                    self.wfile.write(b"data: [DONE]\n\n")

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with sync_playwright() as playwright:
                    try:
                        browser = playwright.chromium.launch(
                            channel="chrome",
                            args=["--disable-features=MacAppCodeSignClone",
                                  "--disable-component-update", "--disable-default-apps"],
                        )
                    except Error as error:
                        if "distribution 'chrome' is not found" in str(error):
                            self.skipTest("requires Google Chrome")
                        raise
                    try:
                        page = browser.new_page()
                        page.goto(f"http://127.0.0.1:{server.server_port}/")
                        message = page.locator("#input")
                        send = page.get_by_role("button", name="Send")
                        message.fill("old question")
                        send.click()
                        expect(page.get_by_text("Old partial", exact=True)).to_be_visible()
                        page.get_by_role("button", name="New chat").click()
                        expect(send).to_be_enabled()
                        expect(page.get_by_text("Old partial", exact=True)).to_have_count(0)
                        message.fill("new question")
                        send.click()
                        expect(page.get_by_text("New answer", exact=True)).to_be_visible()
                        self.assertEqual(requests.get_nowait(), [{"role": "user", "content": "old question"}])
                        self.assertEqual(requests.get_nowait(), [{"role": "user", "content": "new question"}])
                    finally:
                        release_old.set()
                        browser.close()
            finally:
                release_old.set()
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
