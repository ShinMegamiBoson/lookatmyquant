"""Optional browser regressions. Install Playwright and Google Chrome, then run
python -m unittest runtime_mlx.test_chat_browser.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import sleep
import unittest

try:
    from playwright.sync_api import Error, expect, sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "requires Playwright and Google Chrome")
class ChatBrowserTest(unittest.TestCase):
    def setUp(self):
        page_html = Path(__file__).with_name("chat.html").read_bytes()
        self.release_old = Event()
        self.requests = Queue()
        self.required_key = None
        self.model_id = "fixture"
        self.models_unavailable = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, kind, body):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.end_headers()
                self.wfile.write(body)

            def reply_json(self, status, data):
                self.reply(status, "application/json", json.dumps(data).encode())

            def allowed(self):
                if fixture.required_key is None or self.headers.get("Authorization") == f"Bearer {fixture.required_key}":
                    return True
                self.reply_json(401, {"error": {"message": "unauthorized"}})
                return False

            def do_GET(self):
                if self.path == "/v1/models":
                    if self.allowed():
                        if fixture.models_unavailable:
                            self.reply_json(503, {"error": {"message": "model unavailable"}})
                        else:
                            self.reply_json(200, {"data": [{"id": fixture.model_id, "context_length": 1024}]})
                else:
                    self.reply(200, "text/html; charset=utf-8", page_html)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.put(body)
                if not self.allowed():
                    return
                if body.get("model") != fixture.model_id:
                    self.reply_json(409, {"error": {"type": "model_changed"}})
                    return
                prompt = body["messages"][-1]["content"]
                if prompt == "too long":
                    self.reply_json(400, {"error": {
                        "message": "the request exceeds the available context size, try increasing it (prompt 1200 tokens, context 1024)",
                        "type": "exceed_context_size_error",
                    }})
                    return
                if prompt == "server error":
                    self.reply_json(500, {"error": {"message": "server error"}})
                    return
                if prompt == "overloaded":
                    self.reply_json(503, {"error": {"message": "model is busy", "type": "server_overloaded"}})
                    return
                content = {"old question": "Old partial", "limited question": "Partial "}.get(prompt, "New answer")
                if prompt == "long question":
                    content = "\n".join(f"Line {number}" for number in range(1, 121))
                frame = json.dumps({"choices": [{"delta": {"content": content}}]})
                self.reply(200, "text/event-stream", f"data: {frame}\n\n".encode())
                self.wfile.flush()
                if prompt in {"old question", "long question"}:
                    fixture.release_old.wait(10)
                reason = "length" if prompt == "limited question" else "stop"
                finish = json.dumps({"choices": [{"finish_reason": reason}]})
                usage = json.dumps({"choices": [], "usage": {"completion_tokens": 8, "prompt_tokens": 21}})
                try:
                    if prompt == "limited question":
                        sleep(0.18)
                        frame = json.dumps({"choices": [{"delta": {"content": "answer"}}]})
                        self.wfile.write(f"data: {frame}\n\n".encode())
                    self.wfile.write(f"data: {finish}\n\ndata: {usage}\n\ndata: [DONE]\n\n".encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.url = f"http://127.0.0.1:{server.server_port}/"
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join)
        self.addCleanup(server.shutdown)
        self.addCleanup(self.release_old.set)

        playwright = sync_playwright().start()
        self.addCleanup(playwright.stop)
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
        self.addCleanup(browser.close)
        self.page = browser.new_page()
        self.message = self.page.get_by_role("textbox", name="Message")
        self.send = self.page.get_by_role("button", name="Send", exact=True)
        self.stop = self.page.get_by_role("button", name="Stop", exact=True)
        self.replies = self.page.locator(".msg.assistant .body")

    def ready(self):
        expect(self.page.locator("#status")).to_have_text(self.model_id)

    def ask(self, text):
        self.message.fill(text)
        self.send.click()

    def enter_key(self, text):
        key = self.page.get_by_label("API key")
        key.fill(text)
        key.press("Enter")
        self.ready()

    def test_new_chat_discards_pending_turn(self):
        self.page.goto(self.url)
        self.ready()
        expect(self.send).to_be_disabled()
        self.ask("old question")
        expect(self.replies).to_have_text(["Old partial"])
        self.page.get_by_role("button", name="New chat").click()
        expect(self.send).to_be_disabled()
        expect(self.replies).to_have_count(0)
        self.ask("new question")
        expect(self.replies).to_have_text(["New answer"])
        self.release_old.set()
        expect(self.stop).to_be_hidden()
        self.assertEqual(self.requests.get(timeout=2)["messages"], [{"role": "user", "content": "old question"}])
        self.assertEqual(self.requests.get(timeout=2)["messages"], [{"role": "user", "content": "new question"}])

    def test_limits_keep_draft_and_only_finished_replies_enter_history(self):
        self.page.goto(self.url)
        self.ready()
        self.page.get_by_text("Settings", exact=True).click()
        budget = self.page.get_by_role("spinbutton", name="Max tokens")
        budget.fill("0")
        self.ask("limited question")
        expect(self.page.locator("#budgetError")).to_contain_text("positive whole number")
        expect(self.message).to_have_value("limited question")
        self.assertTrue(self.requests.empty())

        budget.fill("8")
        self.message.press("Enter")
        expect(self.page.locator(".notice").filter(has_text="Partial reply")).to_be_visible()
        expect(self.message).to_have_value("limited question")
        self.assertEqual(self.requests.get(timeout=2)["max_tokens"], 8)
        self.page.get_by_text("Usage", exact=True).click()
        usage = self.page.locator(".stats-value")
        expect(usage).to_contain_text("8 output tokens")
        expect(usage).to_contain_text("tok/s observed")
        self.page.get_by_role("button", name="Adjust generation settings").click()
        expect(budget).to_be_focused()
        budget.fill("")
        self.page.get_by_text("Settings", exact=True).click()

        self.ask("new question")
        expect(self.replies).to_have_text(["Partial answer", "New answer"])
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.assertEqual(self.requests.get(timeout=2)["messages"], [{"role": "user", "content": "new question"}])
        self.page.get_by_role("button", name="Latest reply").click()
        expect(self.page.locator(".msg.assistant").last).to_be_focused()

        self.ask("too long")
        expect(self.page.locator("#sendError")).to_contain_text("Shorten your message.")
        expect(self.message).to_have_value("too long")
        expect(self.replies).to_have_text(["Partial answer", "New answer"])
        self.assertEqual(self.requests.get(timeout=2)["messages"], [
            {"role": "user", "content": "new question"}, {"role": "assistant", "content": "New answer"},
            {"role": "user", "content": "too long"},
        ])
        self.page.get_by_role("button", name="Start fresh with draft").click()
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("too long")
        self.page.get_by_role("button", name="Undo").click()
        expect(self.replies).to_have_text(["Partial answer", "New answer"])
        expect(self.message).to_have_value("too long")

        self.ask("server error")
        expect(self.page.locator("#sendError")).to_contain_text("HTTP 500")
        expect(self.message).to_have_value("server error")
        expect(self.replies).to_have_text(["Partial answer", "New answer"])

        self.ask("overloaded")
        expect(self.page.locator("#sendError")).to_have_text("The server is busy. Wait a moment, then send again.")
        expect(self.message).to_have_value("overloaded")
        expect(self.replies).to_have_text(["Partial answer", "New answer"])

    def test_reload_restores_finished_chat_and_draft_and_interruption(self):
        self.page.goto(self.url)
        self.ready()
        self.ask("finished question")
        expect(self.stop).to_be_hidden()
        expect(self.replies).to_have_text(["New answer"])
        self.requests.get(timeout=2)
        self.message.fill("  unsent draft  ")
        self.page.reload()
        self.ready()
        expect(self.replies).to_have_text(["New answer"])
        expect(self.message).to_have_value("  unsent draft  ")
        self.page.get_by_role("button", name="New chat").click()
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("")
        self.page.get_by_role("button", name="Undo").click()
        expect(self.replies).to_have_text(["New answer"])
        expect(self.message).to_have_value("  unsent draft  ")

        self.ask("old question")
        expect(self.replies).to_have_text(["New answer", "Old partial"])
        self.requests.get(timeout=2)
        self.page.reload()
        self.ready()
        expect(self.message).to_have_value("old question")
        expect(self.page.get_by_text("Page reloaded before the reply finished.", exact=False)).to_be_visible()
        self.ask("new question")
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.assertEqual(self.requests.get(timeout=2)["messages"], [
            {"role": "user", "content": "finished question"}, {"role": "assistant", "content": "New answer"},
            {"role": "user", "content": "new question"},
        ])

    def test_authenticated_cache_waits_for_matching_model_and_key(self):
        self.required_key = "fixture-initial"
        self.page.goto(self.url)
        expect(self.page.get_by_label("API key")).to_be_visible()
        expect(self.page.get_by_role("button", name="New chat")).to_be_disabled()
        self.message.fill("draft before connecting")
        self.enter_key(self.required_key)
        expect(self.message).to_have_value("draft before connecting")
        self.ask("finished question")
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.message.fill("current draft")
        self.page.reload()
        self.ready()
        expect(self.replies).to_have_text(["New answer"])
        expect(self.message).to_have_value("current draft")

        gate = self.page.locator("#gateMessage")
        self.required_key = "fixture-temporary"
        self.send.click()
        expect(gate).to_contain_text("API key rejected.")
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("")
        self.required_key = "fixture-initial"
        self.enter_key(self.required_key)
        expect(self.message).to_have_value("current draft")
        expect(self.replies).to_have_text(["New answer"])

        self.required_key = "fixture-rotated"
        self.send.click()
        expect(gate).to_contain_text("API key rejected.")
        self.enter_key(self.required_key)
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("")

        self.ask("finished question")
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.message.fill("draft on first model")
        self.required_key = "fixture-temporary"
        self.page.reload()
        expect(gate).to_contain_text("API key rejected.")
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("")
        self.required_key = "fixture-rotated"
        self.enter_key(self.required_key)
        expect(self.replies).to_have_text(["New answer"])
        expect(self.message).to_have_value("draft on first model")

        self.required_key = "fixture-temporary"
        self.send.click()
        expect(gate).to_contain_text("API key rejected.")
        self.required_key = "fixture-rotated"
        self.model_id = "replacement-model"
        self.enter_key(self.required_key)
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("")

    def test_reconnect_after_model_restart_keeps_only_the_draft_with_the_same_key(self):
        self.required_key = "fixture-initial"
        self.page.goto(self.url)
        self.enter_key(self.required_key)
        self.ask("finished question")
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.assertEqual(self.requests.get(timeout=2)["model"], "fixture")

        self.model_id = "fixture-replacement-model"
        self.ask("draft for the new model")
        gate = self.page.locator("#gateMessage")
        expect(gate).to_contain_text("Reconnect to start a fresh chat with your draft.")
        expect(self.replies).to_have_text(["New answer"])
        expect(self.message).to_have_value("draft for the new model")
        self.assertEqual(self.requests.get(timeout=2)["model"], "fixture")
        self.page.get_by_role("button", name="Reconnect").click()
        self.ready()
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("draft for the new model")
        expect(self.page.locator("#status")).to_have_attribute("title", self.model_id)
        self.page.get_by_text("Settings", exact=True).click()
        expect(self.page.locator("#contextInfo")).to_have_text(f"{self.model_id} · 1,024 context tokens")
        self.page.get_by_text("Settings", exact=True).click()

        self.send.click()
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        request = self.requests.get(timeout=2)
        self.assertEqual(request["model"], self.model_id)
        self.assertEqual(request["messages"], [{"role": "user", "content": "draft for the new model"}])

        self.model_id = "another-model"
        self.ask("draft bound to the old key")
        expect(gate).to_contain_text("Reconnect to start a fresh chat")
        self.required_key = "fixture-rotated"
        self.page.get_by_role("button", name="Reconnect").click()
        expect(gate).to_contain_text("API key rejected.")
        self.enter_key(self.required_key)
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("")

    def test_keyboard_reader_can_scroll_first_streaming_reply(self):
        self.page.set_viewport_size({"width": 930, "height": 380})
        self.page.goto(self.url)
        self.ready()
        self.ask("long question")
        expect(self.replies).to_contain_text("Line 120")
        log = self.page.locator("#log")
        self.page.wait_for_function("document.querySelector('#log').scrollTop > 500")
        self.message.press("Shift+Tab")
        expect(log).to_be_focused()
        scroll_top = log.evaluate("(node) => node.scrollTop")
        self.page.keyboard.press("PageUp")
        self.page.wait_for_function("(previous) => document.querySelector('#log').scrollTop < previous", arg=scroll_top)
        expect(self.stop).to_be_visible()

    def test_reload_after_discovery_failure_restores_a_sendable_draft(self):
        self.models_unavailable = True
        self.page.goto(self.url)
        gate = self.page.locator("#gateMessage")
        expect(gate).to_contain_text("Could not connect")
        self.message.fill("written offline")
        self.page.reload()
        expect(gate).to_contain_text("Could not connect")

        self.models_unavailable = False
        self.page.get_by_role("button", name="Reconnect").click()
        self.ready()
        expect(self.message).to_have_value("written offline")
        expect(self.send).to_be_enabled()
        self.send.click()
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.assertEqual(self.requests.get(timeout=2)["messages"], [{"role": "user", "content": "written offline"}])

    def test_reload_after_model_change_only_restores_the_draft_for_the_same_key(self):
        self.required_key = "fixture-initial"
        self.page.goto(self.url)
        self.enter_key(self.required_key)
        self.ask("finished question")
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        self.requests.get(timeout=2)

        self.model_id = "fixture-replacement-model"
        self.ask("draft for the new model")
        gate = self.page.locator("#gateMessage")
        expect(gate).to_contain_text("The server switched models.")
        self.requests.get(timeout=2)
        self.page.reload()
        self.ready()
        expect(self.replies).to_have_count(0)
        expect(self.message).to_have_value("draft for the new model")
        self.send.click()
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        request = self.requests.get(timeout=2)
        self.assertEqual(request["model"], self.model_id)
        self.assertEqual(request["messages"], [{"role": "user", "content": "draft for the new model"}])

        self.model_id = "third-model"
        self.ask("draft for the old key")
        expect(gate).to_contain_text("The server switched models.")
        self.required_key = "fixture-rotated"
        self.page.reload()
        expect(gate).to_contain_text("API key rejected.")
        expect(self.message).to_have_value("")
        expect(self.replies).to_have_count(0)
        self.enter_key(self.required_key)
        expect(self.message).to_have_value("")
        expect(self.replies).to_have_count(0)
        expect(self.send).to_be_disabled()

    def test_storage_failure_only_warns_when_there_is_unsaved_work(self):
        self.page.add_init_script("""(() => {
            const original = Storage.prototype.setItem;
            Storage.prototype.setItem = function (name, value) {
                if (this === sessionStorage) throw new DOMException("Storage full", "QuotaExceededError");
                return original.call(this, name, value);
            };
        })()""")
        self.page.goto(self.url)
        self.ready()
        is_dirty = "() => !window.dispatchEvent(new Event('beforeunload', { cancelable: true }))"
        self.assertFalse(self.page.evaluate(is_dirty))
        self.message.fill("unsaved draft")
        self.assertTrue(self.page.evaluate(is_dirty))
        self.message.fill("")
        self.assertFalse(self.page.evaluate(is_dirty))
        self.ask("finished question")
        expect(self.page.locator("#announcer")).to_have_text("Reply ready.")
        expect(self.replies).to_have_text(["New answer"])
        self.assertTrue(self.page.evaluate(is_dirty))


if __name__ == "__main__":
    unittest.main()
