"""Minimal stdlib HTTP server for Railway health checks and live metrics.

No web framework dependency: a background thread serves ``/health`` (always 200
once the process is up) and ``/`` or ``/metrics`` (the latest loop status as
JSON).  Railway pings ``/health`` to know the service is alive.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

log = logging.getLogger(__name__)


def start_health_server(port: int, status_fn: Callable[[], dict]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (stdlib naming)
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok"})
            else:
                self._send(200, status_fn())

        def log_message(self, *args):  # silence default access logging
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("health server listening on :%d", port)
    return server
