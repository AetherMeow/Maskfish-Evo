#!/usr/bin/env python3
"""EasyTier 内网传输服务：给笔记本（Omen）下载数据、上传网络用。

只绑定 EasyTier 接口（默认 10.20.0.1），不暴露到局域网/公网。

  GET  /<name>              从 --dir 读取并下载
  PUT  /upload/<name>       把请求体保存为 --upload 目录下的 <name>（先写 .part 再改名）
  GET  /list                列出两边目录里的文件及大小

用法：
  python3 transfer_server.py --dir /path/transfer --upload /path/transfer/in --bind 10.20.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % a))

    def _safe(self, root: Path, name: str) -> Path | None:
        name = unquote(name).lstrip("/")
        if not name or ".." in name.split("/"):
            return None
        p = (root / name)
        try:
            p.resolve().relative_to(root.resolve())
        except ValueError:
            return None
        return p

    # ---------------- GET ----------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/list":
            lines = []
            for label, root in (("data", Path(ARGS.dir)), ("upload", Path(ARGS.upload))):
                if root.is_dir():
                    for f in sorted(root.iterdir()):
                        if f.is_file():
                            lines.append(f"{label}\t{f.name}\t{f.stat().st_size}")
            body = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        f = self._safe(Path(ARGS.dir), path)
        if f is None or not f.is_file():
            self.send_error(404, "not found")
            return
        size = f.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(f, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ---------------- PUT ----------------
    def do_PUT(self):
        path = urlparse(self.path).path
        if not path.startswith("/upload/"):
            self.send_error(400, "PUT must be /upload/<name>")
            return
        target = self._safe(Path(ARGS.upload), path[len("/upload/"):])
        if target is None:
            self.send_error(400, "bad name")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        length = int(self.headers.get("Content-Length", "0"))
        tmp = target.with_suffix(target.suffix + ".part")
        got = 0
        with open(tmp, "wb") as fh:
            while got < length:
                chunk = self.rfile.read(min(1 << 20, length - got))
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
        if got != length:
            tmp.unlink(missing_ok=True)
            self.send_error(400, f"short body: {got}/{length}")
            return
        os.replace(tmp, target)
        body = f"saved {target.name} {got}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="下载目录")
    ap.add_argument("--upload", required=True, help="上传落地目录")
    ap.add_argument("--bind", default="10.20.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ARGS = ap.parse_args()
    Path(ARGS.dir).mkdir(parents=True, exist_ok=True)
    Path(ARGS.upload).mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((ARGS.bind, ARGS.port), Handler)
    print(f"transfer server on http://{ARGS.bind}:{ARGS.port}  dir={ARGS.dir}  upload={ARGS.upload}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
