"""NFL dashboard: python app.py  (opens http://127.0.0.1:8030 in the browser)

Serves site/ and two endpoints: POST /api/refresh reruns pipeline.py in the background,
GET /api/status says whether a refresh is running and how it ended. Data older than 6 hours is
refreshed automatically when the app starts. Listens on this PC only."""
import json, os, subprocess, sys, threading, time, webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, 'site')
PORT = int(os.getenv('NFL_APP_PORT', '8030'))
state: dict = dict(running=False, started=None, finished=None, ok=None, message='')
lock = threading.Lock()


def refresh():
    with lock:
        if state['running']:
            return False
        state.update(running=True, started=time.time(), message='Refreshing data...')

    def work():
        py = sys.executable.replace('pythonw.exe', 'python.exe')
        try:
            r = subprocess.run([py, 'pipeline.py'], cwd=HERE, capture_output=True, text=True, timeout=1800,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            ok = r.returncode == 0
            msg = (r.stdout.strip().splitlines() or [''])[-1] if ok else (r.stderr.strip().splitlines() or ['failed'])[-1]
        except Exception as e:
            ok, msg = False, f'{type(e).__name__}: {e}'
        with lock:
            state.update(running=False, finished=time.time(), ok=ok, message=msg)
    threading.Thread(target=work, daemon=True).start()
    return True


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SITE, **k)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status':
            with lock:
                return self._json(dict(state))
        if self.path.startswith('/data.json'):
            self.path = '/data.json'
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/refresh':
            return self._json(dict(started=refresh()))
        self.send_error(404)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, format, *args):
        pass


def main():
    data = os.path.join(SITE, 'data.json')
    if not os.path.exists(data) or time.time() - os.path.getmtime(data) > 6 * 3600:
        refresh()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    url = f'http://127.0.0.1:{PORT}/'
    print(f'NFL dashboard running at {url}  (close this window to stop it)')
    if '--no-browser' not in sys.argv:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == '__main__':
    main()
