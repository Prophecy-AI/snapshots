import http.server
import json
import os
import socketserver

PORT = 8000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'index.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        if self.path == '/app.js':
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        if self.path == '/api/files':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
            self.wfile.write(json.dumps(sorted(files)).encode())
            return
        if self.path.startswith('/api/data/'):
            file_name = self.path.split('/')[-1]
            file_path = os.path.join(DATA_PATH, file_name)
            if os.path.exists(file_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                trees = []
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.strip().split(',')
                        trees.append({'id': i, 'x': float(parts[0]), 'y': float(parts[1]), 'angle': float(parts[2])})
                self.wfile.write(json.dumps(trees).encode())
            else:
                self.send_response(404)
                self.end_headers()
            return
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/api/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            file_name = data['filename']
            trees = data['trees']

            # Basic security check
            if '..' in file_name or not file_name.endswith('.txt'):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Invalid filename'}).encode())
                return

            file_path = os.path.join(DATA_PATH, file_name)

            try:
                with open(file_path, 'w') as f:
                    for tree in trees:
                        f.write(f'{tree["x"]},{tree["y"]},{tree["angle"]}\n')

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())
            return

        self.send_response(404)
        self.end_headers()


Handler = MyHttpRequestHandler

with socketserver.TCPServer(('', PORT), Handler) as httpd:
    print(f'Go to http://localhost:{PORT}')
    httpd.serve_forever()
