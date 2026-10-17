"""Separate loopback P1 product API; existing dashboard services remain untouched."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sqlite3
from urllib.parse import unquote, urlsplit
from .repository import Repository, DEFAULT_DB, NotFound
from .models import ENTITY_TYPES


def handler(repository):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # No patient IDs, record bodies or request paths in logs.

        def reply(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('JSON required')
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > 16384:
                raise ValueError('Invalid body size')
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError('Duplicate field')
                    result[key] = value
                return result
            def invalid(_):
                raise ValueError('Non-finite JSON')
            value = json.loads(self.rfile.read(size), object_pairs_hook=pairs, parse_constant=invalid)
            if not isinstance(value, dict):
                raise ValueError('Object required')
            return value

        def dispatch(self):
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.reply(403, {'error': 'Origin denied'})
            parts = [unquote(p) for p in urlsplit(self.path).path.strip('/').split('/')]
            if parts == ['api', 'product', 'health'] and self.command == 'GET':
                return self.reply(200, {'status': 'ok', 'schema_version': 1, 'inference_connected': False})
            if parts[:2] == ['api', 'product']:
                parts = parts[2:]
            if not parts or parts[0] != 'patients':
                return self.reply(404, {'error': 'Route not found'})
            if len(parts) == 1:
                if self.command == 'GET':
                    return self.reply(200, {'patients': repository.list_patients()})
                if self.command == 'POST':
                    value = self.body()
                    if set(value) - {'display_name', 'date_of_birth', 'sex', 'status', 'synthetic'}:
                        raise ValueError('Unsupported field')
                    if 'display_name' not in value:
                        raise ValueError('Missing display name')
                    return self.reply(201, repository.create_patient(**value))
            elif len(parts) == 2:
                if self.command == 'GET':
                    return self.reply(200, repository.get_patient(parts[1]))
                if self.command == 'PATCH':
                    return self.reply(200, repository.update_patient(parts[1], **self.body()))
            elif self.command == 'GET':
                if len(parts) == 3 and parts[2] == 'clinical-profile':
                    return self.reply(200, repository.get_profile(parts[1]))
                if parts[2] in ENTITY_TYPES:
                    if len(parts) == 3:
                        return self.reply(200, {'patient_id': parts[1], 'items': repository.list_owned(parts[1], parts[2])})
                    if len(parts) == 4:
                        return self.reply(200, repository.get_owned(parts[1], parts[2], parts[3]))
            self.reply(405, {'error': 'Method not supported'})

        def run_request(self):
            try:
                self.dispatch()
            except NotFound:
                self.reply(404, {'error': 'Record not found'})
            except (ValueError, TypeError, sqlite3.IntegrityError, UnicodeError):
                self.reply(400, {'error': 'Invalid request'})
            except Exception:
                self.reply(500, {'error': 'Service error'})

        do_GET = run_request
        do_POST = run_request
        do_PATCH = run_request
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--db', default=str(DEFAULT_DB))
    args = parser.parse_args()
    repository = Repository(args.db)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(repository))
    print(f'Patient domain API listening on http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
