"""Deterministic input parsing and private canonical artifact writing."""
import csv
import hashlib
import io
import json
from pathlib import Path

from .quality import SchemaError
from .schema import SCHEMA_VERSION, record


def strict_json_loads(text):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise SchemaError('Duplicate JSON object key')
            output[key] = value
        return output
    def nonfinite(token):
        raise SchemaError('Nonfinite JSON numeric literal')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def dumps(value):
    return json.dumps(record(value), sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_table(text, expected, delimiter=','):
    reader = csv.reader(io.StringIO(text)) if delimiter == ',' else csv.reader(io.StringIO(text), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SchemaError('Empty CSV input') from exc
    names = [name or f'_unnamed_{index}' for index, name in enumerate(header)]
    if len(names) != len(set(names)):
        raise SchemaError('Duplicate CSV headers')
    if set(names) != set(expected):
        added, removed = set(names) - set(expected), set(expected) - set(names)
        raise SchemaError(f'Unexpected schema: {len(added)} added, {len(removed)} removed fields')
    rows = []
    for lineno, values in enumerate(reader, 2):
        if len(values) != len(names):
            raise SchemaError(f'Ragged CSV at line {lineno}')
        rows.append((lineno, dict(zip(names, values))))
    return rows


class ArtifactWriter:
    """Write only into a caller-created private staging directory."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.handles = {}
        self.counts = {}

    def emit(self, table, value):
        if table not in self.handles:
            path = self.directory / f'{table}.jsonl'
            self.handles[table] = path.open('w', encoding='utf-8')
            path.chmod(0o600)
            self.counts[table] = 0
        self.handles[table].write(dumps(value) + '\n')
        self.counts[table] += 1

    def json(self, name, value):
        path = self.directory / f'{name}.json'
        path.write_text(dumps(value) + '\n', encoding='utf-8')
        path.chmod(0o600)

    def csv(self, name, header, rows):
        path = self.directory / f'{name}.csv'
        with path.open('w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(rows)
        path.chmod(0o600)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def ensure_tables(self, tables):
        for table in tables:
            if table not in self.counts:
                path = self.directory / f'{table}.jsonl'
                path.touch(mode=0o600)
                self.counts[table] = 0

    def manifest(self, inputs, seed, config, source_code_hash):
        files = {p.name: {'sha256': file_hash(p), 'bytes': p.stat().st_size}
                 for p in sorted(self.directory.iterdir()) if p.is_file() and p.name != 'manifest.json'}
        self.json('manifest', {'schema_version': SCHEMA_VERSION, 'inputs': inputs,
                             'seed': seed, 'config': config,
                             'source_code_sha256': source_code_hash, 'artifacts': files})


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            yield strict_json_loads(line)
