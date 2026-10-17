"""Small reproducible PDF writer for authored synthetic plain-text records (no extraction)."""
import textwrap
from pathlib import Path


def write_pdf(path, title, content):
    lines = []
    for line in content.splitlines():
        lines.extend(textwrap.wrap(line, width=90, break_long_words=False) or [''])
    pages = [lines[i:i+52] for i in range(0, len(lines), 52)] or [[]]
    objects = [b'', b'', b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    page_ids = []
    def escape(text):
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)').encode('latin-1', 'strict')
    for index, lines in enumerate(pages):
        page_id = len(objects) + 1
        stream_id = page_id + 1
        page_ids.append(page_id)
        stream = b'BT /F1 13 Tf 48 790 Td (' + escape(title) + b') Tj ET\n'
        stream += b'BT /F1 9 Tf 48 763 Td 12 TL\n'
        for line in lines:
            stream += b'(' + escape(line) + b') Tj T*\n'
        stream += b'ET\nBT /F1 7 Tf 48 32 Td (SYNTHETIC TEST RECORD - fictional clinical record - page ' + str(index+1).encode() + b') Tj ET\n'
        objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>'.encode())
        objects.append(b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'endstream')
    objects[0] = b'<< /Type /Catalog /Pages 2 0 R >>'
    objects[1] = b'<< /Type /Pages /Kids [' + b' '.join(f'{p} 0 R'.encode() for p in page_ids) + b'] /Count ' + str(len(pages)).encode() + b' >>'
    result = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f'{index} 0 obj\n'.encode() + obj + b'\nendobj\n')
    xref = len(result)
    result.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        result.extend(f'{offset:010d} 00000 n \n'.encode())
    result.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    Path(path).write_bytes(result)
    Path(path).chmod(0o600)
