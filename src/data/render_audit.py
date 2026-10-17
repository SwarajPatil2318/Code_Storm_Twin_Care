"""Render aggregate field/code inventories; never emit row values or patient IDs."""
import json
from pathlib import Path

MARKER = '<!-- GENERATED -->'


def escape(value):
    return str(value).replace('|', r'\|').replace('\n', ' ')


def emit_table(lines, header, rows):
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    lines.extend('| ' + ' | '.join(escape(v) for v in row) + ' |' for row in rows)
    lines.append('')


def field_rows(fields, dictionary=None):
    for key, field in fields.items():
        values = [key, field['missing'], f"{field['missing_pct']:.2f}%", field['distinct']]
        if key in {'inpatient.number', 'Patient_ID', 'user_id'} or key.startswith('_unnamed_'):
            values.append('identifier/export index range omitted')
        else:
            values.append(' / '.join(f"{field[k]:.6g}" for k in ['min', 'median', 'max']) if field['numeric'] else 'not numeric / all missing')
        if dictionary is not None:
            values.append(dictionary.get(key, 'Export index; exclude'))
        yield values


def render(report):
    lines = ['','### Input manifest','']
    emit_table(lines, ['File', 'Bytes', 'SHA-256'], ([x['file'], x['bytes'], x['sha256']] for x in report['inventory']))
    for name, title in [('zigong','Zigong clinical'),('chiron','CHIRON')]:
        lines += [f'### {title} field profiles', '', 'Range column is min / median / max of nonmissing numeric cells. Distinct counts include original lexical forms. Export-index ranges are mechanical, not patient physiology.', '']
        a=report[name]; dictionary = a['dictionary'] if name=='zigong' else None
        header=['Field','Missing','Missing %','Distinct','Numeric range']
        if dictionary is not None: header.append('Local dictionary description / unit')
        emit_table(lines,header,field_rows(a['fields'],dictionary))
    lines += ['### Zigong medication vocabulary', '', 'Names are exposure categories, without dose, administration time or adherence.', '']
    emit_table(lines, ['Source drug name'], ([name] for name in report['zigong']['drug_names']))
    lines += ['### PerHeart questionnaire field profiles', '', 'Free-text values and individual answers are intentionally omitted.', '']
    emit_table(lines,['Field','Missing','Missing %','Distinct','Numeric range'],field_rows(report['perheart']['questionnaires']['fields']))
    for name,a in report['perheart']['medical'].items():
        lines += [f'### PerHeart {name} field profiles','']
        emit_table(lines,['Field','Missing','Missing %','Distinct','Numeric range'],field_rows(a['fields']))
    a=report['synthea'];lines += ['### Synthea resource inventory','']
    emit_table(lines,['Resource type','Count'],sorted(a['resource_counts'].items()))
    lines += ['### Synthea quantity units','', 'Counts include top-level and component quantities. Three quantities have empty unit/system/code; missing display units do not authorize inferred conversion.', '']
    emit_table(lines,['System','Unit code','Display unit','Quantity count'],([v['system'],v['code'],v['unit'],v['count']] for v in a['units']))
    for kind,codes in a['codes'].items():
        lines += [f'### Synthea {kind} coding inventory','', 'Codes below count coding occurrences, not independently observed people. Observation inventory covers top-level codes; components retain their own source coding in the input.', '']
        emit_table(lines,['System','Code','Display','Count'],([v['system'],v['code'],v['display'],v['count']] for v in codes))
    return '\n'.join(lines)+'\n'


def main():
    report=json.loads(Path('data/audit/report.json').read_text())
    path=Path('docs/DATA_AUDIT.md');text=path.read_text()
    if MARKER not in text: raise ValueError('Missing documentation marker')
    path.write_text(text.split(MARKER,1)[0]+MARKER+'\n'+render(report))
    print('Rendered aggregate audit appendices:', path)

if __name__=='__main__':main()
