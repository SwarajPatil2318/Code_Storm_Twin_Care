"""Reconstruct confirmed historical fields from immutable P2 revisions."""
from datetime import date
import json
import math
import pandas as pd
from patient_connections.contracts import clock
from patient_ingestion.service import IngestionService
from src.metabolic_d35.data import SAFE, medication_category


def confirmed_context(db, pid, reference):
    fields = {}
    sources = {}
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_revisions'").fetchone()
    rows = db.execute('SELECT revision_id,created_at,data FROM extraction_revisions WHERE patient_id=? ORDER BY created_at,rowid', (pid,)) if exists else []
    for row in rows:
        revision = json.loads(row['data'])
        if revision['operation'] != 'confirmed' or clock(row['created_at']) > reference:
            continue
        candidate = revision['candidate']
        if clock(candidate['confirmed_at']) > reference:
            continue
        for fact in candidate['facts']:
            if fact['decision'] != 'accept':
                continue
            if any(fact.get(k) and clock(fact[k]) > reference for k in ('available_time', 'observed_time')):
                continue
            name = fact['field_name']
            value = fact['normalized_value']
            if name in ('medications','labs'):
                merger = IngestionService._merge_medications if name == 'medications' else IngestionService._merge_labs
                value = merger(fields.get(name), value)
            fields[name] = value
            sources[name] = {'patient_id':pid, 'document_id':candidate['document_id'],
                'fact_id':fact['fact_id'], 'revision_id':row['revision_id'],
                'confirmed_at':candidate['confirmed_at'], 'revision_available_at':row['created_at'],
                'observed_time':fact.get('observed_time'), 'available_time':fact.get('available_time'),
                'unit':fact.get('unit'),'document_date':fact.get('source_document_date')}
    values = {name:fields.get(name) for name in SAFE}
    mappings = {name:name for name in SAFE if fields.get(name) is not None}
    if values['age_years'] is None and fields.get('date_of_birth'):
        born=date.fromisoformat(fields['date_of_birth']);as_of=reference.date()
        values['age_years']=as_of.year-born.year-((as_of.month,as_of.day)<(born.month,born.day))
        mappings['age_years']='date_of_birth'
    if values['medication_context'] is None and fields.get('medications'):
        # Preserve P2's explicitly reviewed list, including recorded status. This
        # reproduces the historical source-text category, not current drug use.
        values['medication_context']=medication_category(' '.join(m['name'] for m in fields['medications']))
        mappings['medication_context']='medications'
    for name in ('age_years','diabetes_duration_years','smoking_pack_years'):
        value=values[name]
        if type(value) not in (int,float) or not math.isfinite(value) or (not 18<=value<=120 if name=='age_years' else value<0):
            values[name]=None
    for name in ('sex','alcohol_history','medication_context','other_medication_context'):
        if isinstance(values[name],str):values[name]=values[name].strip().lower()
    missing=[name for name in SAFE if values[name] is None]
    frame=pd.DataFrame([{'participant_id':pid,**values}], columns=['participant_id',*SAFE])
    return {'patient_id':pid,'fields':fields,'static_values':values,'mappings':mappings,
        'sources':sources,'missing_fields':missing,'risk_context_available':len(missing)<len(SAFE)},frame
