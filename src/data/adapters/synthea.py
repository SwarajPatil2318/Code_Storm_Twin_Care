"""FHIR engineering ingestion with cross-bundle and contained reference lookup.

This is a canonical extractor, not a complete FHIR profile validator or server.
"""
from collections import Counter, defaultdict
import datetime as dt
import json
from urllib.parse import parse_qs
import zipfile

from .base import Adapter
from ..io import strict_json_loads
from ..quality import SchemaError, classify, normalize_id
from ..schema import Observation, Outcome, record

FILENAME = 'synthea_sample_data_fhir_latest.zip'
SELECTED_TYPES = {'Patient','Condition','Observation','MedicationRequest','Encounter','Procedure'}
# Code mapping is small and explicit; unknown concepts retain original system/code.
LOINC = {
    '8867-4': ('heart_rate',{'/min','bpm'}), '8480-6': ('systolic_bp',{'mm[Hg]','mmHg'}),
    '8462-4': ('diastolic_bp',{'mm[Hg]','mmHg'}), '2708-6': ('spo2',{'%'}),
    '29463-7': ('weight',{'kg'}), '8310-5': ('temperature',{'Cel','degC'}),
}


def coding(concept):
    if concept is None:
        return []
    if not isinstance(concept, dict) or not isinstance(concept.get('coding', []), list):
        raise SchemaError('Selected FHIR code must be a CodeableConcept')
    return concept.get('coding', [])


def parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        timestamp = dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:
        return None
    # Date-only/partial/timezone-less values stay source metadata, not invented UTC instants.
    if timestamp.tzinfo is None:
        return None
    return timestamp.astimezone(dt.timezone.utc).isoformat()


def temporal(resource):
    kind = resource['resourceType']
    observed = (resource.get('effectiveDateTime') or resource.get('effectivePeriod',{}).get('start')
                or resource.get('onsetDateTime') or resource.get('period',{}).get('start')
                or resource.get('performedDateTime') or resource.get('performedPeriod',{}).get('start')
                or resource.get('authoredOn'))
    available = resource.get('issued') if kind == 'Observation' else resource.get('recordedDate') if kind == 'Condition' else resource.get('authoredOn') if kind == 'MedicationRequest' else None
    return parse_time(observed), parse_time(available), observed, available


class ReferenceResolver:
    def __init__(self):
        self.aliases = {}
        self.identifiers = defaultdict(set)
        self.metadata = {}

    def add(self, resource, full_url=None):
        kind, identifier = resource.get('resourceType'), resource.get('id')
        if not isinstance(kind, str) or not isinstance(identifier, str) or not kind or not identifier:
            raise SchemaError('FHIR resource missing type or ID')
        key = kind + '/' + identifier
        if key in self.metadata:
            raise SchemaError('Duplicate FHIR resource identity; adjudication required')
        self.metadata[key] = {'resourceType':kind, 'id':identifier,
                              'birthDate':resource.get('birthDate'), 'code':resource.get('code')}
        for alias in (key, full_url):
            if not alias:
                continue
            if alias in self.aliases and self.aliases[alias] != key:
                raise SchemaError('Conflicting FHIR reference aliases')
            self.aliases[alias] = key
        for item in resource.get('identifier', []):
            self.identifiers[(kind,item.get('system',''),str(item.get('value','')))].add(key)

    def resolve(self, reference, owner=None):
        if not reference:
            return None
        if reference.startswith('#'):
            contained = [r for r in (owner or {}).get('contained', []) if r.get('id') == reference[1:]]
            return 'contained:' + contained[0]['resourceType'] + '/' + reference[1:] if len(contained) == 1 else None
        direct = self.aliases.get(reference)
        if direct:
            return direct
        if '/_history/' in reference:
            return self.aliases.get(reference.split('/_history/', 1)[0])
        if '?identifier=' in reference:
            kind, query = reference.split('?',1)
            values = parse_qs(query).get('identifier', [])
            if len(values) != 1:
                return None
            token = values[0]
            system, sep, value = token.partition('|')
            if sep:
                candidates = self.identifiers.get((kind,system,value),set())
            else:
                candidates = set().union(*(keys for (t, _, v), keys in self.identifiers.items() if t == kind and v == token))
            return next(iter(candidates)) if len(candidates) == 1 else None
        return None


def all_references(resource):
    if isinstance(resource, dict):
        if isinstance(resource.get('reference'), str):
            yield resource['reference']
        for value in resource.values():
            yield from all_references(value)
    elif isinstance(resource, list):
        for value in resource:
            yield from all_references(value)


class SyntheaAdapter(Adapter):
    def __init__(self, root, writer):
        super().__init__(root, writer, 'synthea')
        self.resolver = ReferenceResolver()

    def bundles(self, archive):
        for member in sorted(archive.namelist()):
            if not member.endswith('.json'):
                if not member.endswith('/'):
                    raise SchemaError('Unexpected non-JSON FHIR archive member')
                continue
            bundle = strict_json_loads(archive.read(member))
            if bundle.get('resourceType') != 'Bundle' or not isinstance(bundle.get('entry'),list):
                raise SchemaError('Expected FHIR Bundle with entry array')
            for index, entry in enumerate(bundle['entry']):
                if not isinstance(entry.get('resource'),dict):
                    raise SchemaError('FHIR bundle entry lacks resource object')
                yield member, index, entry

    def build(self):
        self.register_input(FILENAME)
        with zipfile.ZipFile(self.root/FILENAME) as archive:
            for _, _, entry in self.bundles(archive):
                self.resolver.add(entry['resource'],entry.get('fullUrl'))
            patient_keys = sorted(key for key,value in self.resolver.metadata.items() if value['resourceType']=='Patient')
            for key in patient_keys:
                self.add_patient(self.resolver.metadata[key]['id'], ['synthetic_fhir_integration'])
            resource_counts, refs = Counter(), Counter()
            for member, index, entry in self.bundles(archive):
                resource = entry['resource']
                kind = resource['resourceType']
                resource_counts[kind] += 1
                locator = f'{member}:entry:{index}'
                self.preserve(member, f'entry:{index}',resource,kind)
                unresolved = []
                for reference in all_references(resource):
                    refs['total'] += 1
                    resolved = self.resolver.resolve(reference,resource)
                    if not resolved:
                        unresolved.append(reference)
                        refs['unresolved'] += 1
                        self.issue('unresolved_fhir_reference', source_record=locator)
                    if '?identifier=' in reference:
                        refs['conditional_identifier'] += 1
                        refs['conditional_identifier_resolved'] += resolved is not None
                if kind not in SELECTED_TYPES:
                    continue
                if kind == 'Patient':
                    patient = normalize_id(resource['id'],self.dataset)
                else:
                    target = self.resolver.resolve(resource.get('subject',{}).get('reference'),resource)
                    if not target or not target.startswith('Patient/'):
                        self.issue('inconsistent_patient_reference',source_record=locator)
                        self.writer.emit('quarantine',{'record_kind':'fhir_resource','source_record':locator,'raw_resource':resource,
                                                       'reason':'inconsistent_patient_reference'})
                        continue
                    patient = normalize_id(target.split('/',1)[1],self.dataset)
                observed, available, raw_observed, raw_available = temporal(resource)
                if kind == 'Patient':
                    attrs = {}
                    for field in ('gender','birthDate','maritalStatus','extension'):
                        prov=self.provenance(member, f'entry:{index}',field,field,'preserve_fhir_demographic',archive=FILENAME)
                        attrs[field]=classify(resource.get(field),field,prov,numeric=False)
                    self.context(patient,attrs,'synthetic_patient')
                    if resource.get('deceasedDateTime'):
                        prov=self.provenance(member,f'entry:{index}','deceasedDateTime','synthetic_death','preserve_synthetic_event',archive=FILENAME)
                        value=classify(resource['deceasedDateTime'],'synthetic_death',prov,numeric=False)
                        self.outcome(Outcome(patient,self.dataset,'synthetic_death',value,parse_time(resource['deceasedDateTime']),'absolute_utc',
                                             metadata={'synthetic':True,'not_clinical_validation':True}))
                elif kind == 'Observation':
                    self.extract_observation(resource,patient,member,index,unresolved)
                else:
                    if kind == 'MedicationRequest' and resource.get('medicationReference'):
                        key = self.resolver.resolve(resource['medicationReference'].get('reference'),resource)
                        code = self.resolver.metadata.get(key,{}).get('code')
                    else:
                        code = resource.get('medicationCodeableConcept') if kind=='MedicationRequest' else resource.get('code')
                    codes = coding(code) if code is not None else []
                    if kind=='Encounter':
                        codes = [c for concept in resource.get('type',[]) for c in coding(concept)]
                    prov=self.provenance(member,f'entry:{index}','resource',kind,'preserve_fhir_event_fields',archive=FILENAME)
                    table={'Condition':'conditions','MedicationRequest':'medications','Encounter':'encounters','Procedure':'procedures'}[kind]
                    self.writer.emit(table, {'patient_id':patient,'dataset':self.dataset,'resource_id':resource['id'],
                                            'resource_type':kind,'coding':codes,'observed_time':observed,
                                            'available_time':available,'raw_observed_time':raw_observed,'raw_available_time':raw_available,
                                            'status':resource.get('status') or resource.get('clinicalStatus'),
                                            'raw_fields':resource,'provenance':prov.__dict__,
                                            'resolved_references':{ref:self.resolver.resolve(ref,resource) for ref in all_references(resource)},
                                            'as_of_caution':'Future status/abatement/end fields are not approved predictors.'})
                    if observed is None:
                        self.issue('missing_or_uninterpreted_event_timestamp',source_record=locator,status='SUSPECT')
        self.stats.update(resource_counts)
        self.extra_report['fhir_references']=dict(sorted(refs.items()))
        self.extra_report['resource_counts']=dict(sorted(resource_counts.items()))
        self.notes += ['Synthetic integration data only; no clinical risk labels or evaluation partitions.',
                       'All original resources preserved; profile conformance and historical status semantics not certified.',
                       'Date-only/unzoned instants remain raw metadata and are not invented UTC times.']

    def extract_observation(self, resource, patient, member, index, unresolved):
        observed, available, raw_observed, raw_available = temporal(resource)
        components = resource.get('component',[])
        candidates = []
        if any(key.startswith('value') for key in resource) or not components:
            candidates.append(('root',resource))
        candidates.extend((f'component:{i}', component) for i,component in enumerate(components))
        for path, component in candidates:
            codes = coding(component.get('code'))
            concept = next((LOINC[c['code']][0] for c in codes if c.get('system')=='http://loinc.org' and c.get('code') in LOINC),
                           'fhir:' + (codes[0].get('system','unknown') + ':' + codes[0].get('code','unknown') if codes else 'unknown'))
            quantities = component.get('valueQuantity')
            raw = quantities.get('value') if quantities is not None else next((v for k,v in component.items() if k.startswith('value')), None)
            unit = quantities.get('unit') if quantities is not None else None
            unit_code = quantities.get('code') if quantities is not None else None
            unit_system = quantities.get('system') if quantities is not None else None
            reasons = []
            if not codes: reasons.append('missing_fhir_observation_code')
            if observed is None: reasons.append('missing_or_invalid_timestamp')
            if unresolved: reasons.append('unresolved_fhir_reference')
            if available is not None and observed is not None and available < observed: reasons.append('available_before_observed_time')
            for c in codes:
                if quantities is not None and c.get('system')=='http://loinc.org' and c.get('code') in LOINC:
                    if unit_system!='http://unitsofmeasure.org' or (unit_code or unit) not in LOINC[c['code']][1]:
                        reasons.append('unknown_or_unsupported_physiological_unit')
            birth = self.resolver.metadata['Patient/'+patient.split(':',1)[1]].get('birthDate')
            pediatric = False
            if birth and observed:
                try:
                    observed_date, birth_date = dt.date.fromisoformat(observed[:10]), dt.date.fromisoformat(birth)
                    age = observed_date.year - birth_date.year - ((observed_date.month, observed_date.day) < (birth_date.month, birth_date.day))
                    pediatric = age < 18
                except ValueError: reasons.append('invalid_birth_date')
            prov=self.provenance(member,f'entry:{index}',f'{path}.valueQuantity' if quantities is not None else f'{path}.value[x]',concept,
                                  'extract_fhir_component_no_unit_conversion',unit=unit,support='fhir_quantity' if quantities is not None else 'not_applicable',archive=FILENAME)
            value=classify(raw,concept,prov,unit,numeric=quantities is not None,extra_reasons=reasons,pediatric=pediatric)
            self.observation(Observation(f'synthea:{resource["id"]}:{path}',patient,self.dataset,concept,observed,'absolute_utc',
                                         available,'issued' if available else 'unknown_source_availability',value,codes,unit_code,unit_system,
                                         {'raw_observed_time':raw_observed,'raw_available_time':raw_available,'fhir_status':resource.get('status'),
                                          'data_absent_reason':component.get('dataAbsentReason'),'source_resource_id':resource['id'],
                                          'component_path':path, 'pediatric_weight_screen_exempt':pediatric and concept=='weight'}))
