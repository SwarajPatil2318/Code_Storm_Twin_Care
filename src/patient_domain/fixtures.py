"""Author mock documents from an ignored, manually authored fixture specification.

This is document creation and manual fixture seeding, never EHR extraction.
No real data is bundled into source. All generated files stay under ignored data/.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5
from .models import ClinicalField, ClinicalProfile, UploadedDocument
from .repository import Repository, DEFAULT_DB, NotFound
from .pdf import write_pdf

FIXTURE_NAMESPACE = UUID('e32e29c1-b2a0-4e4a-a323-aa43605ccab7')
DEFAULT_SPEC = Path('data/mock_ehr/fixture_spec.json')


def stable(kind, key):
    return f'twincare-{kind}:{uuid5(FIXTURE_NAMESPACE, key)}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    Path(path).chmod(0o600)


def fields(patient, visit):
    born = date.fromisoformat(patient['dob'])
    at = date.fromisoformat(visit)
    age = at.year-born.year-((at.month, at.day) < (born.month, born.day))
    return {'age_years': age, 'sex': patient['sex'], 'diabetes_duration_years': patient['duration'],
            'diagnoses': ['Type 2 Diabetes', *patient['comorbidities']],
            'medications': patient['medications'], 'smoking_history': patient['smoking'],
            'smoking_pack_years': patient['pack_years'], 'alcohol_history': patient['alcohol_category'],
            'medication_context': 'explicit_insulin_term' if patient['folder']=='patient_c' else 'recorded_without_explicit_insulin_term',
            'other_medication_context': 'recorded_without_explicit_insulin_term',
            'labs': [{'test': label, 'value': value, 'unit': unit, 'observed_time': visit+'T08:00:00+05:30',
                      'available_time': visit+'T11:00:00+05:30', 'source': 'synthetic laboratory report'}
                     for label,value,unit in patient['labs']],
            'notes': patient['story']}


def content(patient, visit, kind, conflict=False):
    values = fields(patient, visit)
    meds = patient['medications']
    if conflict:
        meds = ['Metformin discontinued on 2026-10-01 because of persistent gastrointestinal intolerance',
                'Empagliflozin 10 mg orally daily started on 2026-10-01', *meds[1:]]
    header = ['SYNTHETIC TEST RECORD', 'Fictional TwinCare Test Clinic - authored integration fixture',
              f'Document date: {visit}', f'Patient: {patient["name"]}',
              f'DOB: {patient["dob"]} | Age on document date: {values["age_years"]} years | Sex: {patient["sex"]}',
              f'Primary diagnosis: Type 2 Diabetes, recorded in {patient["diagnosis_year"]}; duration {patient["duration"]} years.',
              'Relevant history: '+', '.join(patient['comorbidities']), '', 'MEDICATION RECONCILIATION', *meds, '',
              'SOCIAL HISTORY', patient['smoking'], patient['alcohol'], '', 'SELECTED LABORATORY CONTEXT',
              'Samples collected 2026-09-20 at 08:00 +05:30; results available at 11:00 +05:30.',
              *[f'{label}: {value} {unit}' for label,value,unit in patient['labs']], '']
    sections = {
        'outpatient_follow_up': ['DIABETES FOLLOW-UP', patient['story'], 'ASSESSMENT AND FOLLOW-UP', patient['followup']],
        'medication_history': ['MEDICATION HISTORY', patient['medication_note'], 'SOURCE', 'Authored fictional reconciliation; no pharmacy or device feed is connected.'],
        'lab_report': ['LABORATORY COMMENT', 'Values above are fictional test results with original units. Reference ranges and diagnostic flags are intentionally not asserted. Collection time and result availability are distinct.',
                       'CLINICAL CONTEXT', patient['story']],
        'clinical_history': ['LONGITUDINAL HISTORY', patient['history'], 'CURRENT PATTERN', patient['story']],
        'referral_follow_up': ['ROUTINE REFERRAL', patient['followup'], 'REASON', 'Routine retinal screening following diabetes review. No urgent visual symptoms are reported.'],
        'discharge_summary': ['SHORT-STAY SUMMARY', 'Admission 2026-09-07; discharge 2026-09-08. Dehydration accompanied a presumed viral gastrointestinal illness. Symptoms improved with supportive care and oral intake was tolerated before discharge.',
                              'POST-DISCHARGE RECONCILIATION', patient['medication_note'],
                              'FOLLOW-UP ADDENDUM DATED 2026-09-20', 'The laboratory context in this document is from the subsequent follow-up, not from the admission. '+patient['followup']],
        'medication_change_conflict': ['DATED MEDICATION CHANGE', 'This newer fictional record explicitly supersedes the older medication statement for conflict-review testing. The September 20 record says metformin 500 mg twice daily; this October 1 record discontinues it and starts empagliflozin.',
                                      'DEVELOPER FIXTURE NOTE', 'Deliberate conflict case. Do not silently merge either statement into a confirmed profile. The baseline profile remains as of September 20.']}
    return '\n'.join(header+sections[kind]+['', 'PROVENANCE', 'Entirely invented for integration testing. No clinical performance, extraction result or prediction is claimed.'])+'\n'


def fhir(patient, visit, pid):
    """FHIR R4 collection, local UUID references, text concepts instead of invented coding."""
    uuid = pid.split(':')[1]
    tag = {'tag': [{'system': 'urn:twincare:fixture', 'code': 'synthetic-test-record', 'display': 'SYNTHETIC TEST RECORD'}]}
    resources = [{'resourceType':'Patient','id':uuid,'meta':tag,'name':[{'text':patient['name']}],
                  'gender':patient['sex'],'birthDate':patient['dob']},
                 {'resourceType':'Condition','id':str(uuid5(FIXTURE_NAMESPACE, patient['folder']+'condition')),
                  'meta':tag,'clinicalStatus':{'coding':[{'system':'http://terminology.hl7.org/CodeSystem/condition-clinical','code':'active'}]},
                  'code':{'text':'Type 2 Diabetes'},'subject':{'reference':'urn:uuid:'+uuid},
                  'onsetString':'Recorded in '+str(patient['diagnosis_year'])}]
    for index, medication in enumerate(patient['medications']):
        resources.append({'resourceType':'MedicationRequest','id':str(uuid5(FIXTURE_NAMESPACE, patient['folder']+'med'+str(index))),
                          'meta':tag,'status':'active','intent':'order','subject':{'reference':'urn:uuid:'+uuid},
                          'medicationCodeableConcept':{'text':medication},'authoredOn':visit,
                          'dosageInstruction':[{'text':medication}]})
    for index, (label,value,unit) in enumerate(patient['labs']):
        resources.append({'resourceType':'Observation','id':str(uuid5(FIXTURE_NAMESPACE, patient['folder']+'lab'+str(index))),
                          'meta':tag,'status':'final','code':{'text':label},'subject':{'reference':'urn:uuid:'+uuid},
                          'effectiveDateTime':visit+'T08:00:00+05:30','issued':visit+'T11:00:00+05:30',
                          'valueQuantity':{'value':value,'unit':unit}})
    for label, value in [('Diabetes duration (years)',patient['duration']),('Smoking history',patient['smoking']),('Alcohol history',patient['alcohol']),('Clinical history',patient['history']),('Current reported pattern',patient['story'])]:
        resources.append({'resourceType':'Observation','id':str(uuid5(FIXTURE_NAMESPACE,patient['folder']+label)),
                          'meta':tag,'status':'final','code':{'text':label},'subject':{'reference':'urn:uuid:'+uuid},
                          'effectiveDateTime':visit+'T12:00:00+05:30','valueString':str(value)})
    return {'resourceType':'Bundle','type':'collection','meta':tag,
            'entry':[{'fullUrl':'urn:uuid:'+r['id'],'resource':r} for r in resources]}


def seed(repository, spec_path=DEFAULT_SPEC, output_root=Path('data/mock_ehr')):
    spec = json.loads(Path(spec_path).read_text())
    if spec.get('synthetic') is not True or len(spec['patients']) != 3:
        raise ValueError('Exactly three explicitly synthetic fixtures required')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    document_count = 0
    for patient in spec['patients']:
        folder = root/patient['folder']
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.chmod(0o700)
        pid = stable('patient', patient['folder'])
        try:
            existing = repository.get_patient(pid)
            if existing['synthetic'] is not True:
                raise ValueError('Fixture ownership collision')
        except NotFound:
            repository.create_patient(patient_id=pid, display_name=patient['name'],date_of_birth=patient['dob'],sex=patient['sex'],synthetic=True)
        visit = spec['document_date']
        kinds = ['outpatient_follow_up','medication_history','lab_report','clinical_history','fhir_bundle']
        if patient['folder']=='patient_b':
            kinds.append('referral_follow_up')
        if patient['folder']=='patient_c':
            kinds.append('discharge_summary')
        if patient['folder']=='patient_a':
            kinds.append('medication_change_conflict')
        inventory = []
        profile_source = None
        for kind in kinds:
            conflict = kind=='medication_change_conflict'
            location = root/'conflict_cases' if conflict else folder
            location.mkdir(parents=True, exist_ok=True, mode=0o700)
            location.chmod(0o700)
            document_date = '2026-10-01' if conflict else visit
            docid = stable('entity',patient['folder']+'/'+kind)
            expected = fields(patient, document_date)
            expected['labs'] = fields(patient, visit)['labs']
            expected.update({'display_name': patient['name'], 'date_of_birth': patient['dob']})
            if conflict:
                expected['medications'] = ['Metformin discontinued 2026-10-01','Empagliflozin 10 mg daily',*patient['medications'][1:]]
                # Current medication context is deliberately not confirmed/merged.
                expected.pop('medication_context')
            if kind=='fhir_bundle':
                target = location/'fhir_bundle.json'
                write_json(target,fhir(patient,visit,pid))
                sources = []
                media = 'application/fhir+json'
                # FHIR fixture carries raw medication text, not model normalization or derived age.
                for key in ('age_years','medication_context','other_medication_context','smoking_pack_years','notes'):
                    expected.pop(key,None)
                expected.pop('diagnoses',None)
                expected['diagnosis'] = 'Type 2 Diabetes'
                expected['alcohol_history'] = patient['alcohol']
            else:
                source = location/(kind+'.md')
                source.write_text(content(patient,document_date,kind,conflict))
                source.chmod(0o600)
                target = location/(kind+'.pdf')
                write_pdf(target,kind.replace('_',' ').title(),source.read_text())
                sources = [{'path':str(source),'sha256':sha(source)}]
                media = 'application/pdf'
                # Raw records contain medication/social text; no normalized model categories claimed as extracted.
                expected.pop('medication_context',None)
                expected.pop('other_medication_context',None)
                expected['alcohol_history'] = patient['alcohol']
                expected['diagnoses'] = ['Type 2 Diabetes',*patient['comorbidities']]
                expected.pop('notes',None)
            metadata = {'patient_id':pid,'document_id':docid,'document_type':kind,'document_date':document_date,
                        'synthetic':True,'expected_extractable_fields':expected,'storage_path':str(target),
                        'sha256':sha(target),'sources':sources,'review_status':'unreviewed',
                        'deliberate_conflict':conflict,'model_input_connected':False}
            write_json(location/(kind+'.metadata.json'),metadata)
            repository.save('documents',UploadedDocument(patient_id=pid,entity_id=docid,synthetic=True,
                            document_type=kind,document_date=document_date,filename=target.name,media_type=media,
                            storage_path=str(target),sha256=metadata['sha256'],metadata=metadata))
            if kind=='outpatient_follow_up':
                profile_source = docid
            inventory.append(metadata)
            document_count += 1
            if conflict:
                write_json(location/'metadata.json',{**metadata,'documents':[metadata]})
        write_json(folder/'metadata.json',{'patient_id':pid,'document_type':'patient_fixture_bundle',
                   'document_date':visit,'synthetic':True,'expected_extractable_fields':fields(patient,visit),
                   'documents':[d for d in inventory if not d['deliberate_conflict']]})
        values = fields(patient,visit)
        profile = ClinicalProfile(patient_id=pid,fields={name:ClinicalField(patient_id=pid,name=name,value=value,
                  unit='years' if name in ('age_years','diabetes_duration_years') else 'pack-years' if name=='smoking_pack_years' else None,
                  observed_time=visit+'T12:00:00+05:30',available_time=visit+'T12:00:00+05:30',
                  source_document_id=profile_source,source_provenance={'kind':'manually_authored_synthetic_fixture','as_of':visit})
                  for name,value in values.items()})
        repository.put_profile(profile)
    return {'synthetic_patients':3,'logical_documents':document_count,'pdf_documents':document_count-3,
            'fhir_bundles':3,'inference_connected':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['seed','reset'])
    parser.add_argument('--db',default=str(DEFAULT_DB))
    parser.add_argument('--spec',default=str(DEFAULT_SPEC))
    parser.add_argument('--output',default='data/mock_ehr')
    args=parser.parse_args()
    repo=Repository(args.db)
    if args.action=='reset':
        repo.reset_synthetic()
        print(json.dumps({'synthetic_database_reset':True}))
    else:
        print(json.dumps(seed(repo,args.spec,Path(args.output))))


if __name__=='__main__':
    main()
