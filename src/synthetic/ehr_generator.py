"""Synthea-inspired FHIR-compatible historical context, linked to virtual identity.

Resource fields/coding patterns come from the audited Synthea sample. This is
not a new run of Synthea's disease modules and does not imply full R4 validation.
Unsupported scenario concepts use an explicit local synthetic code system.
"""
import datetime as dt
import hashlib
from src.synthetic.config import SIGNALS,UNITS,EVIDENCE,STATUS

SYSTEM='https://twin-patient.example/synthetic-concept'
LOINC={'heart_rate':'8867-4','systolic_bp':'8480-6','diastolic_bp':'8462-4','spo2':'2708-6','weight':'29463-7','temperature':'8310-5'}
UCUM={'bpm':'/min','mmHg':'mm[Hg]','degC':'Cel','count/hour':'{steps}/h','h':'h','kg':'kg','%':'%','/min':'/min','ms':'ms'}


def instant(patient,hour):return (dt.datetime.fromisoformat(patient['start_time'])+dt.timedelta(hours=float(hour))).isoformat()


def code(concept,grounding=None,table='conditions'):
    search={'hf':'heart failure','hypertension':'hypertension','diabetes':'diabetes','ckd':'chronic kidney','beta_blocker':'metoprolol','ace_arb':'lisinopril','diuretic':'furosemide','mra':'spironolactone','sglt2':'empagliflozin'}.get(concept,concept)
    if grounding:
        for item in grounding['synthea']['common_codes'].get(table,[]):
            if search in item['display'].lower():return {'coding':[{k:item[k] for k in ['system','code','display']}]}
    return {'coding':[{'system':SYSTEM,'code':concept,'display':'Synthetic '+concept.replace('_',' ')}]}


def resource_id(patient,kind,concept):return hashlib.sha256((patient['patient_id']+':'+kind+':'+concept).encode()).hexdigest()[:24]


def bundle(patient,grounding,stream=None):
    reference='Patient/'+patient['patient_id'].split(':')[1];c=patient['context'];start=dt.datetime.fromisoformat(patient['start_time']);birth=(start.date()-dt.timedelta(days=365.2425*c['age'])).isoformat()
    resources=[{'resourceType':'Patient','id':reference.split('/')[1],'identifier':[{'system':SYSTEM,'value':patient['patient_id']}],
      'gender':'female' if c['female'] else 'male','birthDate':birth,'meta':{'tag':[{'system':SYSTEM,'code':'synthetic-poc','display':'SYNTHETIC POC / NOT VALIDATED'}]}}]
    encounter_id=resource_id(patient,'Encounter','baseline');encounter={'resourceType':'Encounter','id':encounter_id,'status':'finished','subject':{'reference':reference},
      'class':{'system':'http://terminology.hl7.org/CodeSystem/v3-ActCode','code':'AMB'},'period':{'start':instant(patient,-24*21),'end':instant(patient,-24*21+1)}};resources.append(encounter)
    for concept in ['hf','hypertension','diabetes','ckd']:
        if concept!='hf' and not c[concept]:continue
        resources.append({'resourceType':'Condition','id':resource_id(patient,'Condition',concept),'subject':{'reference':reference},'code':code(concept,grounding),
          'clinicalStatus':{'coding':[{'system':'http://terminology.hl7.org/CodeSystem/condition-clinical','code':'active'}]},
          'verificationStatus':{'coding':[{'system':'http://terminology.hl7.org/CodeSystem/condition-ver-status','code':'confirmed'}]},'onsetDateTime':instant(patient,-24*90),'recordedDate':instant(patient,-24*21)})
    for med in ['beta_blocker','ace_arb','diuretic','mra','sglt2']:
        if c[med]:resources.append({'resourceType':'MedicationRequest','id':resource_id(patient,'MedicationRequest',med),'status':'active','intent':'order',
          'subject':{'reference':reference},'authoredOn':instant(patient,-24*21),'medicationCodeableConcept':code(med,grounding,'medications')})
    units={'ef':'%','bmi':'kg/m2','hf_severity':'1','creatinine':'mg/dL','sodium':'mmol/L','hemoglobin':'g/dL','bnp':'pg/mL'}
    for concept,unit in units.items():
        if c[concept] is None:continue
        resources.append({'resourceType':'Observation','id':resource_id(patient,'Observation',concept),'status':'final','subject':{'reference':reference},'code':code(concept),
          'effectiveDateTime':instant(patient,-24*21),'issued':instant(patient,-24*20.9),'valueQuantity':{'value':c[concept],'unit':unit,'system':'http://unitsofmeasure.org','code':unit}})
    if stream is not None:
        import numpy as np
        # Selected historical demo observations, with original availability and QC.
        for j,signal in enumerate(SIGNALS):
            indices=np.flatnonzero(~stream['missingness_mask'][:,j])[:3]
            for t in indices:
                resources.append({'resourceType':'Observation','id':resource_id(patient,'Observation',signal+':'+str(t)),'status':'final','subject':{'reference':reference},
                  'code':{'coding':[{'system':'http://loinc.org' if signal in LOINC else SYSTEM,'code':LOINC.get(signal,signal),'display':signal}]},
                  'effectiveDateTime':instant(patient,t),'issued':instant(patient,stream['available_hour'][t,j]),
                  'valueQuantity':{'value':float(stream['value'][t,j]),'unit':UNITS[j],'system':'http://unitsofmeasure.org','code':UCUM[UNITS[j]]}})
    return {'resourceType':'Bundle','type':'collection','meta':{'tag':[{'system':SYSTEM,'code':'synthetic-poc','display':EVIDENCE+' / '+STATUS}]},
      'entry':[{'fullUrl':'https://twin-patient.example/fhir/'+r['resourceType']+'/'+r['id'],'resource':r} for r in resources]}
