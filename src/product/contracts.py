"""Explicit approved-field boundary. Reject unknown keys rather than leaking them."""
import re

STATE_FIELDS = {'patient','timestamp','replay_day','risk','divergence','status','actual_state',
                'shadow_state','personal_baseline','why_now','data_quality','signal_units',
                'state_aggregation','ehr_context','evidence','population_context'}
PATIENT_FIELDS = {'id','name','age','synthetic','label','story','context','note'}
PRIVATE_KEYS = {'latent','state','states','profile','seed','event_time','event_time_hours',
                'event_within72h','labels','outcomes','ground_truth','vulnerability','deterioration_onset',
                'adjusted_risk','recovery_score','measurement_needed','fit_patient_ids','patient_index_offset'}
SIGNALS = {'heart_rate','hrv_rmssd','systolic_bp','diastolic_bp','spo2','respiration','weight','steps','sleep_hours','temperature'}
NESTED = {
 'risk': {'next_24h','next_48h','next_72h','likely_window','bands','source'},
 'divergence': {'descriptive_tdi','status','components'},
 'why_now': {'personal_deviations','temporal_changes','missing_data','interpretation'},
 'data_quality': {'level','availability_score','missing','scope'},
 'evidence': {'risk','divergence','clinical','range'},
 'population_context': {'definition','broadly_in_core_bands'},
}

def scan_safe(value):
    if isinstance(value, dict):
        if set(value) & PRIVATE_KEYS:
            raise ValueError('Private data rejected at product boundary')
        for child in value.values(): scan_safe(child)
    elif isinstance(value, list):
        for child in value: scan_safe(child)
    elif isinstance(value, str) and re.search(r'synthetic:[0-9a-f]{24}', value):
        raise ValueError('Source identifier rejected at product boundary')


def exact(value, fields):
    if set(value) != fields:
        raise ValueError('Unapproved product schema')


def validate_state(value):
    exact(value, STATE_FIELDS)
    exact(value['patient'], PATIENT_FIELDS)
    if not value['patient']['synthetic'] or value['patient']['label']!='SYNTHETIC DEMO PATIENT':
        raise ValueError('Synthetic label required')
    for key, fields in NESTED.items(): exact(value[key], fields)
    for key in ('actual_state','shadow_state','personal_baseline','signal_units'):
        exact(value[key], SIGNALS)
    for signal in SIGNALS:
        exact(value['shadow_state'][signal], {'value','lower','upper'})
        exact(value['personal_baseline'][signal], {'reference','scale','trusted_count'})
    for deviation in value['why_now']['personal_deviations']:
        exact(deviation, {'signal','observed','expected','difference','relative_percent','standardized_residual',
                          'divergence_component','recent_48h_deviation_fraction','recent_count'})
    for change in value['why_now']['temporal_changes']: exact(change, {'signal','recent_minus_prior_48h'})
    for missing in value['why_now']['missing_data']: exact(missing, {'signal','received_recent_count','baseline_confidence'})
    r = [value['risk'][f'next_{h}h'] for h in (24,48,72)]
    if not 0 <= r[0] <= r[1] <= r[2] <= 1: raise ValueError('Invalid cumulative horizon risk')
    if value['data_quality']['level'] not in {'Good','Moderate','Limited'}: raise ValueError('Unknown quality')
    scan_safe(value)


def _validate_structure(cache):
    exact(cache, {'metadata','patients'})
    scan_safe(cache)
    if len(cache['patients']) != 3: raise ValueError('Three demo patients required')
    for item in cache['patients']:
        exact(item, {'patient','timeline','ehr','observations'})
        exact(item['patient'], PATIENT_FIELDS)
        if len(item['timeline']) != 8: raise ValueError('Eight daily replay points required')
        for state in item['timeline']: validate_state(state)
        if [s['replay_day'] for s in item['timeline']] != list(range(-7,1)):
            raise ValueError('Invalid replay chronology')
        dates = [s['timestamp'] for s in item['timeline']]
        if dates != sorted(set(dates)): raise ValueError('Invalid replay timestamps')
    return cache

CONTEXT_FIELDS={'demographics','diagnoses','medications','labs','scope'}
METADATA_FIELDS={'version','reference','model_code_sha256','config_sha256','model_sha256','calibration_sha256',
                 'prediction_cache_sha256','sources','limitations','cohort_separation','adaptive_sensing','selection','display_bands'}


def validate_context(context, fhir=False):
    exact(context, CONTEXT_FIELDS | ({'fhir'} if fhir else set()))
    exact(context['demographics'], {'age','gender'})
    for lab in context['labs']: exact(lab, {'name','value','unit','observed_time','available_time'})
    if fhir:
        bundle=context['fhir']
        exact(bundle, {'resourceType','type','meta','entry'})
        for entry in bundle['entry']:
            exact(entry, {'fullUrl','resource'})
            r=entry['resource']
            allowed={
                'Patient': {'resourceType','id','identifier','gender','birthDate','meta'},
                'Encounter': {'resourceType','id','status','subject','class','period'},
                'Condition': {'resourceType','id','subject','code','clinicalStatus','verificationStatus','onsetDateTime','recordedDate'},
                'MedicationRequest': {'resourceType','id','status','intent','subject','authoredOn','medicationCodeableConcept'},
                'Observation': {'resourceType','id','status','subject','code','effectiveDateTime','issued','valueQuantity'},
            }
            if r.get('resourceType') not in allowed: raise ValueError('Unapproved FHIR resource')
            exact(r, allowed[r['resourceType']])
            # Nested FHIR value shapes also have a closed key vocabulary.
            vocabulary={'system','value','tag','code','display','coding','reference','start','end','unit'}
            def nested(value):
                if isinstance(value,dict):
                    if not set(value)<=vocabulary: raise ValueError('Unapproved nested FHIR field')
                    for child in value.values():nested(child)
                elif isinstance(value,list):
                    for child in value:nested(child)
            for key in set(r)-{'resourceType','id','gender','birthDate','status','intent','onsetDateTime','recordedDate','authoredOn','effectiveDateTime','issued'}:
                nested(r[key])
        nested(bundle['meta'])


def validate_cache(cache):
    _validate_structure(cache)
    exact(cache['metadata'],METADATA_FIELDS)
    for source in cache['metadata']['sources']:exact(source, {'name','role','label'})
    for item in cache['patients']:
        validate_context(item['ehr'],True)
        for state in item['timeline']:
            validate_context(state['ehr_context'])
            exact(state['risk']['bands'], {'24','48','72'})
            exact(state['divergence']['components'],SIGNALS)
            for m in state['data_quality']['missing']: exact(m, {'signal','received_recent_count','baseline_confidence'})
        for obs in item['observations']:
            exact(obs, {'observation_id','patient_id','dataset','concept','observed_time','time_basis',
                        'available_time','availability_basis','value','coding','unit_code','unit_system','metadata'})
            if obs['patient_id']!=item['patient']['id'] or obs['concept'] not in SIGNALS:
                raise ValueError('Observation reference mismatch')
            exact(obs['value'], {'raw_value','usable_value','quality_status','reasons','unit','provenance'})
            exact(obs['value']['provenance'], {'dataset','source_file','source_record','original_field',
                   'transformed_field','transformation','source_sha256','original_unit','unit_support'})
            exact(obs['metadata'], {'synthetic','sensor_source','scheduled_opportunity_time'})
    return cache
