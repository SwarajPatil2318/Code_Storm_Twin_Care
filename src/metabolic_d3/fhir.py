"""FHIR context adapter reusing the existing Synthea reference resolver and terminology helpers."""
from src.data.adapters.synthea import ReferenceResolver,coding,all_references
from .events import Event,timestamp

class FHIRAdapter:
    def adapt(self,bundle,*,patient_id,source,event_id,available_time,synthetic=False):
        if bundle.get('resourceType')!='Bundle' or not isinstance(bundle.get('entry'),list):raise ValueError('FHIR Bundle required')
        resolver=ReferenceResolver();resources=[entry['resource'] for entry in bundle['entry']]
        for entry in bundle['entry']:resolver.add(entry['resource'],entry.get('fullUrl'))
        patient_resources=[r for r in resources if r['resourceType']=='Patient']
        if len(patient_resources)!=1:raise ValueError('Exactly one FHIR patient required')
        namespace,identifier=patient_id.split(':',1);patient=patient_resources[0]
        if identifier!=patient['id']:raise ValueError('FHIR identity does not match the explicitly registered namespace')
        owner='Patient/'+identifier
        for resource in resources:
            subject=resource.get('subject',{}).get('reference')
            if subject and resolver.resolve(subject,resource)!=owner:raise ValueError('Cross-patient FHIR subject')
            if any(resolver.resolve(ref,resource) is None for ref in all_references(resource)):raise ValueError('Unresolved FHIR reference')
        cutoff,basis,_=timestamp(available_time);visible=[];withheld=0
        for resource in resources:
            if resource['resourceType']=='Patient':visible.append(resource);continue
            dates=[resource[k] for k in ['effectiveDateTime','issued','recordedDate','authoredOn','onsetDateTime'] if k in resource]
            eligible=bool(dates)
            for value in dates:
                try:
                    ns,resource_basis,_=timestamp(value)
                    eligible=eligible and resource_basis==basis and ns<=cutoff
                except (ValueError,TypeError):eligible=False
            if eligible:visible.append(resource)
            else:withheld+=1
        conditions=[{'text':r.get('code',{}).get('text'),'coding':coding(r.get('code'))} for r in visible if r['resourceType']=='Condition']
        medications=[{'text':r.get('medicationCodeableConcept',{}).get('text'),'coding':coding(r.get('medicationCodeableConcept')),
            'status':r.get('status'),'kind':'Medication order; not confirmed administration'} for r in visible if r['resourceType']=='MedicationRequest']
        labs=[{'text':r.get('code',{}).get('text'),'coding':coding(r.get('code')),'quantity':r.get('valueQuantity'),
            'observed_time':r.get('effectiveDateTime'),'available_time':r.get('issued')} for r in visible if r['resourceType']=='Observation']
        context={'demographics':{'birth_date':patient.get('birthDate'),'gender':patient.get('gender')},'diagnoses':conditions,
            'medications':medications,'labs':labs,'resources':visible,'withheld_temporal_resources':withheld,
            'label':'SYNTHETIC DEMO EHR' if synthetic else 'Historical EHR context','predictive_use':False,
            'scope':'Display context only; no medication response or HbA1c benefit is inferred'}
        return Event(event_id,patient_id,source,'ehr',available_time,available_time,None,None,None,None,synthetic,context)
