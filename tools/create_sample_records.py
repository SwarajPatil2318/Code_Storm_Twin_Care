"""Generate an explicitly fictional upload example locally; no record is shipped in Git."""
from datetime import date
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.patient_domain.pdf import write_pdf


def create(output=ROOT/'data/sample_records'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True,mode=0o700)
    # All values are manually authored integration fixtures, not clinical evidence.
    profile={'sex':'female','date_of_birth':'1970-01-01','age_years':date.today().year-1970,
             'diagnoses':['Type 2 Diabetes'],'diabetes_duration_years':12,'smoking_pack_years':0,
             'alcohol_history':'non-drinker',
             'medications':[{'name':'Metformin','status':'active','dose':'500 mg','frequency':'twice daily'}],
             'medication_context':'recorded_without_explicit_insulin_term',
             'other_medication_context':'recorded_without_explicit_insulin_term',
             'labs':[{'test':'HbA1c','value':7.2,'unit':'%'}]}
    value={'schema_version':'twincare-clinical/1','synthetic':True,
           'fields':{k:{'value':v}for k,v in profile.items()}}
    path=output/'clinical_history.json'
    if path.exists():raise FileExistsError('Example already exists; refusing overwrite')
    path.write_text(json.dumps(value,indent=2));path.chmod(0o600)
    pdf=output/'clinical_history.pdf'
    write_pdf(pdf,'Fictional clinical history for integration testing',
              'SYNTHETIC TEST RECORD - not a real patient\nDate: '+date.today().isoformat()+'\n'
              'Patient: Example Test Patient\nDate of birth: 1970-01-01\nSex: Female\n'
              'Diagnosis: Type 2 Diabetes\nDiabetes duration: 12 years\nSmoking: Never\n'
              'Alcohol: Never\nMedications: Metformin\nOther medications: None\nHbA1c: 7.2 %\n')
    pdf.chmod(0o600)
    print('Fictional upload examples created under data/sample_records/; review all extracted fields before confirmation.')
    return path

if __name__=='__main__':create()
