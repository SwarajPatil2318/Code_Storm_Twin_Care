"""Add causal connection-state snapshots without changing P3/P4 source semantics."""
import json
from patient_meals.connections import MealConnections
from src.patient_domain.models import json_safe


class ReadinessConnections(MealConnections):
    def _save(self,db,value):
        super()._save(db,value)
        snapshot={key:value.get(key) for key in ['patient_id','source_type','status','connected_at','error']}
        row=db.execute('SELECT data FROM readiness_source_revisions WHERE patient_id=? AND source_type=? ORDER BY revision_id DESC LIMIT 1',
            (value['patient_id'],value['source_type'])).fetchone()
        if not row or json.loads(row[0])!=snapshot:
            db.execute('INSERT INTO readiness_source_revisions(patient_id,source_type,available_at,data) VALUES(?,?,?,?)',
                (value['patient_id'],value['source_type'],self._stamp(),json_safe(snapshot)))
