"""P4 extensions over frozen P3 source behavior and canonical queries."""
from patient_connections.service import Connections
from patient_connections.contracts import clock


class MealConnections(Connections):
    def __init__(self,meals,**kwargs):
        self.meals=meals;super().__init__(repository=meals.repo,**kwargs)
    def _meal_card(self,pid):
        items=self.meals.list(pid)['items'];value=self._empty(pid,'MEAL_LOGGER')
        value.update({'provider_type':'internal_meal_logger','display_status':'Active' if items else 'Ready to use',
            'status':'CONNECTED' if items else 'NOT_CONNECTED','event_count':len(items),
            'last_event_at':items[0]['meal_occurred_at'] if items else None,'last_sync_at':max(m['updated_at'] for m in items) if items else None,
            'data_start_at':min(m['meal_occurred_at'] for m in items) if items else None,'data_end_at':items[0]['meal_occurred_at'] if items else None,
            'metadata':{'meals_logged':len(items),'external_device':False},'provenance':{'origin':'User-confirmed MealEvents; contextual estimates only'},
            'quality_status':'USER_CONFIRMED' if items else 'INSUFFICIENT'})
        return value
    def list(self,pid):
        value=super().list(pid);value['items']=[self._meal_card(pid) if c['source_type']=='MEAL_LOGGER' else c for c in value['items']];return value
    def status(self,pid,source):return self._meal_card(pid) if source=='MEAL_LOGGER' else super().status(pid,source)
    def _events(self,db,pid,source=None,kind=None,start=None,end=None):
        if kind!='MEAL_EVENT':return super()._events(db,pid,source,kind,start,end)
        if source not in (None,'MEAL_LOGGER'):return []
        return [e for e in super()._events(db,pid,'MEAL_LOGGER',None,start,end) if e['event_type']=='MEAL_EVENT']
