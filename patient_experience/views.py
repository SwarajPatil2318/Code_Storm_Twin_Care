"""Patient-owned presentation projections. No model or threshold calculations."""
from datetime import timedelta
from patient_connections.contracts import clock,check_owner

class ProductViews:
    def __init__(self,inference,connections,records):
        self.inference=inference;self.connections=connections;self.records=records;self.repo=inference.repo
    def summary(self,pid):
        with self.inference._lock:
            result={'patient_id':pid,'patient':self.repo.get_patient(pid),'profile':self.repo.get_profile(pid),
                'readiness':self.inference.readiness.evaluate(pid),'connections':self.connections.list(pid)['items'],
                'twin':self.inference.latest(pid),'active_alert':self.inference.alerts.active(pid)}
            check_owner(result,pid);return result
    def snapshot(self,pid):
        # The inference lock gives one coherent same-process projection while
        # CGM reconciliation executes. All collections remain owner scoped.
        with self.inference._lock:
            patient=self.repo.get_patient(pid)
            ready=self.inference.readiness.evaluate(pid)
            result={'patient_id':pid,'patient':patient,'profile':self.repo.get_profile(pid),
                'readiness':ready,'connections':self.connections.list(pid)['items'],
                'observations':self.connections.observations(pid,limit=500,recent=True)['items'],
                'meals':self.inference.readiness.meals.list(pid)['items'],
                'documents':self.records.list_documents(pid),'twin':self.inference.latest(pid),
                'alerts':self.inference.alerts.list(pid)['items'],'active_alert':self.inference.alerts.active(pid),
                'notifications':self.inference.alerts.inbox(pid)['items'],
                'predictions':self.repo.list_owned(pid,'predictions'),
                'audit':self.repo.list_owned(pid,'audit-events'),'evaluated_at':self.inference.readiness._stamp()}
            check_owner(result,pid)
            result['notification_deliveries']=[delivery for alert in result['alerts']
                for delivery in self.inference.alerts.deliveries(pid,alert['alert_id'])['items']]
            check_owner(result,pid)
            result['twin_sync']=self.freshness(result)
            result['prediction_comparisons']=self.comparisons(result)
            return result
    def freshness(self,result):
        reference=clock(result['evaluated_at'])
        eligible=[e for e in result['observations'] if e['event_type']=='CGM_READING'
                  and e['quality']['status']=='GOOD' and e['payload'].get('glucose_mg_dl') is not None
                  and clock(e['observed_at'])<=reference and clock(e['received_at'])<=reference]
        latest=max(eligible,key=lambda e:clock(e['observed_at']),default=None)
        state=result['twin'].get('twin_state')
        pending=bool(latest and (not state or clock(latest['observed_at'])>clock(state['reference_time'])))
        return {'status':('UPDATING' if result['readiness']['capabilities']['short_horizon_forecast'] else 'AWAITING_DATA') if pending else 'CURRENT' if state else 'AWAITING_DATA',
                'latest_observation_time':latest['observed_at'] if latest else None,
                'state_reference_time':state['reference_time'] if state else None}

    def comparisons(self,result):
        # Display exact observed counterparts within the original target
        # tolerance. No interpolated glucose, error score or new model output.
        events=[e for e in result['observations'] if e['event_type']=='CGM_READING' and e['quality']['status']=='GOOD' and e['payload'].get('glucose_mg_dl') is not None and clock(e['observed_at'])<=clock(result['evaluated_at']) and clock(e['received_at'])<=clock(result['evaluated_at'])]
        rows=[]
        for p in result['predictions']:
            if p['prediction_kind']!='metabolic_glucose_and_spike_risk':continue
            cells=[]
            for minutes in (30,60,90,120):
                target=clock(p['prediction_time'])+timedelta(minutes=minutes)
                candidates=[e for e in events if abs((clock(e['observed_at'])-target).total_seconds())<=450]
                selected=min(candidates,key=lambda e:(abs((clock(e['observed_at'])-target).total_seconds()),e['event_id'])) if candidates else None
                cells.append({'horizon_minutes':minutes,'predicted':p['output']['forecast'][f'plus_{minutes}m'],'target_time':target.isoformat(),
                    'observed':selected['payload']['glucose_mg_dl'] if selected else None,'observation_time':selected['observed_at'] if selected else None,'event_id':selected['event_id'] if selected else None})
            rows.append({'prediction_id':p['entity_id'],'reference_time':p['prediction_time'],'values':cells})
        return sorted(rows,key=lambda r:r['reference_time'],reverse=True)
