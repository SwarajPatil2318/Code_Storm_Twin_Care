/** Shared component projection from received observations and the persisted P7 inbox. */
import {mountWearable,unmountWearable} from '/assets/product-wearable.js';
let targets=new Set();
export function clearWearables(){for(const target of targets)unmountWearable(target);targets.clear();}
export function renderWearables(snapshot){
 for(const target of targets)if(!target.isConnected){unmountWearable(target);targets.delete(target);}
 const cutoff=Date.parse(snapshot.evaluated_at);
 const events=snapshot.observations.filter(e=>e.source_type==='WEARABLE'&&e.quality.status==='GOOD'&&Date.parse(e.observed_at)<=cutoff&&Date.parse(e.received_at)<=cutoff).sort((a,b)=>a.observed_at.localeCompare(b.observed_at));
 const latest=type=>events.filter(e=>e.event_type===type).at(-1);
 const hr=latest('HEART_RATE'),activity=latest('ACTIVE_MINUTES'),steps=latest('STEP_COUNT');
 const item=snapshot.notifications[0],alert=item?.alert;
 for(const target of document.querySelectorAll('[data-product-wearable]')){
  const alertView=target.dataset.productWearable==='alert';
  const readings=[];
  if(activity)readings.push({label:'Activity',value:activity.payload.minutes,unit:'active min',detail:`${activity.payload.level} · past ${activity.payload.window_minutes} min`});
  if(steps&&!alertView)readings.push({label:'Steps',value:steps.payload.count,unit:'steps',detail:'Latest step count'});
  if(alertView&&alert){readings.push({label:item.snapshot_live?'Glucose':'Glucose at alert',value:alert.current_glucose,unit:'mg/dL'},{label:'Predicted peak',value:alert.predicted_peak,unit:'mg/dL'},{label:'Spike risk',value:alert.risk_probability*100,unit:'%'});}
  mountWearable(target,{mode:'metabolic',timestamp:alert?.reference_time||hr?.observed_at||snapshot.twin.twin_state?.reference_time||snapshot.evaluated_at,heartRate:hr?.payload.bpm,readings,
   notification:alert?{alertId:alert.alert_id,acknowledged:alert.state==='ACKNOWLEDGED',referenceTime:alert.reference_time}:undefined});
  targets.add(target);
 }
}
