// Display-only helpers. The trend threshold is supplied from the existing config.
export function trend(value,threshold){
 if(!Number.isFinite(value))return {label:'Unavailable',rate:'—',arrow:''};
 if(!Number.isFinite(threshold)||threshold<=0)throw Error('Existing trend threshold required');
 const label=value>threshold?'Rising':value< -threshold?'Falling':'Stable';
 const rounded=Number(value.toFixed(1));
 return {label,rate:(Object.is(rounded,-0)?0:rounded).toFixed(1),arrow:label==='Rising'?'↗':label==='Falling'?'↘':'→'};
}
export function duration(minutes){
 if(!Number.isFinite(minutes)||minutes<0)return 'Not available';
 const m=Math.floor(minutes);if(m<60)return `${m} min`;
 if(m<1440)return `${Math.floor(m/60)} hr${m%60?` ${m%60} min`:''}`;
 return `${Math.floor(m/1440)} ${m<2880?'day':'days'}${Math.floor(m%1440/60)?` ${Math.floor(m%1440/60)} hr`:''}`;
}
export function displayTime(value,short=false){
 if(!value)return 'Not yet';const d=new Date(value);if(!Number.isFinite(d.getTime()))return 'Not available';
 return d.toLocaleString('en-GB',{...(short?{}:{month:'short',day:'numeric'}),hour:'2-digit',minute:'2-digit',timeZone:'UTC'})+' UTC';
}
export function newerSnapshot(current,incoming){
 if(!current)return true;
 const oldState=current.twin?.twin_state,newState=incoming.twin?.twin_state;
 const created=(data,id)=>Date.parse(data.predictions?.find(p=>p.entity_id===id)?.created_at||'');
 if(oldState&&newState&&oldState.reference_time===newState.reference_time&&created(incoming,newState.prediction_id)<created(current,oldState.prediction_id))return false;
 if(oldState&&(!newState||Date.parse(newState.reference_time)<Date.parse(oldState.reference_time)))return false;
 return Date.parse(incoming.evaluated_at)>=Date.parse(current.evaluated_at);
}

// Overview components are presentation projections of a single completed TwinState.
export function LiveStatus(snapshot,{esc,status,date,state:preview,replay=false}){
 const state=preview||snapshot.twin.twin_state,session=snapshot.simulation,running=!replay&&session?.status==='RUNNING';
 const ready=snapshot.readiness.capabilities.short_horizon_forecast;
 const label=replay?'Prediction history':session?.status==='ERROR'?'Monitoring needs attention':snapshot.twin_sync?.status==='UPDATING'?'Updating Twin':ready?(snapshot.readiness.personalization.ready?'Twin ready':'Forecast ready'):status(snapshot.readiness);
 return `<div class="premium-ready ${ready?'is-ready':''} ${running&&ready?'is-live':''}"><i aria-hidden="true"></i><div><strong>${esc(label)}</strong><small>${state?'Last updated '+date(state.reference_time):'Building your personal Twin'}</small></div></div>${state||session?.simulation_time?`<div class="premium-reference" aria-label="${state?'Prediction reference time':'Monitoring time'}">${date(state?.reference_time||session.simulation_time)}</div>`:''}${!replay&&session&&session.status!=='NOT_CONFIGURED'?`<button class="primary premium-monitor-button" data-simulation="${running?'pause':session.status==='PAUSED'?'resume':'start'}" aria-label="${running?'Pause monitoring':session.status==='PAUSED'?'Resume monitoring':'Start monitoring'}"><span aria-hidden="true">${running?'Ⅱ':'▷'}</span>${running?'Pause monitoring':session.status==='PAUSED'?'Resume monitoring':'Start monitoring'}</button>`:''}<details class="premium-actions"><summary aria-label="Twin actions">···</summary><div><button data-action="meal">Log meal</button><button data-action="refresh">Refresh Twin</button><button data-action="replay">Prediction history</button>${!replay&&session&&session.status!=='NOT_CONFIGURED'?'<button data-simulation="stop" aria-label="Stop monitoring">Stop monitoring</button>':''}</div></details>`;
}
export function TwinFlow(state){
 const personalized=state?.shadow.available,magnitude=state?.divergence?.magnitude;
 const gap=personalized&&Number.isFinite(magnitude)?Math.min(2.5,Math.max(0,magnitude))*17:0;
 const path=(shift,width=19)=>`M15 ${130+shift} C110 ${44+shift},155 ${172+shift},270 ${123+shift} S425 ${42+shift},500 ${89+shift} S615 ${179+shift},685 ${110+shift} L685 ${110+shift+width} C615 ${179+shift+width},540 ${119+shift+width},500 ${89+shift+width} S350 ${85+shift+width},270 ${123+shift+width} S110 ${44+shift+width},15 ${130+shift+width} Z`;
 const points=[[65,106],[111,95],[158,122],[207,141],[254,133],[307,98],[360,72],[406,69],[452,84],[506,99],[555,130],[601,146],[647,136]];
 return `<svg class="premium-flow ${personalized?'is-personal':'is-learning'}" viewBox="0 0 700 240" role="img" aria-label="Illustrative Twin streams: coral for the current and predicted response${personalized?', cyan for personal expectation. Visual separation reflects the stored personal comparison':'. Personal patterns are still learning'}"><defs><linearGradient id="twin-coral" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#ff797b" stop-opacity=".12"/><stop offset=".45" stop-color="#f45258" stop-opacity=".95"/><stop offset="1" stop-color="#ffb8b8" stop-opacity=".08"/></linearGradient><linearGradient id="twin-cyan" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#bceefa" stop-opacity=".1"/><stop offset=".5" stop-color="#00a3cd" stop-opacity=".78"/><stop offset="1" stop-color="#9de7fa" stop-opacity=".1"/></linearGradient><radialGradient id="twin-red-particle"><stop stop-color="#ffc2c4"/><stop offset=".45" stop-color="#ff7379"/><stop offset="1" stop-color="#e54956"/></radialGradient><radialGradient id="twin-blue-particle"><stop stop-color="#d6f7ff"/><stop offset=".5" stop-color="#51d3f0"/><stop offset="1" stop-color="#008fb5"/></radialGradient><filter id="twin-soft"><feGaussianBlur stdDeviation="9"/></filter></defs><ellipse cx="360" cy="109" rx="245" ry="58" fill="#fef0ef" opacity=".6" filter="url(#twin-soft)"/><ellipse cx="340" cy="151" rx="240" ry="38" fill="#e0f7fc" opacity=".7" filter="url(#twin-soft)"/>${[-8,0,8].map((n,i)=>`<path class="twin-ribbon" d="${path(n+10+gap/2)}" fill="url(#twin-cyan)" opacity="${.35+i*.2}"/><path class="twin-ribbon" d="${path(n-10-gap/2)}" fill="url(#twin-coral)" opacity="${.35+i*.2}"/>`).join('')}<path d="M15 ${118-gap/2} C110 ${32-gap/2},155 ${160-gap/2},270 ${111-gap/2} S425 ${30-gap/2},500 ${77-gap/2} S615 ${167-gap/2},685 ${98-gap/2}" fill="none" stroke="#ffced0" stroke-width="2"/><path d="M15 ${152+gap/2} C110 ${66+gap/2},155 ${194+gap/2},270 ${145+gap/2} S425 ${64+gap/2},500 ${111+gap/2} S615 ${201+gap/2},685 ${132+gap/2}" fill="none" stroke="#c3f4ff" stroke-width="2"/>${points.map(([x,y],i)=>`<circle class="twin-particle" cx="${x}" cy="${y-22-gap/2}" r="${i%4===0?15:i%3===0?7:3}" fill="url(#twin-red-particle)" style="--drift:${i%2?14:-10}px;animation-delay:-${i*1.7}s"/>${i%2===0?`<circle class="twin-particle" cx="${x+18}" cy="${y+30+gap/2}" r="${i%4===0?10:4}" fill="url(#twin-blue-particle)" style="--drift:12px;animation-delay:-${i*1.1}s"/>`:''}`).join('')}</svg>`;
}
function personalDeparture(state,{fmt}){
 const departure=state?.why_now.find(i=>i.signal==='departure_from_expected_now');
 return departure?`${fmt(Math.abs(departure.value))} mg/dL ${departure.value>=0?'above':'below'} personal expectation`:'Learning your personal pattern';
}
export function ForecastHorizonCard(state,minutes,{fmt}){
 const value=state?.forecast[`plus_${minutes}m`],direction=Number.isFinite(value)?Math.sign(Number(fmt(value))-Number(fmt(state.current_glucose))):0;
 return `<article class="forecast-horizon"><span class="label">+${minutes} min</span><div class="forecast-chip ${direction>0?'is-above':direction<0?'is-below':''}"><div><strong class="value">${fmt(value)}</strong><small>mg/dL</small></div><span class="forecast-direction" aria-hidden="true">${state?(direction>0?'↗':direction<0?'↘':'→'):'—'}</span><p>${state?(direction>0?'Above current':direction<0?'Below current':'Near current'):'Awaiting forecast'}</p></div></article>`;
}
export function MetabolicTwinHero(snapshot,state,h){
 const {esc,glucoseTrend}=h,running=!h.replay&&snapshot.simulation?.status==='RUNNING';
 const statements={SPIKE_WATCH:'The forecast indicates elevated glucose risk',STABLE:state?`The recent glucose trend is ${glucoseTrend(state).label.toLowerCase()}`:'',RISING:state?`The recent glucose trend is ${glucoseTrend(state).label.toLowerCase()}`:'',PERSONALLY_ELEVATED_RESPONSE:'The response is above personal expectation',CURRENT_GLUCOSE_ELEVATED:'Current glucose is elevated'};
 const statement=state?(statements[state.status]||'Your personal glucose trajectory'):'Building your first forecast';
 return `<section class="premium-hero hero"><div class="premium-hero-copy"><div class="premium-live ${running?'is-live':''}"><i aria-hidden="true"></i>${running?'Live monitoring':snapshot.simulation?.status==='PAUSED'?'Monitoring paused':'Your personal Twin'}</div><h2 class="hero-title">Your metabolic twin</h2><p class="hero-status">${esc(statement)}</p><p class="hero-summary">${state?'Your readings and personal history show what may happen next.':'Connect glucose data and build the history for your first forecast.'}</p><div class="premium-personal-pill">${h.icon('glucose')}<strong>${personalDeparture(state,h)}</strong></div><p class="premium-response">${state?.shadow.available?esc(state.divergence?.label||'Personal expectation available'):'Personal baseline learning'}</p></div><div class="premium-hero-visual"><div class="premium-flow-motion">${TwinFlow(state)}</div><div class="forecast-grid">${[30,60,90,120].map(m=>ForecastHorizonCard(state,m,h)).join('')}</div></div></section>`;
}
export function CurrentGlucoseCard(snapshot,state,h){
 const {fmt,date,glucoseTrend}=h,t=state?glucoseTrend(state):{label:'Unavailable',rate:'—',arrow:''};
 const live=snapshot.twin_sync?.status==='CURRENT'&&!h.replay;
 const reading=state?snapshot.observations.filter(e=>e.event_type==='CGM_READING'&&e.quality.status==='GOOD'&&e.payload.glucose_mg_dl===state.current_glucose&&Date.parse(e.observed_at)<=Date.parse(state.reference_time)&&Date.parse(e.received_at)<=Date.parse(state.reference_time)).at(-1):null;
 return `<section class="card current-card premium-current"><div class="card-title"><h2>${state&&!live?'Last observed':'Current'} glucose</h2>${h.icon('glucose')}</div><div class="metric">${fmt(state?.current_glucose)}<span class="unit">mg/dL</span></div><span class="premium-trend ${t.label.toLowerCase()}">${t.arrow} ${state?`${t.label} · ${t.rate} mg/dL/min`:'Awaiting forecast'}</span><div class="premium-observed"><span>Observed at</span><small>${date(reading?.observed_at||state?.reference_time)}</small></div><div class="peak-row"><span class="label">Predicted peak</span><strong>${fmt(state?.predicted_peak.glucose)} mg/dL</strong><small>${state?'~'+state.predicted_peak.approximate_horizon+' min':'Not available yet'}</small></div></section>`;
}
export function RiskRing(state,h,previousRisk){
 const spike=state?.spike,available=spike?.available,percent=available?spike.probability*100:null;
 const elevated=spike?.status==='Above frozen alert threshold';
 const label=available?(elevated?'Elevated':spike.status==='Below frozen alert threshold'?'Below alert threshold':spike.status.includes('180')?'Incident risk not applicable':h.human(spike.status)):'Risk unavailable';
 const circumference=2*Math.PI*49,offset=percent==null?circumference:circumference*(1-Math.max(0,Math.min(100,percent))/100);
 const from=Number.isFinite(previousRisk)?circumference*(1-previousRisk/100):offset;
 return `<section class="card risk-card premium-risk ${elevated?'is-elevated':''}"><div class="card-title"><h2>Spike risk · next 2 hours</h2>${h.icon('evidence')}</div><div class="risk-ring"><svg viewBox="0 0 120 120" aria-hidden="true"><circle class="risk-track" cx="60" cy="60" r="49"/><circle class="risk-arc" cx="60" cy="60" r="49" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}" style="--risk-from:${from};--risk-to:${offset}"/></svg><div class="metric">${available?h.fmt(percent)+'%':'—'}</div></div><span class="pill ${elevated?'coral':'neutral'}">${h.esc(label)}</span>${available?'<button class="text" data-action="risk-info">What contributes to this? →</button>':'<p class="label">A risk estimate appears when the required context is available.</p>'}</section>`;
}
export function PatientContextCard(snapshot,h){
 const info=snapshot.profile.confirmed_information||{},get=k=>info[k]?.value,p=snapshot.patient;
 const age=get('age_years')??p.age_years,dob=p.date_of_birth,sex=get('sex')??p.sex;
 const medications=get('medications')?.map(m=>m.name+(m.status==='discontinued'?' (previous)':'')).join(', ')||get('medication_context')&&h.human(get('medication_context'));
 const rows=[[dob?'Date of birth':'Age',dob?new Date(dob).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'})+(age!=null?` (${Math.floor(age)} years)`:''):age!=null?Math.floor(age)+' years':'Not recorded','timeline'],['Sex',sex&&sex!=='unknown'?h.human(sex):'Not recorded','patients'],['Diagnoses',get('diagnoses')?.join(', ')||'Not recorded','evidence'],['Diabetes duration',get('diabetes_duration_years')!=null?get('diabetes_duration_years')+' years':'Not recorded','timeline'],['Medications',medications||'Not recorded','data']];
 return `<section class="card patient-context premium-context"><div class="card-title"><h2>Patient context</h2><button class="text premium-edit" data-action="profile-edit" aria-label="Edit confirmed patient profile">Edit</button></div><dl>${rows.map(([label,value,icon])=>`<div>${h.icon(icon)}<dt>${label}</dt><dd>${h.esc(value)}</dd></div>`).join('')}</dl><div class="premium-context-actions"><a href="${h.link('data')}?section=records" data-route>View full medical history →</a><button class="text" data-action="create" aria-label="Create New Twin">+ New Twin</button></div></section>`;
}
export function WearableSummary(snapshot,h){
 const c=snapshot.connections.find(c=>c.source_type==='WEARABLE');
 return `<section class="card premium-wearable"><div class="card-title"><h2>Wearable</h2><span class="pill ${c?.status==='CONNECTED'?'':'neutral'}">${h.human(c?.status)}</span></div><div data-product-wearable="context"></div><a href="${h.link('wearable')}" data-route>View wearable data →</a></section>`;
}
export function GlucoseTrajectory(snapshot,state,h){
 return `<section class="card center-trajectory premium-trajectory"><div class="card-title"><h2>${h.icon('glucose')}Glucose trajectory</h2><span class="label">Past 2h · Next 2h</span></div><p class="label">${state?.shadow.available?'Observed readings, forecast and your personal expected trajectory.':'Observed readings'+(state?' and your glucose forecast.':'. Your forecast appears when enough history is available.')}</p>${h.chart(snapshot,state,false,{premium:true})}</section>`;
}
export function PersonalBaselineCard(state,h){
 return `<section class="card premium-baseline"><div class="card-title"><h2>${h.icon('glucose')}Personal baseline</h2><a href="${h.link('data')}" data-route aria-label="View personal baseline data">›</a></div><p>${state?.shadow.available?personalDeparture(state,h)+'.':'Learning your personal pattern from your glucose history.'}</p><span class="pill ${state?.shadow.available?'':'neutral'}">${state?.shadow.available?h.esc(state.divergence?.label||'Personal baseline available'):'Learning'}</span></section>`;
}
export function WhyNowCard(state,h){
 return `<section class="card why-card premium-why"><div class="card-title"><h2>${h.icon('evidence')}Why now?</h2><a href="${h.link('evidence')}" data-route aria-label="View prediction evidence">›</a></div><p class="label">The readings and comparisons behind this prediction.</p>${state?`<ul class="why-list">${h.whyItems(state)}</ul><div class="why-footer"><button data-action="why" aria-label="Understand this prediction">Understand this prediction →</button></div>`:'<p class="no-data">Your reasons will appear with the first forecast.</p>'}</section>`;
}
export function premiumOverview(snapshot,state,h,previous={}){
 const changed=previous.patient===snapshot.patient_id&&previous.prediction&&previous.prediction!==state?.prediction_id;
 const alert=snapshot.active_alert.alert,newAlert=previous.patient===snapshot.patient_id&&alert?.state==='ACTIVE'&&previous.alert!==alert.alert_id;
 const pending=!h.replay&&state&&snapshot.twin_sync?.status!=='CURRENT';
 return `<h1 class="premium-sr-only">Overview</h1>${h.replay?'<div class="inline-info">Prediction history · Viewing a saved forecast. <button class="text" data-action="exit-replay">Return to current Twin</button></div>':''}${pending?`<div class="inline-info" role="status">${snapshot.twin_sync?.status==='UPDATING'?'Updating Twin… Showing the last complete forecast until the latest reading is processed.':'Prediction awaiting sufficient data. Showing the last complete forecast with its original time.'}</div>`:''}${!h.replay?h.alertSurface():''}<div class="overview-grid premium-overview ${changed?'premium-new-state':''} ${newAlert?'premium-alert-arrival':''}" data-twin-state-id="${h.esc(state?.twin_state_id)}" data-prediction-id="${h.esc(state?.prediction_id)}" data-reference-time="${h.esc(state?.reference_time)}"><div class="premium-primary">${MetabolicTwinHero(snapshot,state,h)}${CurrentGlucoseCard(snapshot,state,h)}${RiskRing(state,h,previous.patient===snapshot.patient_id?previous.risk:undefined)}</div><div class="premium-secondary"><div class="premium-context-column">${PatientContextCard(snapshot,h)}${WearableSummary(snapshot,h)}</div>${GlucoseTrajectory(snapshot,state,h)}<div class="premium-right">${PersonalBaselineCard(state,h)}${WhyNowCard(state,h)}</div></div></div>${!state?`<details class="card readiness-details"><summary>Setup and forecast readiness</summary>${h.setup()}</details>`:''}`;
}
