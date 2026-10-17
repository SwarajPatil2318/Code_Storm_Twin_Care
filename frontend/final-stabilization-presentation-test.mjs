import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile('../experience_ui/presentation.js','utf8');
const {trend,duration,newerSnapshot}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const config=JSON.parse(await readFile('../configs/metabolic_d3.json','utf8')),threshold=config.rising_display_slope_mg_dl_per_minute;
const cases=[0,0.0001,-0.0001,-0,threshold,threshold+0.1,-threshold-0.1];
const expected=['Stable','Stable','Stable','Stable','Stable','Rising','Falling'];
for(let i=0;i<cases.length;i++){const t=trend(cases[i],threshold);assert.equal(t.label,expected[i]);assert.notEqual(t.rate,'-0.0');if(t.label!=='Stable')assert.notEqual(t.rate,'0.0');}
assert.equal(duration(59.9),'59 min');assert.equal(duration(90),'1 hr 30 min');assert.equal(duration(1440),'1 day');assert.equal(duration(11460),'7 days 23 hr');
const at=t=>({evaluated_at:t,twin:{twin_state:{reference_time:t}}});assert(!newerSnapshot(at('2026-10-20T08:00:00Z'),at('2026-10-20T03:15:00Z')));
const app=await readFile('../experience_ui/app.js','utf8');assert(app.includes('glucoseTrend(s).label')||source.includes('glucoseTrend(state).label'));assert(!app.includes('slope.value>0'));assert(!app.includes('wearable observations provide recent context'));
console.log(JSON.stringify({passed:true,trendCases:cases.length,durationCases:4,staleResponsesRejected:true,thresholdSource:'configs/metabolic_d3.json'}));
