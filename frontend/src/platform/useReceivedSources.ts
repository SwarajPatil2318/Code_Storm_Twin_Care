import {useEffect,useRef,useState} from 'react';
import type {Observation} from '../types';
const sourceOf:Record<string,string>={heart_rate:'Smartwatch',hrv_rmssd:'Smartwatch',steps:'Smartwatch',sleep_hours:'Smartwatch',systolic_bp:'BP cuff',diastolic_bp:'BP cuff',weight:'Smart scale',spo2:'Pulse oximeter'};
export function useReceivedSources(observations:Observation[],cutoff:string) {
 const previous=useRef<{cutoff:number;patient?:string}|null>(null);
 const [active,setActive]=useState<string[]>([]);
 useEffect(()=>{
  const now=Date.parse(cutoff),patient=observations[0]?.patient_id,prev=previous.current;
  previous.current={cutoff:now,patient};
  if(!prev||prev.patient!==patient||now<=prev.cutoff){setActive([]);return;}
  setActive([...new Set(observations.filter(o=>o.value.usable_value!==null&&o.available_time&&o.observed_time&&Date.parse(o.available_time)>prev.cutoff&&Date.parse(o.available_time)<=now&&Date.parse(o.observed_time)<=now).map(o=>sourceOf[o.concept]).filter(Boolean))]);
  const timer=setTimeout(()=>setActive([]),900);return()=>clearTimeout(timer);
 },[observations,cutoff]);
 return active;
}
