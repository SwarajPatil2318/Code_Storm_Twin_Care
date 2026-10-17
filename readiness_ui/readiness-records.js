// Preserve P2 record workflow; select the explicitly requested owner on P5 navigation.
const requested=new URLSearchParams(location.search).get('patient_id');let selected=!requested;
const link=document.createElement('a');link.textContent='Your Twin setup ↗';document.querySelector('aside').append(link);
function update(){const active=document.querySelector('#patient-list button.active');if(active)link.href='/twin-setup?patient_id='+encodeURIComponent(active.dataset.patient);else link.href='/twin-setup';if(selected)return;const buttons=[...document.querySelectorAll('#patient-list button')];const target=buttons.find(b=>b.dataset.patient===requested);if(target&&!target.disabled){selected=true;if(!target.classList.contains('active'))target.click();}}
new MutationObserver(update).observe(document.querySelector('#patient-list'),{childList:true,subtree:true,attributes:true});update();
