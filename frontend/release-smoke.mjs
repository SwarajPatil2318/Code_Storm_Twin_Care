import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const base=process.env.TWINCARE_URL||'http://127.0.0.1:8113';
const out=new URL('../data/release/browser/',import.meta.url);await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
const page=await browser.newPage();const errors=[];
page.on('pageerror',e=>errors.push(e.name));
try{
 for(const [label,url]of [['patients','/patients'],['cardiovascular','/twin/cardiovascular'],['metabolic','/twin/metabolic']]){
  await page.goto(base+url);await page.waitForLoadState('networkidle');
  assert.equal(await page.locator('body').evaluate(e=>e.scrollWidth>innerWidth),false);
  assert.ok((await page.locator('body').innerText()).includes('TwinCare'));
  await page.screenshot({path:new URL(label+'.png',out).pathname,fullPage:true});
 }
 for(const width of [1440,768,390]){
  await page.setViewportSize({width,height:900});await page.goto(base+'/patients');await page.waitForLoadState('networkidle');
  assert.equal(await page.locator('body').evaluate(e=>e.scrollWidth>innerWidth),false);
 }
 assert.deepEqual(errors,[]);console.log('Release browser smoke passed: three product routes and desktop/tablet/mobile patient shell.');
}finally{await browser.close();}
