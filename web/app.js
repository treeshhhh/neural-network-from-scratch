'use strict';
const $=id=>document.getElementById(id);
const canvas=$('drawing'), ctx=canvas.getContext('2d',{willReadFrequently:true});
const preview=$('processed').getContext('2d');
let drawing=false, pointer=null, timer=null, version=0, request=null, hasInk=false;
const rows=[];
for(let digit=0;digit<10;digit++){
  const row=document.createElement('div');row.className='prob-row';
  const label=document.createElement('span');label.textContent=digit;
  const track=document.createElement('div');track.className='track';
  const fill=document.createElement('div');fill.className='fill';track.append(fill);
  const value=document.createElement('span');value.className='prob-value';value.textContent='—';
  row.append(label,track,value);$('probabilities').append(row);rows.push({row,fill,value});
}
function state(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function abortPending(){clearTimeout(timer);timer=null;version++;if(request)request.abort();request=null;$('predict').disabled=false;}
function resetResult(){
  $('digit').textContent='—';$('confidence').textContent='Waiting for your next digit';
  preview.fillStyle='#000';preview.fillRect(0,0,28,28);
  rows.forEach(r=>{r.fill.style.width='0%';r.value.textContent='—';r.row.classList.remove('best');});
}
function clear(){
  if(pointer!==null&&canvas.hasPointerCapture(pointer))canvas.releasePointerCapture(pointer);
  drawing=false;pointer=null;abortPending();hasInk=false;
  ctx.fillStyle='#000';ctx.fillRect(0,0,280,280);$('canvasHint').hidden=false;
  resetResult();state('Draw a single digit to get started.');
}
function point(event){const rect=canvas.getBoundingClientRect();return [(event.clientX-rect.left)*280/rect.width,(event.clientY-rect.top)*280/rect.height];}
canvas.addEventListener('pointerdown',event=>{
  if(drawing||!event.isPrimary||(event.pointerType==='mouse'&&event.button!==0))return;
  event.preventDefault();abortPending();drawing=true;pointer=event.pointerId;hasInk=true;
  canvas.setPointerCapture(pointer);$('canvasHint').hidden=true;
  resetResult();state('Drawing…');const [x,y]=point(event);
  ctx.strokeStyle='#fff';ctx.fillStyle='#fff';ctx.lineWidth=18;ctx.lineCap='round';ctx.lineJoin='round';
  ctx.beginPath();ctx.arc(x,y,9,0,2*Math.PI);ctx.fill();ctx.beginPath();ctx.moveTo(x,y);
});
canvas.addEventListener('pointermove',event=>{
  if(!drawing||event.pointerId!==pointer)return;
  event.preventDefault();const samples=event.getCoalescedEvents?event.getCoalescedEvents():[];
  for(const sample of samples.length?samples:[event]){const [x,y]=point(sample);ctx.lineTo(x,y);}ctx.stroke();
});
function finish(event){
  if(!drawing||event.pointerId!==pointer)return;
  drawing=false;pointer=null;state('Ready. Predict when you’re happy with your digit.');
  if($('auto').checked){clearTimeout(timer);timer=setTimeout(predict,350);}
}
canvas.addEventListener('pointerup',finish);canvas.addEventListener('pointercancel',finish);canvas.addEventListener('lostpointercapture',finish);
$('clear').onclick=clear;
$('auto').onchange=()=>{clearTimeout(timer);if($('auto').checked&&hasInk&&!drawing)timer=setTimeout(predict,350);};
$('predict').onclick=predict;
async function predict(){
  if(drawing)return;
  if(!hasInk){state('The canvas is empty. Draw a digit first.');return;}
  abortPending();const myVersion=version;const controller=new AbortController();request=controller;
  $('predict').disabled=true;state('Processing your drawing…');
  const rgba=ctx.getImageData(0,0,280,280).data;
  const pixels=Array.from({length:280*280},(_,i)=>rgba[i*4]);
  const timeout=setTimeout(()=>controller.abort(),10000);
  try{
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({width:280,height:280,pixels}),signal:controller.signal});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Prediction failed.');
    if(myVersion!==version)return;
    if(result.blank){resetResult();state('No visible ink found. Try a larger stroke.');return;}
    const image=preview.createImageData(28,28);
    result.processed.flat().forEach((value,i)=>{const v=Math.round(value*255);image.data[i*4]=v;image.data[i*4+1]=v;image.data[i*4+2]=v;image.data[i*4+3]=255;});
    preview.putImageData(image,0,0);
    $('digit').textContent=result.digit;$('confidence').textContent=`${(result.confidence*100).toFixed(1)}% model confidence`;
    result.probabilities.forEach((p,i)=>{rows[i].fill.style.width=`${p*100}%`;rows[i].value.textContent=`${(p*100).toFixed(1)}%`;rows[i].row.classList.toggle('best',i===result.digit);});
    state('Prediction ready. The preview shows the exact normalized model input, displayed as grayscale.');
  }catch(error){
    if(myVersion!==version)return;
    state(error.name==='AbortError'?'Prediction timed out. Please try again.':'Could not predict. Check that the local server is running, then try again.',true);
  }finally{clearTimeout(timeout);if(myVersion===version){request=null;$('predict').disabled=false;}}
}
clear();
