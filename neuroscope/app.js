const video=document.querySelector('#brain');
const motion=document.querySelector('#motion');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function syncMotion(){motion.textContent=video.paused?'▷':'Ⅱ';motion.setAttribute('aria-label',video.paused?'Play brain animation':'Pause brain animation');}
function videoUnavailable(){document.querySelector('#video-error').hidden=false;motion.hidden=true;}
function startNerveLoop(){if(reduced.matches||video.loop)return;video.loop=true;video.querySelector('source').src=video.dataset.loopSrc;video.load();video.play().catch(videoUnavailable);}
if(reduced.matches){video.autoplay=false;video.pause();}
video.addEventListener('play',syncMotion);video.addEventListener('pause',syncMotion);
video.addEventListener('ended',startNerveLoop);
motion.addEventListener('click',()=>{if(video.paused)video.play().catch(()=>{document.querySelector('#video-error').hidden=false;});else video.pause();});
video.addEventListener('error',videoUnavailable);
video.querySelector('source').addEventListener('error',videoUnavailable);
syncMotion();
const menu=document.querySelector('.menu'),nav=document.querySelector('#navigation');
menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open);});
nav.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>{menu.setAttribute('aria-expanded','false');nav.classList.remove('open');}));
const dialog=document.querySelector('#workspace');
document.querySelectorAll('[data-start]').forEach(button=>button.addEventListener('click',()=>window.location.href='upload.html'));
document.querySelectorAll('.close,.done').forEach(button=>button.addEventListener('click',()=>dialog.close()));
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
document.querySelector('#scan').addEventListener('change',event=>{const file=event.target.files[0],status=document.querySelector('#file-status');if(!file){status.textContent='Use de-identified research data only.';return;}if(!/\.(nii|nii\.gz|dcm)$/i.test(file.name)){status.textContent='Please choose a .nii, .nii.gz, or .dcm file.';event.target.value='';return;}status.textContent=`Selected: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB). Demo only — no data has been uploaded or analyzed.`;});
