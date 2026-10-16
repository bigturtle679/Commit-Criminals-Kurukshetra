const video=document.querySelector('#brain');
const motion=document.querySelector('#motion');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function syncMotion(){motion.textContent=video.paused?'▷':'Ⅱ';motion.setAttribute('aria-label',video.paused?'Play brain animation':'Pause brain animation');}
if(reduced.matches){video.autoplay=false;video.pause();}
video.addEventListener('play',syncMotion);video.addEventListener('pause',syncMotion);
motion.addEventListener('click',()=>{if(video.paused)video.play().catch(()=>{document.querySelector('#video-error').hidden=false;});else video.pause();});
video.addEventListener('error',()=>{document.querySelector('#video-error').hidden=false;motion.hidden=true;});
syncMotion();
const menu=document.querySelector('.menu'),nav=document.querySelector('#navigation');
menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open);});
nav.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>{menu.setAttribute('aria-expanded','false');nav.classList.remove('open');}));
const dialog=document.querySelector('#workspace');
document.querySelectorAll('[data-start]').forEach(button=>button.addEventListener('click',()=>dialog.showModal()));
document.querySelectorAll('.close,.done').forEach(button=>button.addEventListener('click',()=>dialog.close()));
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
document.querySelector('#scan').addEventListener('change',event=>{const file=event.target.files[0],status=document.querySelector('#file-status');if(!file){status.textContent='Use de-identified research data only.';return;}if(!/\.(nii|nii\.gz|dcm)$/i.test(file.name)){status.textContent='Please choose a .nii, .nii.gz, or .dcm file.';event.target.value='';return;}status.textContent=`Selected: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB). Demo only — no data has been uploaded or analyzed.`;});

// Scroll-reveal: fade-in sections as they enter viewport
const revealTargets = document.querySelectorAll('.section, .disclaimer');
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('revealed'); io.unobserve(e.target); }
  });
}, { threshold: 0.12 });
revealTargets.forEach(el => { el.classList.add('reveal'); io.observe(el); });

// Navbar: add shadow when scrolled past hero
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });
