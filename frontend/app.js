const video=document.querySelector('#brain');
const motion=document.querySelector('#motion');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function syncMotion(){motion.textContent=video.paused?'▷':'Ⅱ';motion.setAttribute('aria-label',video.paused?'Play brain animation':'Pause brain animation');}
function startNerveLoop(){if(reduced.matches||video.loop)return;video.loop=true;video.querySelector('source').src=video.dataset.loopSrc;video.load();video.play().catch(videoUnavailable);}
if(reduced.matches){video.autoplay=false;video.pause();}
video.addEventListener('play',syncMotion);video.addEventListener('pause',syncMotion);
video.addEventListener('ended',startNerveLoop);
motion.addEventListener('click',()=>{if(video.paused)video.play().catch(()=>{document.querySelector('#video-error').hidden=false;});else video.pause();});
function videoUnavailable(){document.querySelector('#video-error').hidden=false;motion.hidden=true;}
video.addEventListener('error',videoUnavailable);
video.querySelector('source').addEventListener('error',videoUnavailable);
syncMotion();
const menu=document.querySelector('.menu'),nav=document.querySelector('#navigation');
menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));nav.classList.toggle('open',open);});
nav.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>{menu.setAttribute('aria-expanded','false');nav.classList.remove('open');}));
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
