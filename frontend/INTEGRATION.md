# Integrated NeuroScope interface

Run the project with `./start-neuroscope.sh` (or the existing Windows launcher).

- `/` serves the original `frontend/index.html` with the frontend branch's styles.
- `/workspace` serves the functional MRI analyzer with `src/web/workspace.css`.
- The homepage's Start Analysis and Launch workspace links open `/workspace`.
- Existing `/?analysis=...` links redirect to `/workspace?analysis=...`.
- Uploads, sample cases, model selection, job history, NiiVue, reviews and exports
  continue to use the existing `/api/` routes and production application script.

The light lavender workspace shell shares the landing page's palette and typography.
The MRI canvas stays dark and its segmentation colours are unchanged.
The original homepage layout and video controls are preserved. Its 16:9 hero video
is served from `neuroscope/assets/neuroport-brain.mp4` through `/site/assets/`.
It is committed with the project so the animation is available after a fresh clone.
Launch buttons open the real workspace instead of the old demo dialog.
The adapted `home.html` is no longer the application entry point.

Verification: in-process HTTP route/asset checks, legacy analysis-link redirect,
workspace JavaScript element-ID contract, JavaScript syntax and existing upload tests.
Browser visual verification was blocked by a saved localhost browser permission.
