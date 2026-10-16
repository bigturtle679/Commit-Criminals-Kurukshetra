# Integrated NeuroScope interface

Run the project with `./start-neuroscope.sh` (or the existing Windows launcher).

- `/` serves `frontend/home.html`, using the frontend branch's styles.
- `/workspace` serves the functional MRI analyzer with `src/web/workspace.css`.
- The homepage's Start Analysis and Launch workspace links open `/workspace`.
- Existing `/?analysis=...` links redirect to `/workspace?analysis=...`.
- Uploads, sample cases, model selection, job history, NiiVue, reviews and exports
  continue to use the existing `/api/` routes and production application script.

The light lavender workspace shell shares the landing page's palette and typography.
The MRI canvas stays dark and its segmentation colours are unchanged.
The homepage uses a bundled decorative SVG because the original MP4 is not in Git.
The branch's standalone `index.html` remains as its original prototype; the server
uses the integrated `home.html` entry point, which requires no demo dialog or video.

Verification: in-process HTTP route/asset checks, legacy analysis-link redirect,
workspace JavaScript element-ID contract, JavaScript syntax and existing upload tests.
Browser visual verification was blocked by a saved localhost browser permission.
