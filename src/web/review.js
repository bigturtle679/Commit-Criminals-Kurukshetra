// Review state is deliberately separate from immutable inference results.
export function accessFor(attempt, hasExpert) {
  const prediction = !attempt || ['prediction', 'expert'].includes(attempt.stage);
  return { prediction, expert: hasExpert && (!attempt || attempt.stage === 'expert'),
    report: prediction && (!hasExpert || !attempt || attempt.stage === 'expert') };
}

export function worldToVoxel(world, affine, shape) {
  if (!affine || !shape || world.length !== 3 || !world.every(Number.isFinite)) throw Error('Original image geometry is unavailable.');
  // Solve the full affine system, including rotation, flips, shear and translation.
  const rows = affine.slice(0, 3).map((r, i) => [r[0], r[1], r[2], world[i] - r[3]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(rows[r][col]) > Math.abs(rows[pivot][col])) pivot = r;
    [rows[col], rows[pivot]] = [rows[pivot], rows[col]];
    const scale = rows[col][col];
    if (Math.abs(scale) < 1e-12) throw Error('Invalid original image affine.');
    rows[col] = rows[col].map(x => x / scale);
    for (let r = 0; r < 3; r++) if (r !== col) {
      const factor = rows[r][col];
      rows[r] = rows[r].map((x, c) => x - factor * rows[col][c]);
    }
  }
  const voxel = rows.map(r => r[3]);
  if (voxel.some((x, i) => !Number.isFinite(x) || x < -1e-5 || x > shape[i] - 1 + 1e-5)) throw Error('Location lies outside the original image volume.');
  return voxel;
}

export async function caseKey(hashes) {
  const order = ['t1', 't1c', 't2', 'flair'];
  if (!hashes || !order.every(m => /^[a-f0-9]{64}$/.test(hashes[m] || ''))) return null;
  const bytes = new TextEncoder().encode(order.map(m => `${m}:${hashes[m]}`).join('|'));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
}

const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const now = () => new Date().toISOString();
const uid = () => crypto.randomUUID();
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); return true; } catch { return false; } }
};

export class ReviewController {
  constructor(hooks) {
    this.hooks = hooks;
    this.document = null;
    this.pending = null;
    this.loading = true;
    this.saving = false;
    this.jobId = null;
    this.editId = null;
    $('bookmark-form').onsubmit = e => { e.preventDefault(); this.saveBookmark(); };
    $('bookmark-cancel').onclick = () => { this.clearForm(); this.render(); };
    $('start-blind').onclick = () => this.jobId ? this.startAttempt() : this.arm();
    $('submit-attempt').onclick = () => this.submit(false);
    $('submit-empty').onclick = () => this.submit(true);
    $('reveal-prediction').onclick = () => this.advance('prediction');
    $('reveal-expert').onclick = () => this.advance('expert');
    $('exit-blind').onclick = () => this.mutate(doc => { doc.active_attempt_id = null; });
    $('save-reflection').onclick = () => this.mutate(doc => { this.findAttempt(doc).reflection = $('reflection').value; this.findAttempt(doc).updated_at = now(); }, false);
    $('retry-review').onclick = async () => {
      if (this.pending) await this.persist(this.pending);
      if (!this.pending) await this.hooks.reopen();
    };
    $('reload-review').onclick = () => this.hooks.reopen();
    $('export-review').onclick = () => this.export();
    $('attempt-history').onchange = () => this.mutate(doc => { doc.active_attempt_id = $('attempt-history').value || null; });
    $('blind-next').checked = storage.get('neuroscope-blind-next') === 'true';
    $('blind-next').onchange = () => storage.set('neuroscope-blind-next', String($('blind-next').checked));
    this.render();
  }
  arm() {
    $('blind-next').checked = true;
    storage.set('neuroscope-blind-next', 'true');
    $('review-message').textContent = 'Blind mode armed. Choose a case or upload scans.';
  }
  get attempt() { return this.document ? this.findAttempt(this.document) : null; }
  findAttempt(doc) { return doc.attempts.find(a => a.id === doc.active_attempt_id) || null; }
  get access() {
    if (this.loading || !this.document) return {prediction:false, expert:false, report:false};
    const current = accessFor(this.attempt, this.document.metadata.has_expert);
    if (!this.pending) return current;
    const next = accessFor(this.findAttempt(this.pending), this.document.metadata.has_expert);
    return Object.fromEntries(Object.keys(current).map(key => [key, current[key] && next[key]]));
  }
  async load(jobId, blind) {
    if (this.saving) {
      this.loading = true;
      throw Error('A review save is still running. Retry opening this analysis after it finishes.');
    }
    this.loading = true;
    this.jobId = jobId;
    this.document = null;
    this.pending = null;
    this.key = null;
    this.clearForm();
    this.render();
    const doc = await this.hooks.fetch(`/api/jobs/${jobId}/review`);
    const key = await caseKey(doc.metadata.input_hashes);
    if (this.jobId !== jobId) return;
    this.document = doc;
    this.key = key;
    this.loading = false;
    if (blind && !this.attempt) {
      await this.startAttempt(false);
      if (!this.attempt) throw Error('Blind attempt was not saved. Retry before opening this analysis.');
    }
    this.render();
  }
  async startAttempt(refresh = true) {
    if (!this.document || this.loading || this.saving || this.pending) return;
    const seen = this.key && storage.get(`neuroscope-exposed:${this.key}`);
    const exposure = !this.key || seen === null && storage.get('neuroscope-storage-probe') === null ? 'unknown' : seen ? 'practice_replay' : 'unseen';
    await this.mutate(doc => {
      const date = now();
      const attempt = {id:uid(), stage:'inspect', exposure, created_at:date, updated_at:date,
        bookmarks:[], submitted_at:null, submitted_bookmarks:null, no_areas:false, reflection:''};
      doc.attempts.push(attempt);
      doc.active_attempt_id = attempt.id;
    }, refresh);
  }
  async mutate(change, refresh = true) {
    if (!this.document || this.saving || this.pending) return;
    const next = clone(this.document);
    try { change(next); } catch (e) { this.error(e); return; }
    await this.persist(next, refresh);
  }
  async persist(next, refresh = true) {
    if (this.saving) return;
    this.saving = true;
    this.pending = next;
    this.render();
    const jobId = this.jobId;
    try {
      const saved = await this.hooks.fetch(`/api/jobs/${jobId}/review`, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(next)});
      if (this.jobId !== jobId) return;
      this.document = saved;
      this.loading = false;
      this.pending = null;
      $('review-message').textContent = 'Review saved.';
      this.clearForm();
    } catch (e) {
      if (this.jobId === jobId) this.error(e);
    } finally {
      this.saving = false;
      this.render();
    }
    if (!this.pending && refresh && this.jobId === jobId) await this.hooks.refresh();
  }
  error(error) { $('review-message').textContent = error.message; }
  clearForm() {
    this.editId = null;
    $('bookmark-title').value = '';
    $('bookmark-note').value = '';
    $('bookmark-save').textContent = 'Save current location';
  }
  async saveBookmark() {
    const attempt = this.attempt;
    if (attempt && attempt.stage !== 'inspect') return;
    const title = $('bookmark-title').value.trim(), note = $('bookmark-note').value;
    if (!title || title.length > 100 || note.length > 2000) return this.error(Error('Enter a title (up to 100 characters) and a note (up to 2,000).'));
    try {
      const old = this.currentBookmarks().find(b => b.id === this.editId);
      const settings = old || this.hooks.capture();
      const bookmark = {...settings, id:old?.id || uid(), title, note, created_at:old?.created_at || now(), updated_at:now()};
      const voxel = worldToVoxel(bookmark.world_mm, this.document.metadata.affine, this.document.metadata.shape);
      bookmark.voxel = voxel;
      await this.mutate(doc => {
        const list = this.findAttempt(doc)?.bookmarks || doc.bookmarks;
        const index = list.findIndex(b => b.id === bookmark.id);
        if (index < 0) list.push(bookmark); else list[index] = bookmark;
      }, false);
    } catch (e) { this.error(e); }
  }
  currentBookmarks() { return this.attempt?.bookmarks || this.document?.bookmarks || []; }
  async submit(empty) {
    if (!this.attempt || this.attempt.stage !== 'inspect') return;
    if (empty && this.attempt.bookmarks.length) return;
    if (!empty && !this.attempt.bookmarks.length) return;
    await this.mutate(doc => {
      const a = this.findAttempt(doc);
      a.stage = 'submitted'; a.submitted_at = now(); a.updated_at = now();
      a.submitted_bookmarks = clone(a.bookmarks); a.no_areas = empty;
    }, false);
  }
  async advance(stage) {
    if (!this.attempt || (stage === 'prediction' && this.attempt.stage !== 'submitted') || (stage === 'expert' && this.attempt.stage !== 'prediction')) return;
    await this.mutate(doc => { const a = this.findAttempt(doc); a.stage = stage; a.updated_at = now(); }, false);
    if (!this.pending && this.attempt?.stage === stage) await this.hooks.reveal(stage);
  }
  markExposure() {
    if (this.key && this.access.prediction) storage.set(`neuroscope-exposed:${this.key}`, 'true');
  }
  render() {
    const ready = !!this.document && !this.loading;
    const attempt = this.attempt;
    const blocked = !ready || this.saving || !!this.pending;
    $('review-stage').textContent = this.loading ? 'Open an analysis to review' : !attempt ? 'Normal viewing' :
      ({inspect:'Inspect', submitted:'Submitted', prediction:'Prediction revealed', expert:'Expert revealed'})[attempt.stage];
    $('review-exposure').textContent = !attempt ? '' : ({unseen:'No recorded exposure in this browser',practice_replay:'Practice replay — answers previously viewed',unknown:'Prior exposure unknown'})[attempt.exposure];
    $('start-blind').disabled = this.saving || !!this.pending;
    $('start-blind').textContent = attempt ? 'Start another blind attempt' : 'Start blind attempt';
    $('exit-blind').hidden = !attempt;
    $('exit-blind').disabled = blocked;
    $('submit-attempt').hidden = attempt?.stage !== 'inspect' || !attempt.bookmarks.length;
    $('submit-empty').hidden = attempt?.stage !== 'inspect' || !!attempt.bookmarks.length;
    $('reveal-prediction').hidden = attempt?.stage !== 'submitted';
    $('reveal-expert').hidden = attempt?.stage !== 'prediction' || !this.document?.metadata.has_expert;
    for (const id of ['submit-attempt','submit-empty','reveal-prediction','reveal-expert','export-review','save-reflection']) $(id).disabled = blocked;
    $('no-expert').hidden = !ready || this.document.metadata.has_expert;
    $('reflection-wrap').hidden = !attempt || attempt.stage === 'inspect';
    if (document.activeElement !== $('reflection')) $('reflection').value = (this.pending ? this.findAttempt(this.pending)?.reflection : attempt?.reflection) || '';
    $('retry-review').hidden = !this.jobId || (!this.pending && !this.loading);
    $('reload-review').hidden = !this.pending;
    $('retry-review').disabled = this.saving;
    $('reload-review').disabled = this.saving;
    const immutable = attempt && attempt.stage !== 'inspect';
    $('bookmark-title').disabled = blocked || immutable;
    $('bookmark-note').disabled = blocked || immutable;
    $('bookmark-save').disabled = blocked || immutable || (!this.editId && !this.hooks.canCapture());
    $('bookmark-cancel').disabled = blocked;
    $('bookmark-list').replaceChildren();
    const list = this.currentBookmarks();
    for (const b of list) {
      const item = document.createElement('li');
      const jump = document.createElement('button');
      jump.type = 'button'; jump.textContent = b.title;
      jump.disabled = blocked || !this.hooks.canNavigate();
      jump.onclick = () => this.hooks.navigate(b).catch(e => this.error(e));
      const position = document.createElement('small');
      position.textContent = `${b.world_mm.map(x => x.toFixed(1)).join(', ')} mm · voxel ${b.voxel.map(x => x.toFixed(1)).join(', ')} · ${b.modality.toUpperCase()}`;
      const note = document.createElement('p'); note.textContent = b.note;
      item.append(jump, position, note);
      if (!immutable) {
        const edit = document.createElement('button'); edit.type = 'button'; edit.textContent = 'Edit note'; edit.disabled = blocked;
        edit.onclick = () => { this.editId = b.id; $('bookmark-title').value = b.title; $('bookmark-note').value = b.note; $('bookmark-save').textContent = 'Save note changes'; this.render(); };
        const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Delete'; remove.disabled = blocked;
        remove.onclick = () => this.mutate(doc => {
          const target = this.findAttempt(doc) || doc;
          target.bookmarks = target.bookmarks.filter(x => x.id !== b.id);
        }, false);
        item.append(edit, remove);
      }
      $('bookmark-list').append(item);
    }
    $('bookmark-kind').textContent = !attempt ? 'Ordinary case notes' : immutable ? 'Submitted blind observations — locked' : 'Draft blind observations';
    const history = $('attempt-history'); history.replaceChildren(new Option('Ordinary case notes', ''));
    (this.document?.attempts || []).forEach((a, i) => history.add(new Option(`Attempt ${i + 1} · ${a.stage}`, a.id)));
    history.value = this.document?.active_attempt_id || '';
    history.disabled = blocked;
    this.hooks.gate?.();
  }
  export() {
    if (!this.document || this.pending) return;
    const blob = new Blob([JSON.stringify({job_id:this.jobId, ...this.document}, null, 2)], {type:'application/json'});
    const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `review-${this.jobId}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
}

// Failure to persist browser history is represented as unknown exposure.
storage.set('neuroscope-storage-probe', 'ok');
