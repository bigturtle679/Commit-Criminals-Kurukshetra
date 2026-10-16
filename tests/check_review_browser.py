"""Exercise the real UI/NiiVue and review API with generated MRI fixtures, no inference."""
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import uvicorn
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
stub=types.ModuleType('pipeline'); stub.ROOT=ROOT; stub.run_case=lambda *args,**kwargs: None
spec=importlib.util.spec_from_file_location('browser_fixture_app',ROOT/'src/app.py')
api=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'pipeline':stub}): spec.loader.exec_module(api)

def fake_status(): return {'gpu':'Generated test fixture','default_model':'ensemble','demo_ready':False}
for route in api.app.routes:
    if getattr(route,'path','')=='/api/status': route.endpoint=fake_status; route.dependant.call=fake_status

temporary=tempfile.TemporaryDirectory(dir=ROOT/'work')
api.RUNS=Path(temporary.name); api.JOBS={}
affine=np.array([[0,-2,0,40],[3,0,0,-20],[0,0,-4,80],[0,0,0,1]],dtype=float)
shape=(32,32,32)
for name,expert,old in [('expert',True,False),('noexpert',False,False),('old',False,True)]:
    folder=api.RUNS/name; folder.mkdir()
    image=np.random.default_rng(1).normal(100,20,shape).astype(np.float32)
    for modality in (('t1c','flair') if old else ('t1','t1c','t2','flair')):
        nib.save(nib.Nifti1Image(image,affine),folder/f'{modality}.nii.gz')
    labels=np.zeros(shape,dtype=np.uint8)
    if not old: labels[12:20,12:20,12:20]=4
    nib.save(nib.Nifti1Image(labels,affine),folder/'segmentation.nii.gz')
    if expert: nib.save(nib.Nifti1Image(labels,affine),folder/'expert.nii.gz')
    result=dict(case_name=f'Fixture {name}',model='Fixture model',model_name='ensemble',volumes_ml=dict(WT=12.3,TC=8.2,ET=4.1,ED=4.1),
                shape=list(shape),spacing_mm=[3,2,4],runtime_seconds=0.1,peak_gpu_mib=0,meshes=[],metrics={'WT':dict(dice=.8,sensitivity=.9)} if expert else None,
                affine=affine.tolist(),input_hashes={m:hex(i+10)[2:]*64 for i,m in enumerate(('t1','t1c','t2','flair'))} if not old else None)
    if old: result.pop('affine')
    if old: result['volumes_ml'] = dict(WT=0,TC=0,ET=0,ED=0)
    (folder/'report.json').write_text(json.dumps(result))
    api.JOBS[name]=dict(id=name,status='complete',message='Fixture complete',result=result)

server=uvicorn.Server(uvicorn.Config(api.app,host='127.0.0.1',port=8127,log_level='error'))
thread=threading.Thread(target=server.run,daemon=True);thread.start()
for _ in range(100):
    if server.started: break
    time.sleep(.05)
assert server.started
errors=[]
try:
 with sync_playwright() as p:
    expect.set_options(timeout=20000)
    browser=p.chromium.launch(executable_path=os.environ.get('NEUROSCOPE_TEST_BROWSER', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),headless=True,args=['--use-angle=swiftshader','--enable-unsafe-swiftshader','--no-sandbox'])
    ctx=browser.new_context(viewport={'width':1440,'height':1100})
    ctx.route('https://fonts.googleapis.com/**',lambda route:route.abort())
    ctx.route('https://fonts.gstatic.com/**',lambda route:route.abort())
    page=ctx.new_page(); page.on('pageerror',lambda error:errors.append(str(error)))
    requests=[];page.on('request',lambda r:requests.append(r.url))
    page.goto('http://127.0.0.1:8127');page.add_style_tag(content='.canvas-container{height:240px!important;min-height:240px!important}');expect(page.locator('#history-select option')).to_have_count(4)
    page.locator('#blind-next').check()
    page.locator('#history-select').select_option('expert')
    expect(page.locator('#review-stage')).to_have_text('Inspect')
    expect(page.locator('#bookmark-save')).to_be_enabled(timeout=20000)
    assert not any('segmentation.nii.gz' in r or 'expert.nii.gz' in r for r in requests)
    expect(page.locator('.metrics-grid')).to_be_hidden()
    expect(page.locator('#link-download-report')).to_be_hidden()
    expect(page.locator('#select-overlay option[value=expert]')).to_have_js_property('disabled',True)
    expect(page.locator('#review-exposure')).to_contain_text('No recorded exposure')
    # Actual WebGL capture, script-like note text, modality change, bookmark replay.
    page.locator('#bookmark-title').fill('Boundary <script>')
    page.locator('#bookmark-note').fill('<img src=x onerror=alert(1)> observation')
    page.locator('#bookmark-save').click()
    expect(page.locator('#bookmark-list li')).to_have_count(1)
    expect(page.locator('#bookmark-list li p')).to_have_text('<img src=x onerror=alert(1)> observation')
    assert page.locator('#bookmark-list img').count()==0
    doc=page.request.get('http://127.0.0.1:8127/api/jobs/expert/review').json()
    bookmark=doc['attempts'][0]['bookmarks'][0]
    expected_voxel=(np.linalg.inv(affine)@[*bookmark['world_mm'],1])[:3]
    assert np.max(np.abs(np.array(bookmark['voxel'])-expected_voxel))<.00001
    page.locator('#select-background').select_option('t2')
    expect(page.locator('#bookmark-save')).to_be_enabled()
    page.locator('#bookmark-list li button').first.click()
    expect(page.locator('#select-background')).to_have_value(bookmark['modality'])
    expect(page.locator('#hud-crosshair')).to_contain_text('voxel:')
    page.locator('#submit-attempt').click();expect(page.locator('#review-stage')).to_have_text('Submitted')
    expect(page.locator('#bookmark-title')).to_be_disabled()
    expect(page.locator('.metrics-grid')).to_be_hidden()
    page.reload();expect(page.locator('#review-stage')).to_have_text('Submitted')
    expect(page.locator('#gl-canvas')).to_be_visible(timeout=20000)
    page.emulate_media(media='print');expect(page.locator('.metrics-grid')).to_be_hidden();page.emulate_media(media='screen')
    page.locator('#reveal-prediction').click();expect(page.locator('#review-stage')).to_have_text('Prediction revealed')
    expect(page.locator('.metrics-grid')).to_be_visible()
    expect(page.locator('.validation-card')).to_be_hidden()
    expect(page.locator('#link-download-report')).to_be_hidden()
    page.wait_for_function("document.querySelector('#gl-canvas').hidden === false")
    assert any('segmentation.nii.gz' in r for r in requests)
    assert not any('/expert.nii.gz' in r for r in requests)
    page.locator('#reveal-expert').click();expect(page.locator('#review-stage')).to_have_text('Expert revealed')
    expect(page.locator('.validation-card')).to_be_visible();expect(page.locator('#link-download-report')).to_be_visible()
    page.locator('#reflection').fill('Review reflection');page.locator('#save-reflection').click()
    expect(page.locator('#review-message')).to_have_text('Review saved.')
    page.locator('#start-blind').click();expect(page.locator('#review-stage')).to_have_text('Inspect')
    expect(page.locator('#review-exposure')).to_contain_text('Practice replay')
    page.locator('#submit-empty').click();expect(page.locator('#review-stage')).to_have_text('Submitted')
    # Same inputs, different job/model, exposure remains recorded.
    page.locator('#history-select').select_option('noexpert');expect(page.locator('#review-stage')).to_have_text('Inspect')
    expect(page.locator('#review-exposure')).to_contain_text('Practice replay')
    expect(page.locator('#no-expert')).to_be_visible()
    page.locator('#submit-empty').click();page.locator('#reveal-prediction').click()
    expect(page.locator('#review-stage')).to_have_text('Prediction revealed')
    expect(page.locator('#reveal-expert')).to_be_hidden();expect(page.locator('#link-download-report')).to_be_visible()
    page.locator('#exit-blind').click();expect(page.locator('#review-stage')).to_have_text('Normal viewing')
    expect(page.locator('#bookmark-save')).to_be_enabled(timeout=20000)
    page.locator('#bookmark-title').fill('Ordinary bookmark');page.locator('#bookmark-save').click()
    expect(page.locator('#bookmark-list li')).to_have_count(1)
    page.locator('#bookmark-list li button').nth(1).click();page.locator('#bookmark-note').fill('Ordinary edited note');page.locator('#bookmark-save').click()
    expect(page.locator('#bookmark-list li p')).to_have_text('Ordinary edited note')
    with page.expect_download() as download_info: page.locator('#export-review').click()
    exported=json.loads(Path(download_info.value.path()).read_text())
    assert exported['bookmarks'][0]['note']=='Ordinary edited note'
    page.locator('#bookmark-list li button').nth(2).click();expect(page.locator('#bookmark-list li')).to_have_count(0)
    # Older two-modality metadata fallback and unknown exposure.
    page.locator('#history-select').select_option('old');expect(page.locator('#review-stage')).to_have_text('Inspect')
    expect(page.locator('#review-exposure')).to_have_text('Prior exposure unknown')
    expect(page.locator('#select-background option[value=t1]')).to_have_js_property('disabled',True)
    expect(page.locator('#select-background option[value=t2]')).to_have_js_property('disabled',True)
    expect(page.locator('#bookmark-save')).to_be_enabled(timeout=20000)
    page.screenshot(path=str(ROOT/'work/review-blind-browser.png'),full_page=True)
    # Failure/retry must not expose a prediction; use an ordinary bookmark save.
    page.locator('#bookmark-title').fill('Retry note')
    fail=[True]
    def fail_once(route):
        if route.request.method=='PUT' and fail[0]:
            fail[0]=False;route.fulfill(status=503,content_type='application/json',body='{"detail":"Test save failure"}')
        else:route.continue_()
    ctx.route('**/api/jobs/old/review',fail_once)
    page.locator('#bookmark-save').click();expect(page.locator('#review-message')).to_contain_text('Test save failure')
    expect(page.locator('.metrics-grid')).to_be_hidden()
    page.locator('#retry-review').click();expect(page.locator('#bookmark-list li')).to_have_count(1)
    ctx.unroute('**/api/jobs/old/review',fail_once)
    # No WebGL: existing note remains editable/exportable, capture/navigation disabled.
    # A separate native WebGL-disabled process avoids relying on monkey-patched canvas methods.
    browser.close()
    browser=p.chromium.launch(executable_path=os.environ.get('NEUROSCOPE_TEST_BROWSER', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),headless=True,args=['--disable-webgl','--no-sandbox'])
    fallback=browser.new_context()
    fallback.route('**/*',lambda route:route.continue_() if route.request.url.startswith('http://127.0.0.1:8127') else route.abort())
    fp=fallback.new_page();fp.on('pageerror',lambda error:errors.append(str(error)))
    fp.goto('http://127.0.0.1:8127/?analysis=old',wait_until='domcontentloaded');expect(fp.locator('#review-stage')).to_have_text('Inspect')
    expect(fp.locator('#bookmark-save')).to_be_disabled()
    expect(fp.locator('#bookmark-list li button').first).to_be_disabled()
    fp.locator('#bookmark-list li button').nth(1).click()
    expect(fp.locator('#bookmark-save')).to_be_enabled()
    fp.locator('#bookmark-note').fill('Edited without WebGL');fp.locator('#bookmark-save').click()
    expect(fp.locator('#bookmark-list li p')).to_have_text('Edited without WebGL')
    expect(fp.locator('#export-review')).to_be_enabled()
    assert not errors,errors
    browser.close()
 print('PASS: real browser/WebGL capture and replay, persisted stages, separate reveals, print gating, exposure across jobs, old/no-expert cases, failed-save retry, WebGL fallback, safe notes.')
finally:
 server.should_exit=True;thread.join(timeout=3);temporary.cleanup()
