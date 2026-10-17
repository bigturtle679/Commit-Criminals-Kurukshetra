"""In-process route and UI-contract checks. No browser or network access."""
import importlib.util
import re
import sys
import types
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids = []
        self.links = []
        self.assets = []
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if 'id' in attributes:
            self.ids.append(attributes['id'])
        if tag == 'a':
            self.links.append(attributes.get('href', ''))
        if tag in ('script', 'img'):
            self.assets.append(attributes.get('src', ''))
        if tag == 'link' and attributes.get('rel') == 'stylesheet':
            self.assets.append(attributes.get('href', ''))


@pytest.fixture(scope='module')
def client():
    stub = types.ModuleType('pipeline')
    stub.ROOT = ROOT
    stub.run_case = lambda *args, **kwargs: None
    stub._gpu_name = lambda: 'Test hardware'
    spec = importlib.util.spec_from_file_location('frontend_test_app', ROOT / 'src/app.py')
    api = importlib.util.module_from_spec(spec)
    # Import the real routes without touching persisted jobs or starting cleanup.
    with patch.dict(sys.modules, {'pipeline': stub}), \
         patch('threading.Thread.start'), \
         patch.object(Path, 'glob', return_value=[]), \
         patch.object(Path, 'iterdir', return_value=iter(())):
        spec.loader.exec_module(api)
    yield TestClient(api.app)
    api.POOL.shutdown(wait=False)


def test_homepage_launches_real_workspace(client):
    # '/' serves Atharva's landing page (frontend-1 branch, neuroscope/) under /site/.
    response = client.get('/', follow_redirects=False)
    assert response.status_code == 307 and response.headers['location'] == '/site/'
    landing = client.get('/site/')
    assert landing.status_code == 200
    assert landing.text == (ROOT / 'neuroscope/index.html').read_text()
    assert 'upload.html' in Elements(landing.text).links
    assert 'NOT A PATIENT SCAN' in landing.text                      # hero video is illustrative only
    assert 'does not present model output as a diagnosis' in landing.text
    assert client.get('/site/assets/neuroport-brain.mp4').status_code == 200
    assert client.get('/site/assets/nerves_only.mp4').status_code == 200
    upload_js = (ROOT / 'neuroscope/upload.js').read_text()
    assert '/api/analyze-zip' in upload_js and '/api/analyze' in upload_js and '/api/samples/' in upload_js
    assert '/workspace?analysis=' in upload_js
    assert 'Simulation' not in upload_js                              # no fake completion timers
    assert client.get('/upload', follow_redirects=True).status_code == 200


def test_workspace_retains_production_controls(client):
    response = client.get('/workspace')
    assert response.status_code == 200
    ids = Elements(response.text).ids
    assert len(ids) == len(set(ids)), 'Duplicate HTML IDs break control binding'
    for name in ('zip-input', 'upload-form', 'gl-canvas', 'btn-ensemble',
                 'history-select', 'select-overlay', 'link-download-mask'):
        assert name in ids
    assert '/static/workspace.css' in response.text
    assert '/static/app.js' in response.text


def test_static_application_id_contract(client):
    ids = set(Elements(client.get('/workspace').text).ids)
    source = (ROOT / 'src/web/app.js').read_text()
    referenced = set(re.findall(r"\$\('([^']+)'\)", source))
    assert not (referenced - ids), f'Missing controls: {referenced - ids}'


@pytest.mark.parametrize('path', ['/', '/workspace'])
def test_page_assets_are_served(client, path):
    for asset in Elements(client.get(path).text).assets:
        if asset.startswith('/'):
            response = client.get(asset)
            assert response.status_code == 200, asset
            assert response.content, asset


def test_saved_analysis_link_is_preserved(client):
    response = client.get('/?analysis=example123', follow_redirects=False)
    assert response.status_code == 307
    assert response.headers['location'] == '/workspace?analysis=example123'
    assert client.get(response.headers['location']).status_code == 200
