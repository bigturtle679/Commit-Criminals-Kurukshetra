"""HTTP review checks with inference stubbed; no checkpoint loading."""
import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastapi.testclient import TestClient
from test_review_store import META, bookmark


class ReviewApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType('pipeline'); stub.ROOT = ROOT; stub.run_case = lambda *args, **kwargs: None; stub._gpu_name = lambda: 'CPU'
        spec = importlib.util.spec_from_file_location('review_api_fixture', ROOT / 'src/app.py')
        cls.api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'pipeline':stub}): spec.loader.exec_module(cls.api)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'work'); self.addCleanup(self.temp.cleanup)
        self.api.RUNS = Path(self.temp.name)
        folder = self.api.RUNS / 'fixture'; folder.mkdir()
        for m in ('t1','t1c','t2','flair'): (folder / f'{m}.nii.gz').touch()
        self.api.JOBS = {'fixture':dict(id='fixture',status='complete',result=dict(affine=META['affine'],shape=META['shape']))}
        self.client = TestClient(self.api.app)
        self.addCleanup(self.client.close)

    def test_roundtrip_revision_restart_and_downloads(self):
        url='/api/jobs/fixture/review'
        doc=self.client.get(url).json()
        self.assertEqual(doc['metadata']['available_modalities'], ['t1','t1c','t2','flair'])
        self.assertFalse(doc['metadata']['has_expert'])
        stale=copy.deepcopy(doc)
        doc['bookmarks']=[bookmark()]
        response=self.client.put(url,json=doc)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.client.get(url).json(), response.json())
        self.assertEqual(self.client.put(url,json=stale).status_code,409)
        self.assertEqual(self.client.get('/api/files/fixture/t1.nii.gz').status_code,200)
        self.assertEqual(self.client.get('/api/files/fixture/t2.nii.gz').status_code,200)
        self.assertFalse((self.api.RUNS/'fixture/report.json').exists())

    def test_unknown_incomplete_and_invalid_review(self):
        self.assertEqual(self.client.get('/api/jobs/missing/review').status_code,404)
        self.api.JOBS['fixture']['status']='running'
        self.assertEqual(self.client.get('/api/jobs/fixture/review').status_code,409)
        self.api.JOBS['fixture']['status']='complete'
        doc=self.client.get('/api/jobs/fixture/review').json()
        b=bookmark(); b['world_mm']=[999,0,0]; doc['bookmarks']=[b]
        self.assertEqual(self.client.put('/api/jobs/fixture/review',json=doc).status_code,400)
        self.assertEqual(self.client.get('/api/files/fixture/review.json').status_code,404)


if __name__ == '__main__': unittest.main()
