"""Review persistence and geometry tests; no model weights or inference required."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from review_store import empty_review, load_review, save_review, ReviewConflict

DATE = '2026-10-16T12:00:00Z'
META = dict(affine=[[0,-2,0,20],[3,0,0,-10],[0,0,-4,40],[0,0,0,1]], shape=[10,12,14],
            available_modalities=['t1','t1c','t2','flair'], has_expert=True)


def bookmark():
    return dict(id='bookmark-1',title='Location',note='<script>plain text</script>',world_mm=[12, -1, 20],
                voxel=[999,999,999],modality='flair',view=3,overlay='none',opacity=.55,created_at=DATE,updated_at=DATE)


def attempt():
    return dict(id='attempt-1',stage='inspect',exposure='unseen',created_at=DATE,updated_at=DATE,
                bookmarks=[],submitted_at=None,submitted_bookmarks=None,no_areas=False,reflection='')


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def save(self, data, meta=META):
        return save_review(self.folder, data, meta)

    def inspect(self, with_bookmark=True):
        doc = empty_review()
        a = attempt()
        if with_bookmark: a['bookmarks'].append(bookmark())
        doc['attempts'].append(a)
        doc['active_attempt_id'] = a['id']
        return self.save(doc)

    def submit(self, with_bookmark=True):
        doc = self.inspect(with_bookmark)
        a = doc['attempts'][0]
        a.update(stage='submitted', submitted_at=DATE, submitted_bookmarks=copy.deepcopy(a['bookmarks']), no_areas=not with_bookmark)
        return self.save(doc)

    def test_coordinates_use_full_affine_and_server_recomputes_voxel(self):
        doc = empty_review(); doc['bookmarks'] = [bookmark()]
        saved = self.save(doc)
        for actual, expected in zip(saved['bookmarks'][0]['voxel'], [3,4,5]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(load_review(self.folder), saved)
        self.assertEqual(saved['bookmarks'][0]['note'], '<script>plain text</script>')
        self.assertEqual(list(self.folder.glob('*.tmp')), [])
        saved['bookmarks'][0]['title'] = 'Updated'
        saved = self.save(saved)
        saved['bookmarks'] = []
        self.assertEqual(self.save(saved)['bookmarks'], [])

    def test_conflicting_tab_cannot_overwrite(self):
        stale = empty_review(); self.save(empty_review())
        with self.assertRaises(ReviewConflict): self.save(stale)
        self.assertEqual(load_review(self.folder)['revision'], 1)

    def test_nonfinite_and_outside_coordinates_rejected_without_writing(self):
        for world in ([float('nan'),0,0], [999,0,0]):
            doc=empty_review(); b=bookmark(); b['world_mm']=world; doc['bookmarks']=[b]
            with self.assertRaises(ValueError): self.save(doc)
        self.assertEqual(load_review(self.folder), empty_review())

    def test_text_and_view_validation(self):
        for key,value in [('title','x'*101),('note','x'*2001),('opacity',2),('modality','pet'),('view',9)]:
            doc=empty_review(); b=bookmark(); b[key]=value; doc['bookmarks']=[b]
            with self.assertRaises(ValueError): self.save(doc)

    def test_submitted_observations_are_immutable_but_reflection_is_editable(self):
        doc = self.submit()
        for field in ('bookmarks','submitted_bookmarks'):
            changed=copy.deepcopy(doc); changed['attempts'][0][field][0]['note']='changed'
            with self.assertRaises(ValueError): self.save(changed)
        doc['attempts'][0]['reflection']='New reflection'
        doc['attempts'][0]['stage']='prediction'
        doc=self.save(doc)
        doc['attempts'][0]['stage']='expert'
        doc=self.save(doc)
        doc['attempts'][0]['stage']='inspect'
        with self.assertRaises(ValueError): self.save(doc)

    def test_stage_skip_and_expert_without_mask(self):
        doc=self.inspect(); doc['attempts'][0]['stage']='prediction'
        with self.assertRaises(ValueError): self.save(doc)
        doc=load_review(self.folder)
        a=doc['attempts'][0]; a.update(stage='submitted',submitted_at=DATE,submitted_bookmarks=copy.deepcopy(a['bookmarks']))
        doc=self.save(doc); doc['attempts'][0]['stage']='prediction'; doc=self.save(doc)
        doc['attempts'][0]['stage']='expert'
        with self.assertRaises(ValueError): self.save(doc, dict(META,has_expert=False))

    def test_empty_submission_requires_explicit_acknowledgement(self):
        doc=self.inspect(False)
        doc['attempts'][0].update(stage='submitted',submitted_at=DATE,submitted_bookmarks=[])
        with self.assertRaises(ValueError): self.save(doc)
        doc['attempts'][0]['no_areas']=True
        self.assertEqual(self.save(doc)['attempts'][0]['stage'],'submitted')

    def test_new_attempt_preserves_history_and_normal_notes(self):
        doc=self.submit(); old=copy.deepcopy(doc['attempts'][0])
        a=attempt(); a['id']='attempt-2'; doc['attempts'].append(a); doc['active_attempt_id']=a['id']
        doc['bookmarks']=[bookmark()]
        saved=self.save(doc)
        self.assertEqual(saved['attempts'][0],old)
        saved['attempts']=saved['attempts'][1:]
        with self.assertRaises(ValueError): self.save(saved)

    def test_older_two_modality_case(self):
        doc=empty_review(); doc['bookmarks']=[bookmark()]
        self.save(doc,dict(META,available_modalities=['t1c','flair']))
        doc=load_review(self.folder); doc['bookmarks'][0]['modality']='t2'
        with self.assertRaises(ValueError): self.save(doc,dict(META,available_modalities=['t1c','flair']))


if __name__ == '__main__': unittest.main()
