"""CPU regression checks for production validation, metrics and ensemble inference."""
import sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from src.validation import read_expert
from src.metrics import evaluate_case
from scripts.models.model_loader import load_resenc, load_swinunetr, resolve_weights
from scripts.ensemble_inference import compute_voxel_entropy, predict_ensemble

class AuditTests(unittest.TestCase):
    def test_missing_resenc_fails_before_construction(self):
        with patch('scripts.models.model_loader.ResEncUNet') as constructor:
            with self.assertRaises(FileNotFoundError):
                load_resenc(weights_path='/nonexistent/test_checkpoint.pt', device='cpu')
            constructor.assert_not_called()

    def test_missing_explicit_swin_does_not_use_other_weights(self):
        with self.assertRaises(FileNotFoundError):
            load_swinunetr(weights_path='/nonexistent/test_checkpoint.pt', device='cpu')

    def test_checkpoint_incompatible_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.pt';torch.save({'state_dict':{'fake':torch.ones(1)}},p)
            with self.assertRaises(RuntimeError): load_resenc(weights_path=p,device='cpu')

    def test_entropy_low_precision_endpoints(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            h=compute_voxel_entropy(torch.tensor([0,1,0.5],dtype=dtype))
            self.assertTrue(torch.isfinite(h).all())
            self.assertLess(h[0].item(),1e-4)
            self.assertAlmostEqual(h[2].item(),1.0)

    def test_entropy_nonfinite_rejected(self):
        with self.assertRaises(ValueError): compute_voxel_entropy(torch.tensor([float('nan')]))

    def mask(self,data,affine=None,et=3):
        affine=np.eye(4) if affine is None else affine
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'mask.nii.gz';nib.save(nib.Nifti1Image(data.astype(np.float32),affine),p)
            return read_expert(p,nib.Nifti1Image(np.ones((8,8,8)),np.eye(4)),et)

    def test_empty_expert_is_valid(self): self.mask(np.zeros((8,8,8)))
    def test_wrong_et_rejected(self):
        with self.assertRaises(ValueError): self.mask(np.full((8,8,8),4),et=3)
    def test_fractional_labels_rejected(self):
        with self.assertRaises(ValueError): self.mask(np.full((8,8,8),1.5))
    def test_affine_mismatch_rejected(self):
        a=np.eye(4);a[0,3]=1
        with self.assertRaises(ValueError): self.mask(np.zeros((8,8,8)),a)
    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError): self.mask(np.zeros((8,8,9)))
    def test_nonfinite_mask_rejected(self):
        with self.assertRaises(ValueError): self.mask(np.full((8,8,8),np.nan))

    def test_empty_metrics(self):
        z=np.zeros((2,2,2));r=evaluate_case(z,z,np.eye(4),3)['ET']
        self.assertEqual(r['dice'],1);self.assertIsNone(r['sensitivity']);self.assertIsNone(r['precision'])
    def test_metrics_label_mapping_and_volume(self):
        p=np.array([[[4,2]]]);g=np.array([[[3,0]]]);r=evaluate_case(p,g,np.diag([1,1,2,1]),3)
        self.assertEqual(r['ET']['dice'],1);self.assertEqual(r['WT']['precision'],0.5)
        self.assertEqual(r['WT']['pred_volume_ml'],0.004)
    def test_metrics_shape_guard(self):
        with self.assertRaises(ValueError): evaluate_case(np.zeros((2,2)),np.zeros((2,1)),np.eye(4))

    def test_shared_predictor_uses_each_channel_order(self):
        class Network(torch.nn.Module):
            def __init__(self): super().__init__();self.seen=None
            def forward(self,x): self.seen=x[0,:,0,0,0].tolist();return x[:,:3]
        a,b=Network(),Network()
        cfg={'model_a':{'channel_order':['flair','t1c','t1','t2'],'roi_size':[8,8,8]},'model_b':{'channel_order':['flair','t1','t1c','t2'],'roi_size':[8,8,8]}}
        bundle={'model_a':a,'model_b':b,'config':cfg,'weight_a':0.8,'weight_b':0.2}
        x=torch.stack([torch.full((8,8,8),float(i)) for i in range(4)])
        pred,entropy=predict_ensemble(x,bundle,'cpu')
        self.assertEqual(a.seen,[3,0,1,2]);self.assertEqual(b.seen,[3,1,0,2])
        self.assertTrue(np.isfinite(entropy).all());self.assertTrue((pred[2]<=pred[0]).all())

class PathParityTests(unittest.TestCase):
    def test_cli_web_synthetic_outputs_agree(self):
        from scripts.ensemble_inference import run_ensemble
        from src.pipeline import run_case
        class Net(torch.nn.Module):
            def forward(self,x): return x[:,:3]
        cfg={'model_a':{'channel_order':['flair','t1c','t1','t2'],'roi_size':[8,8,8]},'model_b':{'channel_order':['flair','t1','t1c','t2'],'roi_size':[8,8,8]},'postprocessing':{'threshold':0.6}}
        bundle={'model_a':Net(),'model_b':Net(),'config':cfg,'weight_a':0.8,'weight_b':0.2,'desc_a':'synthetic test only','desc_b':'synthetic test only','checkpoint_sha256':{}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);paths={}
            rng=np.random.default_rng(42)
            for k in ('t1','t1c','t2','flair'):
                paths[k]=root/(k+'.nii.gz')
                nib.save(nib.Nifti1Image(rng.normal(size=(8,8,8)).astype(np.float32),np.eye(4)),paths[k])
            expert=root/'gt.nii.gz';nib.save(nib.Nifti1Image(np.zeros((8,8,8),np.uint8),np.eye(4)),expert)
            with patch('scripts.ensemble_inference.load_ensemble_models',return_value=bundle), patch('src.pipeline.load_model',return_value=(bundle,'synthetic')),patch('torch.cuda.is_available',return_value=False):
                a=run_ensemble(paths,root/'cli',expert_path=expert,expert_et=3,device='cpu')
                b=run_case(paths,root/'web',expert=expert,expert_et=3,model_name='ensemble')
            for name in ('segmentation.nii.gz','uncertainty.nii.gz'):
                np.testing.assert_allclose(nib.load(root/'cli'/name).get_fdata(),nib.load(root/'web'/name).get_fdata())
            self.assertEqual(a['metrics'],b['metrics'])

if __name__=='__main__': unittest.main(verbosity=2)
