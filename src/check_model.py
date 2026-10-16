"""Offline CUDA smoke test; synthetic input does not measure medical accuracy."""
import json
import time
from pathlib import Path
import torch
import monai
from monai.networks.nets import SegResNet

root = Path(__file__).resolve().parents[1]
bundle = root / 'models/brats_mri_segmentation'
config = json.loads((bundle / 'configs/inference.json').read_text())['network_def']
config.pop('_target_')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.')
model = SegResNet(**config)
state = torch.load(bundle / 'models/model.pt', map_location='cpu', weights_only=True)
if 'model' in state:
    state = state['model']
model.load_state_dict(state, strict=True)
model = model.eval().cuda()
torch.cuda.reset_peak_memory_stats()
x = torch.randn(1, 4, 96, 96, 96, device='cuda')
started = time.perf_counter()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
    result = model(x)
torch.cuda.synchronize()
assert result.shape == (1, 3, 96, 96, 96), result.shape
assert torch.isfinite(result).all(), 'Non-finite output'
print(json.dumps({
    'status': 'PASS', 'test': 'synthetic 96-cubed patch, NOT a patient scan or accuracy test',
    'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'monai': monai.__version__,
    'output_shape': list(result.shape), 'seconds_including_first_run_setup': round(time.perf_counter()-started, 3),
    'peak_allocated_gpu_mib': round(torch.cuda.max_memory_allocated()/1024**2, 1),
    'input_order': ['T1c', 'T1', 'T2', 'FLAIR'], 'output_order': ['TC', 'WT', 'ET']
}, indent=2))
