"""Run a frozen, hash-verified patient manifest using production inference."""
import argparse,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.pipeline import run_case,MODELS
from src.validation import read_expert
from src.pipeline import read_image
from scripts.models.model_loader import checkpoint_hash,resolve_weights
import numpy as np

def validate_manifest(manifest):
    patients=manifest['patients'];seen=set()
    if not patients: raise ValueError('Empty patient manifest')
    for case in patients:
        ident=(case['source'],case['version'],case['patient_id'])
        if ident in seen: raise ValueError('Duplicate patient')
        seen.add(ident)
        if case['overlap_status'] not in ('unknown','known_training_overlap','excluded_from_known_training'): raise ValueError('Missing overlap status')
        labels=case['label_mapping']
        if [labels[k] for k in ('background','nonenhancing_core','edema')] != [0,1,2] or not labels['evidence']: raise ValueError('Unsupported/undocumented labels')
        paths={**case['modalities'],'expert_mask':case['expert_mask']}
        for name,path in paths.items():
            p=Path(path)
            if not p.is_absolute():p=ROOT/p
            if checkpoint_hash(p)!=case['sha256'][name]: raise ValueError(f'Hash mismatch: {p}')
            paths[name]=p
        imgs=[read_image(paths[k]) for k in ('t1','t1c','t2','flair')]
        ref=imgs[1]
        if any(im.shape!=ref.shape or not np.allclose(im.affine,ref.affine,atol=1e-3,rtol=0) for im in imgs):raise ValueError('Modality geometry mismatch')
        read_expert(paths['expert_mask'],ref,labels['enhancing_tumor'])
    return patients

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--models',nargs='+',choices=['swinunetr','segresnet','resenc','ensemble'],default=['swinunetr']);p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    manifest=Path(args.manifest).resolve();patients=validate_manifest(json.loads(manifest.read_text()))
    if args.validate_only:print(f'Validated {len(patients)} patients');return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    meta={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),'manifest_sha256':checkpoint_hash(manifest),'manifest_path':str(manifest),'torch':torch.__version__,'cuda':torch.version.cuda,'models':args.models,'patient_count':len(patients),'timing':'first patient per model includes cold load; later patients warm cache; end-to-end includes exports'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    (out/'manifest.json').write_text(manifest.read_text())
    (out/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
    rows=[]
    for model in args.models:
        try:
            if model=='ensemble':
                from scripts.models.model_loader import load_ensemble_models
                loaded=load_ensemble_models(device='cpu');hashes={'model_a':checkpoint_hash(loaded['path_a']),'model_b':checkpoint_hash(loaded['path_b'])};del loaded
            else:
                paths={'resenc':'models/resenc_brats/resenc_m_brats.pt','swinunetr':'models/swinunetr_brats/model_weights_BRATS-finetune.pt','segresnet':'models/brats_mri_segmentation/models/model.pt'}
                hashes={model:checkpoint_hash(resolve_weights(paths[model],[]))}
        except Exception as e:
            rows.append({'model':model,'status':'blocked','reason':str(e)});continue
        for i,case in enumerate(patients):
            dest=out/model/f'{i:03d}';start=time.perf_counter()
            row={'patient_id':case['patient_id'],'source':case['source'],'overlap_status':case['overlap_status'],'model':model,'checkpoint_sha256':hashes}
            try:
                paths={k:ROOT/Path(v) for k,v in case['modalities'].items()}
                result=run_case(paths,dest,expert=ROOT/Path(case['expert_mask']),expert_et=case['label_mapping']['enhancing_tumor'],model_name=model)
                row.update(status='complete',result=result)
            except Exception as e:row.update(status='failed',reason=str(e),traceback=traceback.format_exc())
            row['end_to_end_seconds']=time.perf_counter()-start;rows.append(row)
            (out/'results.json').write_text(json.dumps(rows,indent=2));print(model,case['patient_id'],row['status'],flush=True)
        MODELS.clear()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
