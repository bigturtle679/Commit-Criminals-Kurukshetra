"""Integration checks against the running localhost app and its real MRI output."""
import json, time
from pathlib import Path
import httpx
import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
with httpx.Client(base_url='http://127.0.0.1:8000',timeout=30) as client:
    for url in ['/','/static/app.js','/static/style.css','/static/vendor/niivue.js','/api/status']:
        assert client.get(url).status_code==200,url
    bad=client.post('/api/analyze',files={k:('photo.png',b'not-an-mri','image/png') for k in ['t1c','t1','t2','flair']})
    assert bad.status_code==400 and 'NIfTI' in bad.text,bad.text
    assert client.get('/api/files/not-a-job/report.json').status_code==404
    case=ROOT/'data/demo'; prefix='BraTS-GLI-00000-000-'
    mapping={'t1c':'t1c','t1':'t1n','t2':'t2w','flair':'t2f','expert':'seg'}
    handles={k:open(case/(prefix+s+'.nii.gz'),'rb') for k,s in mapping.items()}
    try:
        response=client.post('/api/analyze',files={k:(prefix+mapping[k]+'.nii.gz',f,'application/octet-stream') for k,f in handles.items()},data={'expert_et':'3'})
    finally:
        for f in handles.values():f.close()
    assert response.status_code==200,response.text
    uid=response.json()['id']
    for _ in range(90):
        job=client.get('/api/jobs/'+uid).json()
        if job['status'] in ('complete','error'):break
        time.sleep(1)
    assert job['status']=='complete',job
    out=ROOT/'work/runs'/uid
    original=nib.load(case/(prefix+'t1c.nii.gz'))
    pred=nib.load(out/'segmentation.nii.gz')
    assert pred.shape==original.shape
    assert np.allclose(pred.affine,original.affine)
    labels=pred.get_fdata()
    assert set(np.unique(labels)) <= {0,1,2,4}
    voxel_ml=abs(np.linalg.det(original.affine[:3,:3]))/1000
    assert abs(np.count_nonzero(labels)*voxel_ml-job['result']['volumes_ml']['WT'])<.0011
    gt=nib.load(case/(prefix+'seg.nii.gz')).get_fdata()
    p=labels>0;g=gt>0
    dice=2*np.count_nonzero(p&g)/(p.sum()+g.sum())
    assert abs(dice-job['result']['metrics']['WT']['dice'])<.0001
    for f in ['report.json','segmentation.nii.gz','expert.nii.gz','brain.obj','tumor.obj']:
        assert client.get(f'/api/files/{uid}/{f}').status_code==200,f
    for name in ['brain','tumor']:
        contents=(out/(name+'.obj')).read_text()
        assert '\nf ' in contents and contents.startswith('v ')
    print(json.dumps({'status':'PASS','checks':['local assets','reject screenshots','multipart MRI upload','real GPU inference','original shape and affine','label convention','physical volume','reference Dice','downloads','nonempty meshes'],'job_id':uid,'result':job['result']},indent=2))
