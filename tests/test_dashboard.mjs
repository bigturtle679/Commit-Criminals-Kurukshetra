import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../src/web/dashboard.js',import.meta.url),'utf8');
const {allowedOverlay, composition, slicePosition} = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

test('uncertainty is available only with prediction permission and an actual artifact',()=>{
  assert.equal(allowedOverlay('uncertainty',{prediction:true},true),'uncertainty');
  assert.equal(allowedOverlay('uncertainty',{prediction:false},true),'none');
  assert.equal(allowedOverlay('uncertainty',{prediction:true},false),'none');
  assert.equal(allowedOverlay('expert',{prediction:true,expert:false},true),'none');
  assert.equal(allowedOverlay('expert',{expert:true},false),'expert');
});
test('composition uses disjoint compartments rather than stacking nested regions',()=>{
  const parts=composition({WT:100,TC:30,ET:10,ED:70});
  assert.deepEqual(parts.map(p=>p.value),[70,20,10]);
  assert.ok(Math.abs(parts.reduce((sum,p)=>sum+p.share,0)-1)<1e-12);
  assert.deepEqual(composition({WT:0,TC:0,ET:0,ED:0}),[]);
  assert.deepEqual(composition({WT:100,TC:30,ET:40,ED:70}),[]);
  assert.deepEqual(composition({WT:100,TC:30,ET:10,ED:90}),[]);
  assert.deepEqual(composition({WT:NaN}),[]);
});
test('slice navigation clamps edges and maps voxel centres back to their indices',()=>{
  assert.equal(slicePosition(0,155).index,0);
  assert.equal(slicePosition(1,155).index,154);
  assert.equal(slicePosition(-1,155).index,0);
  assert.equal(slicePosition(0.5,0),null);
  for(let i=0;i<155;i++) assert.equal(slicePosition((i+.5)/155,155).index,i);
});

const { measureShares, filmstripIndices, LABELS } = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
test('measurement rings use explicit denominators and never exceed 100%',()=>{
  const s=measureShares({WT:36,TC:7.75,ET:7.65},1465);
  assert.ok(Math.abs(s.WT-36/1465)<1e-9);
  assert.ok(Math.abs(s.TC-7.75/36)<1e-9);
  assert.ok(Math.abs(s.ET-7.65/36)<1e-9);
  assert.equal(measureShares({WT:10,TC:5,ET:1},undefined).WT,null);
  assert.equal(measureShares({WT:0,TC:0,ET:0},100).TC,null);
});
test('filmstrip spans the volume and always contains the active slice',()=>{
  for (const [count,active] of [[155,0],[155,72],[155,154],[240,119],[5,3]]) {
    const picks=filmstripIndices(count,active);
    assert.ok(picks.includes(active));
    assert.ok(picks.every(i=>i>=0&&i<count));
    assert.equal(picks.length,Math.min(15,count));
  }
  assert.deepEqual(filmstripIndices(0,0),[]);
});
test('label legend follows backend labels 1/2/4',()=>{
  assert.deepEqual(Object.keys(LABELS).map(Number),[1,2,4]);
  assert.equal(LABELS[1].volume({TC:7.75,ET:7.65,ED:28.2}).toFixed(2),'0.10');
  assert.equal(LABELS[2].volume({TC:7.75,ET:7.65,ED:28.2}),28.2);
});
const { stageFromMessage } = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
test('workflow stage follows real backend status messages',()=>{
  assert.equal(stageFromMessage('Uploading scans'),0);
  assert.equal(stageFromMessage('Queued for NVIDIA GeForce RTX 4060 Laptop GPU inference'),1);
  assert.equal(stageFromMessage('Validating MRI dimensions and coordinate alignment'),1);
  assert.equal(stageFromMessage('Loading ensemble models & validating inputs'),1);
  assert.equal(stageFromMessage('Executing Model A (80%): SwinUNETR'),2);
  assert.equal(stageFromMessage('Synthesizing 3D polygonal meshes and measurement report'),2);
});
