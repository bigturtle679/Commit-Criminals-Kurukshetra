import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
if (!globalThis.crypto) globalThis.crypto = webcrypto;
const source = await readFile(new URL('../src/web/review.js', import.meta.url), 'utf8');
const {worldToVoxel, accessFor, caseKey} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
for (const affine of [
  [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  [[0,-2,0,20],[3,0,0,-10],[0,0,-4,40],[0,0,0,1]],
  [[-1,0.2,0,10],[0,2,0.3,-5],[0,0,3,20],[0,0,0,1]]
]) {
  const voxel=[3.2,4.4,5.1];
  const world=affine.slice(0,3).map(row=>row[3]+row.slice(0,3).reduce((sum,x,i)=>sum+x*voxel[i],0));
  worldToVoxel(world,affine,[10,12,14]).forEach((x,i)=>assert.ok(Math.abs(x-voxel[i])<1e-10));
}
assert.throws(()=>worldToVoxel([999,0,0],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[10,10,10]));
assert.throws(()=>worldToVoxel([NaN,0,0],null,null));
for (const stage of ['inspect','submitted']) assert.deepEqual(accessFor({stage},true),{prediction:false,expert:false,report:false});
assert.deepEqual(accessFor({stage:'prediction'},true),{prediction:true,expert:false,report:false});
assert.deepEqual(accessFor({stage:'expert'},true),{prediction:true,expert:true,report:true});
assert.deepEqual(accessFor({stage:'prediction'},false),{prediction:true,expert:false,report:true});
assert.deepEqual(accessFor(null,true),{prediction:true,expert:true,report:true});
const hashes={t1:'a'.repeat(64),t1c:'b'.repeat(64),t2:'c'.repeat(64),flair:'d'.repeat(64)};
assert.equal(await caseKey(hashes),await caseKey({flair:hashes.flair,t2:hashes.t2,t1c:hashes.t1c,t1:hashes.t1}));
assert.notEqual(await caseKey(hashes),await caseKey({...hashes,t1:'e'.repeat(64)}));
assert.equal(await caseKey(null),null);
console.log('PASS: full-affine coordinate round trips, invalid locations, all reveal stages, stable modality-ordered exposure keys.');
