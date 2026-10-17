'use strict';
const assert = require('assert/strict'), fs = require('fs'), path = require('path');
const {build, check} = require('./plan.js');
const bench = path.resolve(__dirname, '../..');
const plan = build(bench);
assert.equal(plan.jobs.length, 40);
const units = plan.jobs.flatMap(j => j.units);
assert.equal(units.length, 540);
assert.equal(new Set(units.map(u => u.relative)).size, 540);
const probes = units.flatMap(u => u.cells.map(c => u.relative + '/' + c.target + '/' + c.labels));
assert.equal(probes.length, 3240);
assert.equal(new Set(probes).size, 3240);
assert.equal(plan.jobs.filter(j => j.account === 'aip-yiweilu').length, 4);
assert.equal(plan.jobs[38].units.length, 20);
assert.equal(plan.jobs[39].units.length, 20);
assert.deepEqual(plan.jobs[38].wait_for, ['6334332','6334333']);
assert.deepEqual(plan.jobs[39].wait_for, ['6334334','6334335']);
for (const j of plan.jobs.slice(0,38))
  assert(j.units.every(u => !(u.source === 'TinyImageNet' && u.model === 'ResNet18' && u.subset !== 0.2)));
console.log('PASS exact 40-job coverage, no duplicate probes, both dependency groups');

// A tiny filesystem fixture exercises preflight without touching real Slurm.
// This test writes only inside the allocation's node-local temporary directory.
assert(process.env.SLURM_JOB_ID && process.env.SLURM_TMPDIR, 'Use Slurm');
const root = fs.mkdtempSync(path.join(process.env.SLURM_TMPDIR, 'random-eval-plan-'));
const job = {...plan.jobs[38], units: [plan.jobs[38].units[0]]};
const fixture = {...plan, jobs: [job], checkpoint_root: root + '/checkpoints',
  venv: root + '/venv', data_root: root + '/data'};
fs.mkdirSync(fixture.data_root);
fs.mkdirSync(fixture.venv + '/bin', {recursive:true});
fs.writeFileSync(fixture.venv + '/bin/activate', 'fixture');
fs.writeFileSync(root + '/plan.json', JSON.stringify(fixture));
const crypto = require('crypto');
const digest = crypto.createHash('sha256').update(fs.readFileSync(root + '/plan.json')).digest('hex');
fs.writeFileSync(root + '/bundle.sha256', digest + '  plan.json\n');
fs.writeFileSync(root + '/gres.txt', 'gpu:l40s:4(S:0-1)\n');
fs.writeFileSync(root + '/accounts.txt', 'aip-boyuwang|\n');
const u = job.units[0], parent = u.ssl === 'simclr' ? '6334332' : '6334333';
fs.writeFileSync(root + '/queue.txt', parent + '|random_TinyImageNet_ResNet18_1_' + u.ssl + '|RUNNING\n');
assert.throws(() => check(root, 'vulcan', 'l40s'), /missing/);
check(root, 'killarney', 'l40s');
assert(fs.readFileSync(root + '/submit.tsv', 'utf8').includes('afterok:' + parent));
fs.writeFileSync(root + '/queue.txt', '');
assert.throws(() => check(root, 'killarney', 'l40s'), /missing/);
const folder = path.join(fixture.checkpoint_root, u.relative);
fs.mkdirSync(folder, {recursive:true});
fs.writeFileSync(folder + '/encoder.pt', 'fixture, not a real tensor');
fs.writeFileSync(folder + '/config.json', '{}');
fs.writeFileSync(folder + '/ssl_complete.json', JSON.stringify({complete:true, epochs:1000,
  seed:u.seed, source:u.source, model:u.model, method:u.ssl}));
check(root, 'killarney', 'l40s');
assert(fs.readFileSync(root + '/submit.tsv', 'utf8').includes('\t-\tl40s'));
console.log('PASS wrong-cluster rejection, missing-input rejection, active/completed dependencies');
fs.rmSync(root, {recursive:true});
