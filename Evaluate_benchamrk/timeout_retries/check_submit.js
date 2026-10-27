const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const file = path.join(__dirname,'../resubmit_timeouts_wang_killarney.sh');
const script = fs.readFileSync(file,'utf8');
require('child_process').execFileSync('bash',['-n',file]);
const rows = script.match(/done <<'SSL_RETRIES'\n([\s\S]*?)\nSSL_RETRIES/)[1].trim().split('\n').map(r=>r.split(/\s+/));
assert.deepEqual(rows,[
    ['dfssl_mkdt_TinyImageNet_p1_simclr_s0','26'],
    ['dfssl_mkdt_TinyImageNet_p1_simclr_s1','26'],
    ['dfssl_ours_TinyImageNet_p1_simclr_s1','25'],
    ['dfssl_mkdt_TinyImageNet_p2_simclr_s0','59'],
    ['dfssl_mkdt_TinyImageNet_p2_simclr_s1','59'],
]);
assert.ok(rows.every(([,minutes])=>Number(minutes)<60));
assert.ok(script.includes('for dataset in CIFAR10 CIFAR100; do'));
assert.ok(script.includes('for start in $(seq 0 5 95); do'));
assert.ok(script.includes('--account=aip-boyuwang'));
assert.ok(!script.includes('--account=aip-yiweilu'));
assert.ok(script.includes('saved >= 4')&&script.includes('$folder/training.pt'));
assert.ok(script.includes('queued "mainlu_${name#dfssl_}"'));
assert.ok(script.includes('--worker "$batch/code" experts'));
assert.ok(!/--worker[^\n]*\b(init|distill|probe)\b/.test(script));
const vulcanFile=path.join(__dirname,'../resubmit_timeouts_wang_vulcan.sh');
const vulcan=fs.readFileSync(vulcanFile,'utf8');
require('child_process').execFileSync('bash',['-n',vulcanFile]);
assert.ok(vulcan.includes('export TIMEOUT_RETRY_CLUSTER=vulcan'));
assert.ok(script.includes('[[ $cluster != killarney ]] || command+=(--exclude=kn101)'));
assert.ok(vulcan.includes('timeout_retries/vulcan_worker.py'));
const worker=fs.readFileSync(path.join(__dirname,'vulcan_worker.py'),'utf8');
assert.ok(worker.includes('worker.experts(')&&worker.includes('worker.train('));
assert.ok(!worker.includes('worker.main('));
assert.ok(!/environ\[[^\n]*CLUSTER[^\n]*\]\s*=/.test(worker));
console.log('OK: Killarney/Vulcan, 40 expert + 5 SSL retries, Wang only, under 1h, no later stages.');
