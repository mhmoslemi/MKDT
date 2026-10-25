const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const {parseTable,eligible,unitFor} = require('./plan');
const cells = parseTable(fs.readFileSync(path.join(__dirname,'../../CVPR_tables/table.tex'),'utf8'));
assert.equal(cells.length,144);
assert.ok(!cells.filter(eligible).some(c=>c.ssl==='barlowtwins'&&c.subset===2));
assert.ok(!cells.filter(eligible).some(c=>c.kind==='mkdt'&&c.subset===0.2&&c.source!=='TinyImageNet'));
for(const kind of ['ours','mkdt','random']) for(const subset of [0.2,1,2]) for(const ssl of ['simclr','barlowtwins']) {
    const u=unitFor({source:'TinyImageNet',kind,subset,ssl},0);
    assert.equal(u.epochs,kind==='random'?({0.2:6000,1:5000,2:4000})[subset]:ssl==='simclr'?2000:10000);
    assert.ok(u.checkpoint_dir.includes('/ResNet18'));
    assert.ok(!u.checkpoint_dir.includes('simclr1k'));
}
const mkdt02=cells.filter(c=>c.kind==='mkdt'&&c.subset===0.2&&eligible(c));
assert.ok(mkdt02.every(c=>c.source==='TinyImageNet'));
const launcher=fs.readFileSync(path.join(__dirname,'../submit_main_table_lu_killarney.sh'),'utf8');
const start=launcher.indexOf('    account=aip-yiweilu\n');
assert.ok(start>=0);
const routing=launcher.slice(start,launcher.indexOf('    printf -v walltime',start));
const accounts=require('child_process').execFileSync('bash',['-c',
    `for seconds in 60 3599 3600 27425; do\n${routing}\nprintf '%s\\n' "$account"\ndone`],{encoding:'utf8'});
assert.equal(accounts,'aip-boyuwang\naip-boyuwang\naip-yiweilu\naip-yiweilu\n');
console.log('OK: table parsing, exclusions, architecture, SSL schedules, and account routing at 1h.');
