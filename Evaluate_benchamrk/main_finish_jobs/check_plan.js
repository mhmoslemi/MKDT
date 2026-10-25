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
console.log('OK: table parsing, exclusions, architecture, and SSL schedules.');
