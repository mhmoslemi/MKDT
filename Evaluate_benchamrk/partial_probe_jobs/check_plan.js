const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {pairs}=require('./plan'),{rows,key}=require('./tables');
const groups=JSON.parse(fs.readFileSync(path.join(__dirname,'cells.json'),'utf8'));
assert.equal(groups.reduce((n,g)=>n+g.cells.length,0),33);
for(const g of groups){
    const p=pairs(g.scheme);assert.equal(p.length,6);assert.equal(new Set(p.map(x=>x.join(':'))).size,6);
    if(g.scheme==='two_ssl')assert.deepEqual(p.map(x=>x[0]),[0,0,0,1,1,1]);
    if(g.scheme==='one_ssl')assert.ok(p.every(x=>x[0]===0));
    if(g.scheme==='four_ssl')assert.deepEqual([...new Set(p.map(x=>x[0]))],[0,1,2,4]);
    for(const [target,labels] of g.cells){
        const c={source:g.source,kind:g.kind,subset:g.subset,ssl:g.ssl,target,labels};
        const file=target===g.source?'table.tex':g.source.toLowerCase()+'_downstream.tex';
        const cells=rows('CVPR_tables/'+file,fs.readFileSync(path.join(__dirname,'../../CVPR_tables',file),'utf8')).flatMap(r=>r.cells);
        assert.ok(cells.some(x=>x.model==='ResNet18'&&key(x)===key(c)));
    }
}
const worker=fs.readFileSync(path.join(__dirname,'worker.py'),'utf8');
assert.ok(worker.includes('linear_probe('));
assert.ok(!/\btrain\s*\(/.test(worker));
assert.ok(worker.includes("state['epoch'] != marker['epochs']"));
assert.ok(worker.includes("'ssl_updates': 0"));

// Exercise real table layouts with all writes kept in memory.
const vm=require('vm'),script=fs.readFileSync(path.join(__dirname,'tables.js'),'utf8');
for(const g of [groups[0],groups[2],groups[4]]){
    const [target,labels]=g.cells[0],files=new Map(),writes=new Set(),mock=Object.create(fs);
    mock.readFileSync=(f,...args)=>files.has(String(f))?files.get(String(f)):fs.readFileSync(f,...args);
    mock.writeFileSync=(f,text)=>files.set(String(f),text);
    mock.chmodSync=()=>{};
    mock.renameSync=(from,to)=>{files.set(String(to),files.get(String(from)));files.delete(String(from));writes.add(String(to));};
    const module={exports:{}};
    vm.runInNewContext(script,{__dirname,module,process,console:{log(){}},
        require:name=>name==='fs'?mock:require(name)});
    const summary={source:g.source,kind:g.kind,subset:g.subset,ssl:g.ssl,target,labels,
        complete:true,std_ddof:1,ssl_epochs:g.epochs,ssl_seeds:[...new Set(pairs(g.scheme).map(p=>p[0]))],
        samples:pairs(g.scheme).map(([ssl_seed,probe_seed])=>({ssl_seed,probe_seed})),
        tex_value:'\\dsacc{12.34}{0.56}',repeated_encoders:g.scheme!=='two_ssl'};
    summary.cell_key=key(summary);
    const tableFiles=[...fs.readdirSync(path.join(__dirname,'../../CVPR_tables')).filter(f=>f.endsWith('.tex')).map(f=>'CVPR_tables/'+f),
        ...fs.readdirSync(path.join(__dirname,'../../CVPR_tables/full')).filter(f=>f.endsWith('.tex')).map(f=>'CVPR_tables/full/'+f)];
    const expectedWrites=tableFiles.filter(f=>rows(f,fs.readFileSync(path.join(__dirname,'../..',f),'utf8'))
        .some(r=>r.cells.some(c=>c.model==='ResNet18'&&key(c)===summary.cell_key&&c.value===''))).length;
    module.exports.applySummary(summary);
    assert.equal(writes.size,expectedWrites,'Expected only blank source/mirrored table updates');
    for(const file of writes){
        const before=rows(file,fs.readFileSync(file,'utf8')).flatMap(r=>r.cells);
        const after=rows(file,files.get(file)).flatMap(r=>r.cells);assert.equal(after.length,before.length);
        before.forEach((c,i)=>assert.equal(after[i].value,
            c.model==='ResNet18'&&key(c)===summary.cell_key&&c.value===''?
            summary.tex_value:c.value));
        assert.ok(!files.get(file).includes('Entries marked \\textsuperscript{*}'));
    }
    const count=writes.size;module.exports.applySummary(summary);assert.equal(writes.size,count);
}
console.log('OK: 33 cells; six real probe pairs each; final-state recovery only; frozen probes; table mirrors and unrelated cells preserved.');
