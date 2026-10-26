const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const {balance,definitions,excluded,epochHeader}=require('./remaining_plan');
const {pairs}=require('./plan'),{rows,key}=require('./tables');
for(let n=0;n<30;n++){
    const jobs=Array.from({length:n},(_,i)=>({name:'job'+i,seconds:(i+1)*123})),split=balance(jobs);
    assert.equal(split.counts['aip-boyuwang'],Math.ceil(n/2));assert.equal(split.counts['aip-yiweilu'],Math.floor(n/2));
    const lu=balance(jobs,'aip-yiweilu');assert.equal(lu.counts['aip-yiweilu'],n);assert.ok(jobs.every(j=>j.account==='aip-yiweilu'));
}
const groups=definitions(),keys=new Set();
for(const g of groups)for(const [target,labels]of g.cells){const k=key({...g,target,labels});assert.ok(!keys.has(k));keys.add(k);}
const blanks=['table.tex','cifar10_downstream.tex','cifar100_downstream.tex','tinyimagenet_simclr.tex','tinyimagenet_barlowtwins.tex'].flatMap(f=>
    rows(f,fs.readFileSync(path.join(__dirname,'../../CVPR_tables',f),'utf8')).flatMap(r=>r.cells))
    .filter(c=>c.model==='ResNet18'&&!c.value&&!excluded(c));
assert.ok(blanks.every(c=>keys.has(key(c))),'Every eligible blank must have a job mapping');
const resumed=groups.find(g=>g.id==='ours_TinyImageNet_p2_simclr');assert.ok(!resumed.hold);
assert.equal(resumed.generation_job_name,'oursgen_TinyImageNet_ResNet18BN_p2');
const worker=fs.readFileSync(path.join(__dirname,'worker.py'),'utf8');
assert.ok(worker.includes('lock_name = f\'{cell["target"]}_lbl{cell["labels"]}_ssl{seed}_probe{sample["probe_seed"]}\''));
if(process.argv[2]){
    const batch=path.resolve(process.argv[2]),coverage=JSON.parse(fs.readFileSync(path.join(batch,'coverage.json')));
    const jobs=fs.readFileSync(path.join(batch,'jobs.tsv'),'utf8').trim().split('\n').filter(Boolean)
        .map(l=>JSON.parse(fs.readFileSync(l.split('\t').at(-1))));
    assert.equal(jobs.length,coverage.jobs);assert.equal(new Set(jobs.map(j=>j.name)).size,jobs.length);
    const seen=new Set(Object.keys(JSON.parse(fs.readFileSync(path.join(batch,'queue.json')))));
    for(const job of jobs){
        assert.ok(job.seconds>=60&&job.buffer_seconds===60);assert.ok(['aip-boyuwang','aip-yiweilu'].includes(job.account));
        if(process.env.TABLE_ACCOUNT)assert.equal(job.account,process.env.TABLE_ACCOUNT);
        if(process.env.TABLE_GROUP)assert.equal(job.group_id,process.env.TABLE_GROUP);
        assert.ok(job.dependencies.every(d=>seen.has(d)),'Parents must be submitted first');seen.add(job.name);
        const g=groups.find(g=>g.id===job.group_id);assert.ok(g&&!g.hold);
        if(job.stage==='probe')assert.deepEqual(job.samples.map(s=>[s.ssl_seed,s.probe_seed]),pairs(g.scheme));
        else{assert.ok(g.source==='TinyImageNet'||g.kind==='mkdt'&&g.subset===0.2);assert.ok(job.resume_epoch_estimate<g.epochs);}
    }
    assert.equal(coverage.blocked_cells,0);
}
const source=path.join(__dirname,'../ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/TinyImageNet/ResNet18BN/pt0.2/barlowtwins/ssl_seed_000');
if(fs.existsSync(path.join(source,'training.pt'))){
    const config=JSON.parse(fs.readFileSync(path.join(source,'config.json'))),epoch=epochHeader(path.join(source,'training.pt'),config.identity);
    assert.ok(epoch>0&&epoch<=10000);console.log('Actual saved Tiny Ours 0.2% Barlow epoch:',epoch);
}
assert.deepEqual(pairs('ten_ssl'),Array.from({length:10},(_,s)=>[s,s]));
console.log('OK: all eligible blanks mapped; held data excluded; Lu-only and balanced accounts; unique cell locks; exact probe counts.');
