// Plan saved-encoder probes on the login node without importing Python or loading tensors.
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const crypto = require('crypto');
const here = __dirname, bench = path.resolve(here, '..'), batch = path.resolve(process.argv[2]);
assert.ok(batch.startsWith(path.join(here, 'finish_tiny_batches') + path.sep));
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const exists = file => fs.existsSync(file) && fs.statSync(file).size > 0;
const active = new Set((process.argv[3] || '').split('\n').map(n => n.trim()).filter(Boolean));
const reference = read(path.join(bench, 'protocol.json'));
const hashes = Object.fromEntries([...fs.readdirSync(bench).filter(f => f.endsWith('.py')).map(f => 'Evaluate_benchamrk/' + f),
    'networks.py', 'benchmarks/downstream_data.py'].sort().map(f => [f,
    crypto.createHash('sha256').update(fs.readFileSync(path.join(bench, '..', f))).digest('hex')]));
const identifier = p => [p.route, p.target, p.labels].join('|');
const routeFile = (subset, ssl, seed) => `random_downstream_jobs/routes/TinyImageNet_p${String(subset).replace('.', 'p')}_${ssl}_s${seed}.json`;
const queued = new Set();
function queuedJob(directory, name) {
    if (!fs.existsSync(directory)) return null;
    for (const dir of fs.readdirSync(directory).sort().reverse()) {
        const file = path.join(directory, dir, name + '.json');
        if (exists(file)) return read(file);
    }
    return null;
}
for (const name of active) {
    if (/^rtiny_[a-f0-9]{16}$/.test(name)) {
        const job = queuedJob(path.join(here, 'finish_tiny_batches'), name);
        assert.ok(job, 'Cannot identify already queued job: ' + name);
        job.probes.forEach(p => queued.add(identifier(p)));
    } else if (/^dp(?:3|15)_[A-Za-z0-9_]+$/.test(name)) {
        const job = queuedJob(path.join(bench, 'downstream_probe_jobs/batches'), name);
        assert.ok(job, 'Cannot identify queued downstream bundle: ' + name);
        for (const bundle of job.bundles || [job]) {
            const u = bundle.unit;
            if (u.kind !== 'random' || u.source !== 'TinyImageNet') continue;
            for (const p of bundle.probes) queued.add(identifier({route:routeFile(u.subset,u.ssl,u.seed),target:p.target,labels:p.labels}));
        }
    }
}
const aircraft = {route:routeFile(2,'simclr',9),target:'Aircraft',labels:1};
if (active.has('rnd_down_TinyImageNet_p2_simclr_s9_Aircraft_l1')) queued.add(identifier(aircraft));

const times = new Map();
for (const kind of ['random','kmeans']) for (const subset of [0.2,1,2]) for (const ssl of ['simclr','barlowtwins']) {
    for (let seed=0;seed<(kind==='random'?10:6);seed++) {
        const dir = path.join(bench,kind==='random'?'random_eval_results_interactive_10seeds':'kmeans_eval_results_interactive_6seeds',
            'TinyImageNet/ResNet18',kind,`pt${subset}`,ssl,`seed_${String(seed).padStart(3,'0')}/evaluations`);
        for (const [target,labels] of [['TinyImageNet',1],['TinyImageNet',5],['Aircraft',1]]) {
            const file = path.join(dir,`${target}_lbl${labels}.json`);
            if (!exists(file)) continue;
            const r=read(file),key=target+'|'+labels;
            if(r.complete && Number.isFinite(r.seconds) && r.seconds>0){if(!times.has(key))times.set(key,[]);times.get(key).push(r.seconds);}
        }
    }
}
for(const [key,values] of times){values.sort((a,b)=>a-b);times.set(key,values[Math.ceil(values.length*.9)-1]);}
const pending=[],counts={complete:0,queued:0,missing_encoder:0,main:0,aircraft:0};
const cached=new Map();
function inputs(request) {
    if(cached.has(request.route))return cached.get(request.route);
    const route=read(path.join(bench,request.route));
    assert.equal(route.mode,'existing');assert.equal(route.source,'TinyImageNet');
    const folder=path.resolve(bench,route.checkpoint_dir),output=path.resolve(bench,route.output_dir);
    assert.ok([folder,output].every(f=>f.startsWith(bench+path.sep)));
    if(!['encoder.pt','config.json','ssl_complete.json'].every(f=>exists(path.join(folder,f)))){cached.set(request.route,null);return null;}
    const config=read(path.join(folder,'config.json')),marker=read(path.join(folder,'ssl_complete.json')),s=config.signature;
    assert.deepEqual([s.source,s.model,s.selection_method,s.subset,s.ssl,config.identity.seed],
        [route.source,'ResNet18','random',route.subset,route.ssl,route.seed]);
    assert.equal(route.epochs,({0.2:6000,1:5000,2:4000})[route.subset]);
    assert.equal(s.protocol.ssl.subset_epochs,route.epochs);
    assert.deepEqual(s.code_hashes,hashes);assert.deepEqual(s.protocol.evaluation,reference.evaluation);
    assert.ok(marker.complete);assert.equal(marker.epochs,route.epochs);assert.deepEqual(marker.identity,config.identity);
    const value={route,config,marker,output};cached.set(request.route,value);return value;
}
function consider(request) {
    const data=inputs(request);if(!data){counts.missing_encoder++;return;}
    const {route,config,marker,output}=data,file=path.join(output,'evaluations',`${request.target}_lbl${request.labels}.json`);
    if(exists(file)){
        const r=read(file),e=r.expected;
        assert.ok(r.complete&&r.encoder_unchanged&&r.encoder_sha256===marker.encoder_sha256,file);
        assert.deepEqual(e.identity,config.identity);
        assert.deepEqual([e.source,e.target,e.label_percentage,e.seed],['TinyImageNet',request.target,request.labels,route.seed]);
        assert.deepEqual(e.evaluation,reference.evaluation);assert.deepEqual(e.evaluation_code_hashes,hashes);
        assert.ok(Number.isFinite(r.accuracy_percent)&&r.accuracy_percent>=0&&r.accuracy_percent<=100);
        counts.complete++;return;
    }
    if(queued.has(identifier(request))){counts.queued++;return;}
    assert.ok(Number.isFinite(times.get(request.target+'|'+request.labels)),'Missing measured probe time');
    pending.push(request);counts[request.target==='TinyImageNet'?'main':'aircraft']++;
}
for(const subset of [0.2,1,2])for(const ssl of ['simclr','barlowtwins']){
    if(subset===2&&ssl==='barlowtwins')continue;
    for(let seed=0;seed<10;seed++)for(const labels of [1,5])consider({route:routeFile(subset,ssl,seed),target:'TinyImageNet',labels});
}
consider(aircraft);
assert.equal(new Set(pending.map(identifier)).size,pending.length);
const cost=probes=>probes.reduce((sum,p)=>sum+times.get(p.target+'|'+p.labels),0);
const short=[],jobs=[];
for(let i=0;i<pending.length;i+=3){const probes=pending.slice(i,i+3);if(cost(probes)+150<1200)short.push(probes);else jobs.push(probes);}
for(let i=0;i<short.length;i+=5)jobs.push(short.slice(i,i+5).flat());
let tsv='';
for(const probes of jobs){
    assert.ok(probes.length>=1&&probes.length<=15);
    const name='rtiny_'+crypto.createHash('sha256').update(probes.map(identifier).join('\n')).digest('hex').slice(0,16);
    const job={name,probes,seconds:Math.ceil(cost(probes)+150)},file=path.join(batch,name+'.json');
    fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
    tsv+=[name,job.seconds,probes.length,file].join('\t')+'\n';
}
fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
console.log('PLAN:',JSON.stringify({...counts,probes:pending.length,jobs:jobs.length,buffer_seconds:150,
    measured_seconds:Object.fromEntries(times),ssl_training:false}));
