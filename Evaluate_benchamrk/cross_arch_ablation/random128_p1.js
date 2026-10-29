const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const {epochHeader}=require('./plan');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH),RUN=path.join(HERE,'random128_p1');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(n=>Number.isFinite(n)&&n>0));a.sort((a,b)=>a-b);return a[Math.ceil(.95*a.length)-1];};
const folder=(source,model,seed)=>path.join(RUN,'results',source,model,'ours/pt1/simclr',`ssl_seed_00${seed}`);

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(RUN,'batches')+path.sep));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const original=read(path.join(HERE,'calibration.json'));assert.ok(original.complete);
    for(const [file,sha] of Object.entries(original.code_hashes))assert.equal(hash(path.join(ROOT,file)),sha,'Original calibration code changed: '+file);
    for(const model of ['ResNet34','ResNet50'])assert.ok(original.models[model].bn_only&&original.models[model].strict_reload&&original.models[model].frozen_bn);
    const synthetics={};
    for(const source of ['CIFAR10','CIFAR100']){
        const dir=path.join(ROOT,'OT-SSL-DD/ours_syntehtic_random128',source,'ResNet18BN/pt1');
        const state=read(path.join(dir,'generation_state.json')),contract=read(path.join(dir,'pipeline_config.json'));
        assert.ok(state.complete&&state.iteration===8000);assert.deepEqual(state.contract,contract);
        assert.equal(contract.generation_args.batch_real,128);assert.equal(contract.generation_args.percentage,1);
        const file=path.relative(ROOT,path.join(dir,`res_IIC_${source}_ResNet18BN_1percent_final.pt`));
        synthetics[file]=hash(path.join(ROOT,file));
    }
    const calibration={...original,synthetics,source_calibration_sha256:hash(path.join(HERE,'calibration.json'))};
    const calibrationFile=path.join(RUN,'calibration.json');
    if(fs.existsSync(calibrationFile))assert.deepEqual(read(calibrationFile),calibration);
    else fs.writeFileSync(calibrationFile,JSON.stringify(calibration,null,2)+'\n',{flag:'wx'});
    const files=[...fs.readdirSync(BENCH).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        'Evaluate_benchamrk/eval_jobs/interactive_ours.py',
        ...['models.py','worker.py','plan.js','calibrate.py','random128_p1.py','random128_p1.js'].map(f=>'Evaluate_benchamrk/cross_arch_ablation/'+f),
        'Evaluate_benchamrk/submit_cross_arch_random128_p1_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const jobs=[],skipped=[],known=new Set(queue.keys());
    function add(unit,stage,seconds,parent){
        const name=`x128_${unit.source}_${unit.model}_p1_s${unit.seed}_${stage}`;
        if(queue.has(name)){skipped.push({name,reason:'active',job:queue.get(name)});return name;}
        const job={...unit,name,stage,seconds:Math.ceil(seconds+80),buffer_seconds:60,startup_seconds:20,
            parent:parent||null,code_hashes,calibration_sha256:hash(calibrationFile),end_epoch:2000};
        job.account=job.seconds>10800?'aip-yiweilu':'aip-boyuwang';
        assert.ok(job.seconds>0&&job.seconds<=604800);
        if(parent)assert.ok(known.has(parent),'Missing parent: '+parent);
        known.add(name);
        const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        jobs.push({...job,manifest});return name;
    }
    for(const source of ['CIFAR10','CIFAR100'])for(const model of ['ResNet34','ResNet50'])for(const seed of [0,1]){
        const synthetic_file=Object.keys(synthetics).find(f=>f.includes('/'+source+'/'));
        const unit={source,model,kind:'ours',size:1,seed,synthetic_file,synthetic_sha256:synthetics[synthetic_file]};
        const out=folder(source,model,seed),configFile=path.join(out,'config.json'),markerFile=path.join(out,'ssl_complete.json');
        const config=fs.existsSync(configFile)?read(configFile):null,marker=fs.existsSync(markerFile)?read(markerFile):null;
        if(config){assert.deepEqual(config.signature.ablation_code_hashes,code_hashes);assert.equal(config.signature.synthetic.path,path.join(ROOT,synthetic_file));}
        let epoch=0,ready=false;
        if(marker){
            assert.ok(marker.complete&&marker.epochs===2000);assert.deepEqual(marker.identity,config.identity);
            assert.equal(hash(path.join(out,'encoder.pt')),marker.checkpoint_file_sha256);ready=true;epoch=2000;
        }else if(fs.existsSync(path.join(out,'training.pt')))epoch=epochHeader(path.join(out,'training.pt'),config.identity);
        assert.ok(epoch>=0&&epoch<=2000);
        const old=path.join(HERE,'results',source,model,'ours/pt1/simclr',`ssl_seed_00${seed}`);
        const records=fs.readFileSync(path.join(old,'ssl.jsonl'),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
        const parent=ready?null:add(unit,'ssl',(2000-epoch)*p95(records.slice(-200).map(r=>r.seconds)),null);
        let seconds=0,missing=0;
        for(const target of [source,source==='CIFAR10'?'CIFAR100':'CIFAR10','TinyImageNet'])for(const labels of [1,5])for(const probe of [0,1]){
            const filename=`${target}_lbl${labels}_probe${probe}.json`,file=path.join(out,'evaluations',filename);
            if(fs.existsSync(file)){
                const p=read(file);assert.ok(ready&&p.complete&&p.encoder_unchanged);
                assert.deepEqual(p.expected.identity,config.identity);assert.equal(p.encoder_sha256,marker.encoder_sha256);
                assert.deepEqual([p.expected.source,p.expected.target,p.expected.label_percentage,p.expected.seed],[source,target,labels,probe]);
                assert.deepEqual(p.expected.evaluation,config.signature.protocol.evaluation);
                assert.deepEqual(p.expected.evaluation_code_hashes,config.signature.code_hashes);continue;
            }
            const timings=[0,1].map(p=>read(path.join(old,'evaluations',`${target}_lbl${labels}_probe${p}.json`)).seconds);
            seconds+=p95(timings);missing++;
        }
        if(missing)add(unit,'probe',seconds,parent);
    }
    assert.ok(jobs.length+skipped.length<=16);
    assert.equal(new Set(jobs.map(j=>j.name)).size,jobs.length);
    assert.ok(jobs.every(j=>j.kind==='ours'&&j.size===1&&j.end_epoch===2000));
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.account,j.seconds,j.parent||'-',j.manifest].join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    const coverage={source_datasets:['CIFAR10','CIFAR100'],models:['ResNet34','ResNet50'],synthetic_size:1,
        expected_encoders:8,expected_probes:96,expected_cells:24,ssl_epochs:2000,probe_epochs:100,
        ssl_seeds:[0,1],probe_seeds:[0,1],jobs:jobs.map(j=>({name:j.name,stage:j.stage,seconds:j.seconds,account:j.account,parent:j.parent})),skipped,
        account_policy:'Wang; jobs longer than three hours use Lu',synthetics};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({jobs:jobs.length,ssl:jobs.filter(j=>j.stage==='ssl').length,probe:jobs.filter(j=>j.stage==='probe').length,skipped:skipped.length,cells:24}));
}
if(require.main===module)main(process.argv[2]);
