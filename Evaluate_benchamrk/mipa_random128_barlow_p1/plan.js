const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const {epochHeader}=require('../cross_arch_ablation/plan');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const oldMain=path.join(BENCH,'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main');
const oldDown=path.join(BENCH,'ours_eval_results_simclr2k_barlow10k_2ssl3probes_downstream');
const p95=values=>{assert.ok(values.length&&values.every(n=>Number.isFinite(n)&&n>0));values.sort((a,b)=>a-b);return values[Math.ceil(.95*values.length)-1];};

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(HERE,'batches')+path.sep));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const files=[...fs.readdirSync(BENCH).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        'Evaluate_benchamrk/eval_jobs/interactive_ours.py','Evaluate_benchamrk/eval_jobs/worker.py',
        'Evaluate_benchamrk/cross_arch_ablation/plan.js',
        'Evaluate_benchamrk/mipa_random128_barlow_p1/worker.py',
        'Evaluate_benchamrk/mipa_random128_barlow_p1/plan.js',
        'Evaluate_benchamrk/submit_mipa_random128_barlow_p1_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const jobs=[],skipped=[];
    for(const source of ['CIFAR10','CIFAR100'])for(const seed of [0,1]){
        const name=`bt128_${source}_p1_s${seed}`;
        if(queue.has(name)){skipped.push({name,reason:'active',job:queue.get(name)});continue;}
        const suffix=path.join(source,'ResNet18BN/pt1/barlowtwins',`ssl_seed_00${seed}`);
        const old=path.join(oldMain,suffix),folder=path.join(HERE,'results',suffix);
        const dir=path.join(ROOT,'OT-SSL-DD/ours_syntehtic_random128',source,'ResNet18BN/pt1');
        const state=read(path.join(dir,'generation_state.json')),contract=read(path.join(dir,'pipeline_config.json'));
        assert.ok(state.complete&&state.iteration===8000);assert.deepEqual(state.contract,contract);
        assert.equal(contract.generation_args.batch_real,128);
        const synthetic_file=path.relative(ROOT,path.join(dir,`res_IIC_${source}_ResNet18BN_1percent_final.pt`));
        const synthetic_sha256=hash(path.join(ROOT,synthetic_file));
        const recipe_file=path.relative(ROOT,path.join(old,'config.json')),recipe_sha256=hash(path.join(ROOT,recipe_file));
        const recipe=read(path.join(ROOT,recipe_file)),protocol=recipe.signature.protocol;
        assert.equal(protocol.ssl.subset_epochs,10000);assert.equal(protocol.evaluation.epochs,100);
        assert.deepEqual(recipe.signature.ssl_seeds,[0,1]);assert.deepEqual(recipe.signature.probe_seeds,[0,1,2]);
        const configFile=path.join(folder,'config.json'),markerFile=path.join(folder,'ssl_complete.json');
        const config=fs.existsSync(configFile)?read(configFile):null,marker=fs.existsSync(markerFile)?read(markerFile):null;
        if(config){assert.deepEqual(config.signature.replacement_code_hashes,code_hashes);assert.equal(config.signature.synthetic.file_sha256,synthetic_sha256);assert.deepEqual(config.signature.protocol,protocol);}
        let epoch=0;
        if(marker){
            assert.ok(config&&marker.complete&&marker.epochs===10000);assert.deepEqual(marker.identity,config.identity);
            assert.equal(hash(path.join(folder,'encoder.pt')),marker.checkpoint_file_sha256);epoch=10000;
        }else if(fs.existsSync(path.join(folder,'training.pt'))){assert.ok(config);epoch=epochHeader(path.join(folder,'training.pt'),config.identity);}
        assert.ok(epoch>=0&&epoch<=10000);
        const timings=fs.readFileSync(path.join(old,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(JSON.parse).map(r=>r.seconds);
        let seconds=(10000-epoch)*p95(timings),missing=0;
        for(const target of [source,source==='CIFAR10'?'CIFAR100':'CIFAR10','TinyImageNet'])for(const labels of [1,5])for(const probe of [0,1,2]){
            const filename=`${target}_lbl${labels}_probe${probe}.json`,file=path.join(folder,'evaluations',filename);
            if(fs.existsSync(file)){
                const p=read(file);assert.ok(marker&&p.complete&&p.encoder_unchanged);
                assert.deepEqual(p.expected.identity,config.identity);assert.equal(p.encoder_sha256,marker.encoder_sha256);
                assert.deepEqual([p.expected.source,p.expected.target,p.expected.label_percentage,p.expected.seed],[source,target,labels,probe]);
                assert.deepEqual(p.expected.evaluation,protocol.evaluation);assert.deepEqual(p.expected.evaluation_code_hashes,config.signature.code_hashes);
                assert.ok(Number.isFinite(p.accuracy_percent)&&p.accuracy_percent>=0&&p.accuracy_percent<=100);continue;
            }
            const baseline=path.join(target===source?oldMain:oldDown,suffix,'evaluations');
            seconds+=p95([0,1,2].map(p=>read(path.join(baseline,`${target}_lbl${labels}_probe${p}.json`)).seconds));missing++;
        }
        if(epoch===10000&&!missing){skipped.push({name,reason:'complete'});continue;}
        const job={name,source,seed,synthetic_file,synthetic_sha256,recipe_file,recipe_sha256,code_hashes,
            seconds:Math.ceil(seconds+80),buffer_seconds:60,startup_seconds:20,ssl_epochs:10000,probe_epochs:100};
        job.account=job.seconds>10800?'aip-yiweilu':'aip-boyuwang';
        const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        jobs.push({...job,manifest});
    }
    assert.equal(jobs.length+skipped.length,4);assert.equal(new Set(jobs.map(j=>j.name)).size,jobs.length);
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.account,j.seconds,'-',j.manifest].join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify({jobs,skipped,expected_encoders:4,
        expected_probes:72,expected_cells:12,ssl_seeds:[0,1],probe_seeds:[0,1,2],ssl_epochs:10000,probe_epochs:100},null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({jobs:jobs.length,skipped:skipped.length,cells:12,probes:72}));
}
if(require.main===module)main(process.argv[2]);
