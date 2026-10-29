const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const SOURCES=['CIFAR10','CIFAR100','TinyImageNet'];
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const folder=(source,size)=>path.join(HERE,'results',source,'ResNet18BN','pt'+size,'simclr/ssl_seed_000');

function probe_rate(source,target,labels){
    for(const size of [1,.2]){
        const seconds=[];
        for(let seed=0;seed<5;seed++){
            const file=path.join(folder(source,size),'evaluations',`${target}_lbl${labels}_probe${seed}.json`);
            if(fs.existsSync(file)){
                const p=read(file);
                if(p.complete&&p.encoder_unchanged&&Number.isFinite(p.seconds)&&p.seconds>0)seconds.push(p.seconds);
            }
        }
        if(seconds.length){seconds.sort((a,b)=>a-b);return seconds[Math.ceil(.95*seconds.length)-1];}
    }
    throw Error('No measured probe timing for '+[source,target,labels].join('/'));
}

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(HERE,'batches')+path.sep));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const extra=Object.fromEntries(['Evaluate_benchamrk/mipa_random128/downstream_2pct.py',
        'Evaluate_benchamrk/mipa_random128/downstream_2pct.js',
        'Evaluate_benchamrk/submit_mipa_random128_downstream2pct_vulcan.sh'].map(f=>[f,hash(path.join(ROOT,f))]));
    const jobs=[],skipped=[];
    for(const source of SOURCES){
        const contract=read(path.join(ROOT,'OT-SSL-DD/ours_syntehtic_random128',source,'ResNet18BN/pt2/pipeline_config.json'));
        for(const [file,expected] of Object.entries(contract.code_hashes))assert.equal(hash(path.join(ROOT,file)),expected,'Original code changed: '+file);
        assert.equal(contract.generation_args.dataset,source);assert.equal(contract.generation_args.percentage,2);
        assert.equal(contract.generation_args.batch_real,128);assert.equal(contract.generation_args.Iteration,6000);
        const out=folder(source,2),configFile=path.join(out,'config.json'),markerFile=path.join(out,'ssl_complete.json');
        const config=fs.existsSync(configFile)?read(configFile):null,marker=fs.existsSync(markerFile)?read(markerFile):null;
        const parent=queue.get(`r128_${source}_p2_ssl`)||null;
        if(config){
            assert.deepEqual(config.signature.generation_args,contract.generation_args);
            assert.deepEqual(config.signature.selection,contract.selection);
            assert.deepEqual(config.signature.teacher,contract.teacher);
            assert.deepEqual(config.signature.driver_code_hashes,contract.code_hashes);
        }
        if(marker){
            assert.ok(config&&marker.complete&&marker.epochs===2000);
            assert.deepEqual(marker.identity,config.identity);
            assert.equal(hash(path.join(out,'encoder.pt')),marker.checkpoint_file_sha256);
        }
        if(!parent){
            const done=read(path.join(out,'complete.json'));
            assert.ok(done.complete&&done.ssl_epochs===2000&&marker);
            assert.deepEqual(done.generation_contract,contract);
        }
        const targets=[...SOURCES,'Aircraft','Dogs'].filter(t=>t!==source);
        for(const labels of [1,5])for(const target of targets){
            const name=`r128_${source}_p2_${target}_b${labels}`;
            if(queue.has(name)){skipped.push({name,reason:'active',job:queue.get(name)});continue;}
            let missing=5;
            for(let seed=0;seed<5;seed++){
                const file=path.join(out,'evaluations',`${target}_lbl${labels}_probe${seed}.json`);
                if(!fs.existsSync(file))continue;
                const p=read(file);assert.ok(config&&marker&&p.complete&&p.encoder_unchanged);
                assert.equal(p.encoder_sha256,marker.encoder_sha256);assert.deepEqual(p.expected.identity,config.identity);
                assert.deepEqual([p.expected.source,p.expected.target,p.expected.label_percentage,p.expected.seed],[source,target,labels,seed]);
                assert.deepEqual(p.expected.evaluation,config.signature.protocol.evaluation);
                assert.deepEqual(p.expected.evaluation_code_hashes,config.signature.code_hashes);missing--;
            }
            const summaryFile=path.join(out,`summary_${target}_lbl${labels}.json`);
            if(fs.existsSync(summaryFile)){
                const s=read(summaryFile);assert.ok(s.complete&&s.completed_probes===5&&s.expected_probes===5&&missing===0);
                assert.deepEqual(s.identity,config.identity);assert.deepEqual(s.generation_args,contract.generation_args);
                assert.deepEqual(s.selection,contract.selection);skipped.push({name,reason:'complete'});continue;
            }
            const rate=probe_rate(source,target,labels);
            const job={...contract,source,subset:2,stage:'downstream',target,labels,name,account:'aip-yiweilu',
                parent_job:parent,seconds:Math.ceil(missing*rate+80),buffer_seconds:60,startup_seconds:20,
                measured_probe_seconds_p95:rate,remaining_probes:missing,downstream_code_hashes:extra,
                evaluation:{ssl:'simclr',ssl_epochs:2000,ssl_seed:0,linear_epochs:100,probe_seeds:[0,1,2,3,4]}};
            assert.ok(job.seconds>0&&job.seconds<=604800);
            const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
            jobs.push({...job,manifest});
        }
    }
    assert.equal(jobs.length+skipped.length,24);
    assert.equal(new Set([...jobs,...skipped].map(j=>j.name)).size,24);
    assert.ok(jobs.every(j=>j.stage==='downstream'&&j.subset===2&&j.account==='aip-yiweilu'));
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.account,j.seconds,j.parent_job||'-',j.manifest].join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify({expected_cells:24,scheduled:jobs.length,skipped,
        account:'aip-yiweilu',ssl_retraining:false,probes_per_cell:5,linear_epochs:100,buffer_seconds:60,
        jobs:jobs.map(j=>({name:j.name,seconds:j.seconds,parent_job:j.parent_job}))},null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({scheduled:jobs.length,skipped:skipped.length,account:'aip-yiweilu',ssl_retraining:false}));
}
if(require.main===module)main(process.argv[2]);
