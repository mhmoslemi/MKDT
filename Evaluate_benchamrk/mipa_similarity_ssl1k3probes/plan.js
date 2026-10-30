// Login-safe: JSON/log timings and scalar checkpoint headers; no Python or tensor loading.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const {epochHeader}=require('../cross_arch_ablation/plan');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const SOURCES=['CIFAR10','CIFAR100','SVHN','FashionMNIST'],SIZES=[.2,1,2];
const METRICS=['l1','l2','mmd','mmd-dist','dm','energy','whitened-cosine','sinkhorn'];
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(n=>Number.isFinite(n)&&n>0),'Missing measured timing');return a.sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];};
const records=f=>fs.existsSync(f)?fs.readFileSync(f,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(HERE,'batches')+path.sep));
    const files=[...fs.readdirSync(BENCH).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json','OT-SSL-DD/eval.py',
        'Evaluate_benchamrk/digits_simclr/datasets.py','Evaluate_benchamrk/fashionmnist/dataset.py',
        'Evaluate_benchamrk/digits_simclr/prepared.json','Evaluate_benchamrk/fashionmnist/full/prepared.json',
        'Evaluate_benchamrk/mipa_similarity_ssl1k3probes/worker.py','Evaluate_benchamrk/mipa_similarity_ssl1k3probes/plan.js',
        'Evaluate_benchamrk/submit_mipa_similarity_ssl1k3probes_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const active=new Set(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|')[0].trim()));
    const jobs=[],skipped=[],preflight=new Map();let index=0;
    for(const source of SOURCES)for(const subset of SIZES){
        const oldRoot=source==='SVHN'?'digits_subset_simclr2k':source==='FashionMNIST'?'fashionmnist/subset':subset===2?'subset_simclr2k_1ssl5probes':'cifar_small_simclr2k_1ssl5probes';
        const baseline=path.join(BENCH,oldRoot,'results',source,'ResNet18/random','pt'+subset,'simclr/ssl_seed_000');
        for(const similarity of METRICS){
            const account=(++index)%5===0?'aip-yiweilu':'aip-boyuwang';
            const name='ms1k_'+source+'_p'+String(subset).replace('.','p')+'_'+similarity;
            const gen=path.join(ROOT,'OT-SSL-DD/ours_syntehtic_similarity_lr0p002_b128_i10k',source,'ResNet18BN','pt'+subset,similarity);
            const state=read(path.join(gen,'generation_state.json')),contract=read(path.join(gen,'pipeline_config.json'));
            assert.deepEqual(state.contract,contract);assert.ok(state.iteration>0&&state.iteration<=10000);
            assert.deepEqual([contract.generation_args.dataset,contract.generation_args.percentage,contract.similarity],[source,subset,similarity]);
            const synthetic=path.join(gen,`res_IIC_${source}_ResNet18BN_${subset}percent_final.pt`);assert.ok(fs.existsSync(synthetic));
            const previous=path.join(BENCH,'mipa_similarity_ablation/results',source,'ResNet18BN','pt'+subset,similarity,'simclr/ssl_seed_000');
            const measured=records(path.join(previous,'ssl.jsonl')).slice(-200);
            const sslRate=p95((measured.length?measured:records(path.join(baseline,'ssl.jsonl')).slice(-200)).map(r=>r.seconds));
            const probeRates=Object.fromEntries([1,5].map(lbl=>{let timings=[];for(let seed=0;seed<5;seed++){
                const f=path.join(previous,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`);if(fs.existsSync(f))timings.push(read(f).seconds);
            }if(!timings.length)timings=[0,1,2,3,4].map(seed=>read(path.join(baseline,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`)).seconds);
            return [lbl,p95(timings)];}));
            const unit={name,source,subset,similarity,account,code_hashes,generation_contract:contract,
                synthetic_path:path.relative(ROOT,synthetic),evaluation:{ssl:'simclr',ssl_epochs:1000,ssl_seed:0,linear_epochs:100,probe_seeds:[0,1,2],labels:[1,5]}};
            let generationSeconds=0;
            if(!state.complete){
                assert.deepEqual([source,subset,similarity],['CIFAR10',1,'dm'],'Unexpected incomplete synthetic set');
                const manifests=fs.readdirSync(path.join(BENCH,'mipa_similarity_ablation/batches')).sort().reverse().map(d=>path.join(BENCH,'mipa_similarity_ablation/batches',d,'mips_CIFAR10_p1_dm.json'));
                // The submitted manifest, not a newer live method1.py, defines the generation being resumed.
                const matched=manifests.find(f=>fs.existsSync(f)&&fs.existsSync(path.join(path.dirname(f),'submitted.tsv'))&&
                    /^mips_CIFAR10_p1_dm\t[0-9]+\t/m.test(fs.readFileSync(path.join(path.dirname(f),'submitted.tsv'),'utf8')));
                assert.ok(matched,'Missing original generation manifest');
                const old=read(matched);for(const key of ['generation_args','selection','teacher','code_hashes','similarity','metric_parameters'])assert.deepEqual(old[key],contract[key]);
                for(const [file,sha]of Object.entries(old.code_hashes))assert.equal(hash(file==='OT-SSL-DD/method1.py'?path.join(path.dirname(matched),'method1.py'):path.join(ROOT,file)),sha,'Resume code changed: '+file);
                assert.ok(!active.has('mips_CIFAR10_p1_dm'),'Generation already running');
                unit.resume_generation=path.relative(ROOT,matched);
                const rates=[];
                for(const file of fs.readdirSync(path.join(BENCH,'mipa_similarity_ablation/logs')).filter(f=>f.startsWith('mips_CIFAR10_p1_dm-')&&f.endsWith('.out'))){
                    let previous;
                    for(const line of fs.readFileSync(path.join(BENCH,'mipa_similarity_ablation/logs',file),'utf8').split('\n')){
                        const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
                        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,iteration:+m[3]};
                        if(previous&&current.iteration>previous.iteration&&current.time>previous.time)
                            rates.push((current.time-previous.time)/(current.iteration-previous.iteration));
                        previous=current;
                    }
                }
                generationSeconds=(10000-state.iteration)*p95(rates);
            }else assert.equal(state.iteration,10000);
            if(active.has(name)){skipped.push({name,reason:'active'});continue;}
            const out=path.join(HERE,'results',source,'ResNet18BN','pt'+subset,similarity,'simclr/ssl_seed_000');
            const configFile=path.join(out,'config.json'),config=fs.existsSync(configFile)?read(configFile):null;
            let epoch=0,missing={1:3,5:3};
            if(config){assert.deepEqual(config.signature.driver_code_hashes,code_hashes);assert.deepEqual(config.signature.generation_contract,contract);assert.equal(config.signature.protocol.ssl.subset_epochs,1000);}
            const markerFile=path.join(out,'ssl_complete.json');
            if(fs.existsSync(markerFile)){const m=read(markerFile);assert.ok(m.complete&&m.epochs===1000);assert.deepEqual(m.identity,config.identity);epoch=1000;}
            else if(fs.existsSync(path.join(out,'training.pt'))){assert.ok(config);epoch=epochHeader(path.join(out,'training.pt'),config.identity);}
            assert.ok(epoch>=0&&epoch<=1000);
            for(const lbl of [1,5])for(let seed=0;seed<3;seed++){
                const file=path.join(out,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`);
                if(fs.existsSync(file)){const r=read(file);assert.ok(r.complete&&r.encoder_unchanged&&epoch===1000);assert.deepEqual(r.expected.identity,config.identity);assert.equal(r.expected.evaluation.epochs,100);missing[lbl]--;}
            }
            if(epoch===1000&&!missing[1]&&!missing[5]){skipped.push({name,reason:'complete'});continue;}
            const seconds=Math.ceil(generationSeconds+(1000-epoch)*sslRate+missing[1]*probeRates[1]+missing[5]*probeRates[5]+20+60);
            assert.ok(seconds>0&&seconds<604800);
            const job={...unit,seconds,startup_seconds:20,buffer_seconds:60,timing:{approximate:true,ssl_seconds_per_epoch_p95:sslRate,
                probe_seconds_p95:probeRates,generation_seconds:generationSeconds}};
            const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});jobs.push({...job,manifest});
            if(!unit.resume_generation&&!preflight.has(source))preflight.set(source,manifest);
        }
    }
    assert.equal(index,96);assert.equal(jobs.length+skipped.length,96);
    const accounts=Object.fromEntries(['aip-boyuwang','aip-yiweilu'].map(a=>[a,jobs.filter(j=>j.account===a).length]));
    const coverage={expected_jobs:96,jobs:jobs.length,skipped,accounts,datasets:SOURCES,sizes:SIZES,metrics:METRICS,
        ssl_epochs:1000,ssl_encoders:96,probe_epochs:100,labels:[1,5],probes_per_label:3,expected_probes:576,
        resumes_generation:jobs.filter(j=>j.resume_generation).map(j=>j.name),tables_modified:false};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.account,j.seconds,j.manifest].join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    fs.writeFileSync(path.join(batch,'preflight.tsv'),[...preflight.values()].join('\n')+'\n',{flag:'wx'});
    console.log(JSON.stringify({...coverage,skipped:skipped.length,shortest_minutes:jobs.length?Math.ceil(Math.min(...jobs.map(j=>j.seconds))/60):0,
        longest_minutes:jobs.length?Math.ceil(Math.max(...jobs.map(j=>j.seconds))/60):0}));
}
if(require.main===module)main(process.argv[2]);
