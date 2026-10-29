const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const {epochHeader}=require('../cross_arch_ablation/plan');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const COUNTS={MNIST:60000,FashionMNIST:60000,SVHN:73257},ITERATIONS={'.2':10000,'0.2':10000,1:8000,2:6000};
const p95=v=>{assert.ok(v.length&&v.every(n=>Number.isFinite(n)&&n>0));v.sort((a,b)=>a-b);return v[Math.ceil(.95*v.length)-1];};
const genArgs=(source,subset)=>({dataset:source,percentage:subset,model:'ResNet18BN',seed:0,
    Iteration:ITERATIONS[subset],lr_img:.001,batch_real:subset===.2?1024:128,temperature:.2,
    disp_weight:0,pair_weight:0,num_random_networks:1,random_models:'ResNet18BN',dsa:true,
    distill_aug_strategy:'color_crop_cutout_flip_scale_rotate',distill_aug_mode:'M',teacher_source:'pretrained',
    teacher_seed:[0],teacher_ssl:'simclr',teacher_path:path.join(BENCH,source==='FashionMNIST'?'fashionmnist/full/checkpoints':'digits_simclr/checkpoints'),feature_batch:128});

function largeBatchRate(){
    let previous;const rates=[];
    for(const line of fs.readFileSync(path.join(ROOT,'OT-SSL-DD/ours_syntehtic/logs/CIFAR10_ResNet18BN_pt0.2.log'),'utf8').split('\n')){
        const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,iteration:+m[3]};
        if(previous&&current.time>previous.time&&current.iteration>previous.iteration&&current.time-previous.time<300)
            rates.push((current.time-previous.time)/(current.iteration-previous.iteration));
        previous=current;
    }
    return p95(rates);
}

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(HERE,'batches')+path.sep));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const files=fs.readdirSync(BENCH).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f).concat([
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        'OT-SSL-DD/method1.py','OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py',
        'OT-SSL-DD/utils.py','OT-SSL-DD/networks.py','Evaluate_benchamrk/mipa_random128/worker.py',
        'Evaluate_benchamrk/digits_simclr/datasets.py','Evaluate_benchamrk/fashionmnist/dataset.py',
        'Evaluate_benchamrk/digits_mipa/worker.py','Evaluate_benchamrk/digits_mipa/plan.js',
        'Evaluate_benchamrk/cross_arch_ablation/plan.js','Evaluate_benchamrk/submit_digits_mipa_vulcan.sh']);
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const timings=read(path.join(BENCH,'mipa_random128/timing.json')),largeRate=largeBatchRate();
    const jobs=[],skipped=[],units=[];let finishedGeneration=0;
    for(const source of Object.keys(COUNTS)){
        const teacherName=source==='FashionMNIST'?'fashion_full_s0':`digits_${source}_full_s0`;
        const teacherRoot=path.join(BENCH,source==='FashionMNIST'?'fashionmnist/full/checkpoints':'digits_simclr/checkpoints');
        const teacherFolder=path.join(teacherRoot,source,'ResNet18/full/pt100/simclr/seed_000');
        const configFile=path.join(teacherFolder,'config.json'),cfg=read(configFile),markerFile=path.join(teacherFolder,'ssl_complete.json');
        const teacherReady=fs.existsSync(markerFile)&&read(markerFile).complete;
        if(teacherReady){assert.equal(read(markerFile).epochs,1000);assert.deepEqual(read(markerFile).identity,cfg.identity);assert.equal(hash(path.join(teacherFolder,'encoder.pt')),read(markerFile).checkpoint_file_sha256);}
        else assert.ok(queue.has(teacherName),'No active seed-0 teacher: '+source);
        assert.equal(cfg.signature.source,source);assert.equal(cfg.signature.ssl,'simclr');assert.equal(cfg.identity.seed,0);
        const preparedFile=path.join(BENCH,source==='FashionMNIST'?'fashionmnist/full/prepared.json':'digits_simclr/prepared.json');
        const prepared=read(preparedFile);assert.deepEqual(cfg.signature.protocol,prepared.protocol);
        const teacher={path:path.relative(ROOT,path.join(teacherFolder,'encoder.pt')),config:path.relative(ROOT,configFile),
            config_sha256:hash(configFile),identity:cfg.identity};
        for(const subset of [.2,1,2]){
            const baseline=path.join(BENCH,source==='FashionMNIST'?'fashionmnist/subset':'digits_subset_simclr2k',
                'results',source,'ResNet18/random','pt'+subset,'simclr/ssl_seed_000');
            const origin=path.join(baseline,'config.json'),old=read(origin),indices=old.identity.selection.indices;
            const count=Math.floor(COUNTS[source]*subset/100);
            assert.equal(old.identity.seed,0);assert.equal(old.identity.selection_method,'random');
            assert.equal(old.signature.source,source);assert.equal(old.signature.subset,subset);
            assert.equal(indices.length,count);assert.equal(new Set(indices).size,count);
            assert.ok(indices.every(i=>Number.isInteger(i)&&i>=0&&i<COUNTS[source]));
            assert.equal(old.signature.source_fingerprint,prepared.datasets[source].train_fingerprint);
            const selection={method:'saved_random_subset',seed:0,config:path.relative(ROOT,origin),config_sha256:hash(origin),
                source_fingerprint:old.signature.source_fingerprint,selection_sha256:old.identity.selection.selection_sha256,indices};
            const unit={source,subset,generation_args:genArgs(source,subset),selection,teacher,
                prepared_file:path.relative(ROOT,preparedFile),prepared_sha256:hash(preparedFile),code_hashes};
            units.push(unit);
            const dir=path.join(ROOT,'OT-SSL-DD/ours_syntehtic_digits_random',source,'ResNet18BN','pt'+subset);
            const out=path.join(HERE,'results',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
            let iteration=0,done=false;
            const stateFile=path.join(dir,'generation_state.json');
            if(fs.existsSync(stateFile)){
                const state=read(stateFile);assert.deepEqual(state.contract.generation_args,unit.generation_args);
                assert.deepEqual(state.contract.selection,selection);assert.deepEqual(state.contract.code_hashes,code_hashes);
                for(const [k,v] of Object.entries(teacher))assert.deepEqual(state.contract.teacher[k],v);
                iteration=state.iteration;done=state.complete;assert.ok(iteration>=0&&iteration<=ITERATIONS[subset]);
                if(done)assert.equal(iteration,ITERATIONS[subset]);
            }
            if(done)finishedGeneration++;
            function add(stage,seconds,parent){
                const name=`digm_${source}_p${String(subset).replace('.','p')}_${stage==='generation'?'gen':'ssl'}`;
                if(queue.has(name)){skipped.push({name,reason:'active',id:queue.get(name)});return name;}
                seconds=Math.ceil(seconds+80);assert.ok(seconds>0&&seconds<604800);
                const job={...unit,name,stage,seconds,buffer_seconds:60,startup_seconds:20,parent:parent||null,
                    account:seconds>10800?'aip-yiweilu':'aip-boyuwang',
                    timing_policy:'measured Random SSL/probes; same-backbone MIPA timing scaled by synthetic count'};
                const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
                jobs.push({...job,manifest});return name;
            }
            const rate=subset===.2?largeRate:timings.units['CIFAR10_p'+subset].generation_seconds_p95;
            const scaledRate=rate*Math.max(1,count/Math.floor(50000*subset/100));
            const parent=done?null:add('generation',(ITERATIONS[subset]-iteration)*scaledRate,teacherReady?null:teacherName);
            let epoch=0,missing=0,probeSeconds=0;
            const configPath=path.join(out,'config.json'),markerPath=path.join(out,'ssl_complete.json');
            const config=fs.existsSync(configPath)?read(configPath):null,marker=fs.existsSync(markerPath)?read(markerPath):null;
            if(config){assert.deepEqual(config.signature.driver_code_hashes,code_hashes);assert.deepEqual(config.signature.generation_args,unit.generation_args);assert.deepEqual(config.signature.selection,selection);}
            if(marker){assert.ok(marker.complete&&marker.epochs===2000);assert.deepEqual(marker.identity,config.identity);assert.equal(hash(path.join(out,'encoder.pt')),marker.checkpoint_file_sha256);epoch=2000;}
            else if(fs.existsSync(path.join(out,'training.pt'))){assert.ok(config);epoch=epochHeader(path.join(out,'training.pt'),config.identity);}
            assert.ok(epoch>=0&&epoch<=2000);
            for(const labels of [1,5])for(const probe of [0,1,2,3,4]){
                const file=path.join(out,'evaluations',`${source}_lbl${labels}_probe${probe}.json`);
                if(fs.existsSync(file)){
                    const p=read(file);assert.ok(marker&&p.complete&&p.encoder_unchanged);assert.deepEqual(p.expected.identity,config.identity);
                    assert.equal(p.encoder_sha256,marker.encoder_sha256);assert.deepEqual(p.expected.evaluation_code_hashes,config.signature.code_hashes);
                    assert.deepEqual([p.expected.source,p.expected.target,p.expected.seed,p.expected.label_percentage],[source,source,probe,labels]);continue;
                }
                missing++;probeSeconds+=p95([0,1,2,3,4].map(p=>read(path.join(baseline,'evaluations',`${source}_lbl${labels}_probe${p}.json`)).seconds));
            }
            if(epoch===2000&&!missing){skipped.push({source,subset,reason:'SSL and all probes complete'});continue;}
            const records=fs.readFileSync(path.join(baseline,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(JSON.parse);
            add('ssl_main',(2000-epoch)*p95(records.map(r=>r.seconds))+probeSeconds,parent);
        }
    }
    assert.equal(units.length,9);assert.equal(jobs.length+skipped.length+finishedGeneration,18);
    const known=new Set(queue.keys());for(const job of jobs){if(job.parent)assert.ok(known.has(job.parent),'Unknown dependency: '+job.parent);known.add(job.name);}
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.stage,j.account,j.seconds,j.parent||'-',j.manifest].join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify({units,jobs,skipped,expected_synthetic_sets:9,
        expected_ssl_encoders:9,expected_cells:18,expected_probes:90,ssl_epochs:2000,linear_epochs:100,
        probe_seeds:[0,1,2,3,4],labels:[1,5],table_scope:'CVPR_tables/table.tex'},null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({jobs:jobs.length,skipped:skipped.length,synthetic_sets:9,encoders:9,probes:90}));
}
if(require.main===module)main(process.argv[2]);
