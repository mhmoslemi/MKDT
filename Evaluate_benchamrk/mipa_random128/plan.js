const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const SOURCES=['CIFAR10','CIFAR100','TinyImageNet'],SIZES=[.2,1,2],ITERATIONS={'.2':10000,'0.2':10000,'1':8000,'2':6000};
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=v=>{assert.ok(v.length&&v.every(x=>Number.isFinite(x)&&x>0));return v.sort((a,b)=>a-b)[Math.ceil(.95*v.length)-1];};
const key=(source,subset)=>source+'_p'+String(subset).replace('.','p');
const folder=(source,subset)=>path.join(HERE,'results',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
const synthetic=(source,subset)=>path.join(ROOT,'OT-SSL-DD/ours_syntehtic_random128',source,'ResNet18BN','pt'+subset);

function generation_args(source,subset){
    return {dataset:source,percentage:subset,model:'ResNet18BN',seed:0,Iteration:ITERATIONS[subset],lr_img:.001,
        batch_real:128,temperature:.2,disp_weight:0,pair_weight:0,num_random_networks:1,random_models:'ResNet18BN',
        dsa:true,distill_aug_strategy:'color_crop_cutout_flip_scale_rotate',distill_aug_mode:'M',teacher_source:'pretrained',
        teacher_seed:[source==='TinyImageNet'?1:0],teacher_ssl:'simclr',
        teacher_path:path.join(BENCH,'full_ssl_checkpoints_1000ep_3seeds'),feature_batch:128};
}

function downstream_cells(){
    const text=fs.readFileSync(path.join(ROOT,'CVPR_tables/cifar_downstream.tex'),'utf8');
    for(const source of ['CIFAR10','CIFAR100','Tiny ImageNet'])assert.ok(text.includes(source+' pretraining'));
    const header=text.split('\n').find(l=>l.startsWith('& & & CIFAR100 &'));
    assert.ok(header,'Cannot identify downstream columns');
    const targets=header.split('&').slice(3).map(s=>s.replace(/\\shortstack|[{}\\\s]/g,''));
    assert.equal(targets.length,12);
    const cells=[];let labels,subset;
    for(const line of text.split('\n')){
        let m=line.match(/\\multirow\{\d+\}\{\*\}\{([15])\\%\}/);if(m)labels=+m[1];
        m=line.match(/\\multirow\{\d+\}\{\*\}\{(0\.2|1|2)\}/);if(m)subset=+m[1];
        if(!/&\s*MIPA\s*&/.test(line))continue;
        assert.ok([1,5].includes(labels)&&[.2,1].includes(subset));
        targets.forEach((target,i)=>cells.push({source:SOURCES[Math.floor(i/4)],subset,target,labels}));
    }
    assert.equal(cells.length,48);assert.equal(new Set(cells.map(c=>JSON.stringify(c))).size,48);
    return cells;
}

function generation_rate(file){
    const rates=[];let previous;
    for(const line of fs.readFileSync(file,'utf8').split('\n')){
        const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,iteration:+m[3]};
        if(previous&&current.time>previous.time&&current.iteration>previous.iteration&&current.time-previous.time<300)
            rates.push((current.time-previous.time)/(current.iteration-previous.iteration));
        previous=current;
    }
    return p95(rates);
}

function probe_rate(source,target,labels){
    const roots=[path.join(BENCH,'downstream_simclr2k_1ssl5probes/results',source,'ResNet18/random/pt0.2/simclr/ssl_seed_000'),
        path.join(BENCH,'ours_eval_results_simclr2k_barlow10k_2ssl3probes_downstream',source,'ResNet18BN/pt0.2/simclr/ssl_seed_000')];
    for(const root of roots){
        const values=[];
        for(let seed=0;seed<5;seed++){
            const file=path.join(root,'evaluations',`${target}_lbl${labels}_probe${seed}.json`);
            if(fs.existsSync(file)){const r=read(file);if(r.complete&&r.encoder_unchanged)values.push(r.seconds);}
        }
        if(values.length)return p95(values);
    }
    throw Error('No measured probe timing: '+[source,target,labels].join('/'));
}

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(HERE,'batches')+path.sep));
    const files=fs.readdirSync(BENCH).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f).concat([
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        'OT-SSL-DD/method1.py','OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py',
        'OT-SSL-DD/utils.py','OT-SSL-DD/networks.py','Evaluate_benchamrk/mipa_random128/worker.py',
        'Evaluate_benchamrk/mipa_random128/plan.js','Evaluate_benchamrk/submit_mipa_random128_vulcan.sh']);
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const timingFile=path.join(HERE,'timing.json'),timing=fs.existsSync(timingFile)?read(timingFile):null;
    if(timing)assert.deepEqual(timing.code_hashes,code_hashes);
    const units=[];
    for(const source of SOURCES)for(const subset of SIZES){
        const previousRoot=source==='TinyImageNet'||subset===2?'subset_simclr2k_1ssl5probes':'cifar_small_simclr2k_1ssl5probes';
        const origin=path.join(BENCH,previousRoot,'results',source,'ResNet18/random','pt'+subset,'simclr/ssl_seed_000/config.json');
        const cfg=read(origin),indices=cfg.identity.selection.indices;
        assert.equal(cfg.identity.seed,0);assert.equal(cfg.identity.selection_method,'random');
        assert.equal(cfg.signature.source,source);assert.equal(cfg.signature.subset,subset);
        assert.equal(indices.length,(source==='TinyImageNet'?100000:50000)*subset/100);
        assert.equal(new Set(indices).size,indices.length);
        assert.ok(indices.every(i=>Number.isInteger(i)&&i>=0&&i<(source==='TinyImageNet'?100000:50000)));
        const selection={method:'saved_random_subset',seed:0,config:path.relative(ROOT,origin),config_sha256:hash(origin),
            source_fingerprint:cfg.signature.source_fingerprint,selection_sha256:cfg.identity.selection.selection_sha256,indices};
        const teacherSeed=source==='TinyImageNet'?1:0;
        const teacherFile=path.join(BENCH,'full_ssl_checkpoints_1000ep_3seeds',source,'ResNet18/full/pt100/simclr',`seed_00${teacherSeed}`,'encoder.pt');
        const teacherMarker=path.join(path.dirname(teacherFile),'ssl_complete.json');assert.ok(read(teacherMarker).complete);
        const teacher={path:path.relative(ROOT,teacherFile),file_sha256:hash(teacherFile),marker_sha256:hash(teacherMarker)};
        units.push({source,subset,generation_args:generation_args(source,subset),selection,teacher,code_hashes});
    }
    const cells=downstream_cells(),jobs=[],parents=new Map();
    function add(unit,stage,target,labels,seconds,parent){
        const suffix=stage==='generation'?'gen':stage==='ssl_main'?'ssl':target+'_b'+labels;
        const name='r128_'+key(unit.source,unit.subset)+'_'+suffix;
        if(queue.has(name)){parents.set(name,queue.get(name));console.log('SKIP active '+name);return name;}
        assert.ok(Number.isFinite(seconds)&&seconds>0&&seconds<=604800);
        const job={...unit,name,stage,target,labels,account:stage==='downstream'?'aip-yiweilu':'aip-boyuwang',
            parents:parent?[parent]:[],seconds:Math.ceil(seconds),buffer_seconds:60,
            evaluation:{ssl:'simclr',ssl_epochs:2000,ssl_seed:0,linear_epochs:100,probe_seeds:[0,1,2,3,4],main_labels:[1,5]}};
        const file=path.join(batch,name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        jobs.push({...job,manifest:file});parents.set(name,null);return name;
    }
    for(const unit of units){
        const {source,subset}=unit,out=folder(source,subset),gen=synthetic(source,subset);
        const contract={generation_args:unit.generation_args,selection:unit.selection,teacher:unit.teacher,code_hashes};
        let generationDone=false,iteration=0;
        const state=path.join(gen,'generation_state.json');
        if(fs.existsSync(state)){
            const s=read(state);assert.deepEqual(s.contract,contract);iteration=s.iteration;
            assert.ok(iteration>=0&&iteration<=ITERATIONS[subset]);generationDone=s.complete;
            if(generationDone){assert.equal(iteration,ITERATIONS[subset]);assert.ok(fs.existsSync(path.join(gen,`res_IIC_${source}_ResNet18BN_${subset}percent_final.pt`)));}
        }
        const matched=path.join(BENCH,'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
        const sslRate=p95(fs.readFileSync(path.join(matched,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds));
        const probeRates=Object.fromEntries([1,5].map(lbl=>[lbl,p95([0,1,2].map(seed=>read(path.join(matched,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`)).seconds))]));
        // Before calibration, the unchanged batch-1024 timing is only a conservative planning fallback.
        const generationRate=timing?timing.units[key(source,subset)].generation_seconds_p95:generation_rate(path.join(ROOT,`OT-SSL-DD/ours_syntehtic/logs/${source}_ResNet18BN_pt${subset}.log`));
        const generationParent=generationDone?null:add(unit,'generation',null,null,(ITERATIONS[subset]-iteration)*generationRate+80,null);
        let sslDone=false,epoch=0,missing={1:5,5:5};
        const markerFile=path.join(out,'ssl_complete.json');
        if(fs.existsSync(markerFile)){const m=read(markerFile);assert.ok(m.complete&&m.epochs===2000);epoch=2000;}
        else if(fs.existsSync(path.join(out,'training.pt'))&&fs.existsSync(path.join(out,'ssl.jsonl'))){
            const lines=fs.readFileSync(path.join(out,'ssl.jsonl'),'utf8').trim().split('\n').filter(Boolean);
            if(lines.length)epoch=Math.floor(JSON.parse(lines.at(-1)).epoch/500)*500;
        }
        for(const lbl of [1,5])for(let seed=0;seed<5;seed++){
            const f=path.join(out,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`);
            if(fs.existsSync(f)){const p=read(f);assert.ok(p.complete&&p.encoder_unchanged&&epoch===2000);missing[lbl]--;}
        }
        const done=path.join(out,'complete.json');
        if(fs.existsSync(done)){const m=read(done);assert.ok(m.complete&&m.ssl_epochs===2000&&missing[1]===0&&missing[5]===0);assert.deepEqual(m.generation_contract,contract);sslDone=true;}
        const sslParent=sslDone?null:add(unit,'ssl_main',null,null,(2000-epoch)*sslRate+missing[1]*probeRates[1]+missing[5]*probeRates[5]+80,generationParent);
        for(const cell of cells.filter(c=>c.source===source&&c.subset===subset)){
            const {target,labels}=cell;let missingProbes=5;
            const summary=path.join(out,`summary_${target}_lbl${labels}.json`);
            if(fs.existsSync(summary)){const s=read(summary);assert.ok(s.complete&&s.completed_probes===5&&s.ssl_epochs===2000);assert.deepEqual(s.generation_args,unit.generation_args);assert.deepEqual(s.selection,unit.selection);console.log('SKIP complete '+[source,subset,target,labels].join('/'));continue;}
            for(let seed=0;seed<5;seed++){const f=path.join(out,'evaluations',`${target}_lbl${labels}_probe${seed}.json`);if(fs.existsSync(f)){const p=read(f);assert.ok(p.complete&&p.encoder_unchanged);missingProbes--;}}
            add(unit,'downstream',target,labels,missingProbes*probe_rate(source,target,labels)+80,sslParent);
        }
    }
    const known=new Set(queue.keys());let tsv='';
    for(const job of jobs){for(const parent of job.parents)assert.ok(known.has(parent),'Missing parent: '+parent);known.add(job.name);
        tsv+=[job.name,job.stage,job.account,job.seconds,job.parents[0]||'-',job.manifest].join('\t')+'\n';}
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const coverage={expected_generation_jobs:9,expected_ssl_main_jobs:9,expected_downstream_jobs:48,main_cells:18,downstream_cells:cells,
        generation_jobs:jobs.filter(j=>j.stage==='generation').length,ssl_main_jobs:jobs.filter(j=>j.stage==='ssl_main').length,
        downstream_jobs:jobs.filter(j=>j.stage==='downstream').length,total_jobs:jobs.length,
        initialization:'exact saved Random seed-0 indices',generation_batch:128,generation_lr:.001,iterations:ITERATIONS,
        ssl_epochs:2000,linear_epochs:100,probes_per_cell:5,buffer_seconds:60,calibrated:Boolean(timing),
        accounts:{'aip-boyuwang':jobs.filter(j=>j.account==='aip-boyuwang').length,'aip-yiweilu':jobs.filter(j=>j.account==='aip-yiweilu').length},
        downstream_table_sha256:hash(path.join(ROOT,'CVPR_tables/cifar_downstream.tex')),
        units:units.map(u=>({source:u.source,subset:u.subset,count:u.selection.indices.length,teacher_seed:u.generation_args.teacher_seed[0]})),
        walltimes:jobs.map(j=>({name:j.name,stage:j.stage,seconds:j.seconds,account:j.account}))};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({jobs:jobs.length,generation:coverage.generation_jobs,ssl_main:coverage.ssl_main_jobs,downstream:coverage.downstream_jobs,accounts:coverage.accounts,calibrated:coverage.calibrated}));
}
if(require.main===module)main(process.argv[2]);
