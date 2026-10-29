const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const p95=values=>{assert.ok(values.length&&values.every(v=>Number.isFinite(v)&&v>0));return values.sort((a,b)=>a-b)[Math.ceil(.95*values.length)-1];};
const variants={random_seed_000:{method:'random',seed:0,kmeans_iterations:0},
    random_seed_001:{method:'random',seed:1,kmeans_iterations:0},
    random_seed_002:{method:'random',seed:2,kmeans_iterations:0},
    kmeans100_seed_000:{method:'kmeans',seed:0,kmeans_iterations:100}};
const generation_args={dataset:'CIFAR10',percentage:.2,model:'ResNet18BN',seed:0,Iteration:10000,
    lr_img:.001,batch_real:1024,temperature:.2,disp_weight:0,pair_weight:0,num_random_networks:1,
    random_models:'ResNet18BN',dsa:true,distill_aug_strategy:'color_crop_cutout_flip_scale_rotate',
    distill_aug_mode:'M',teacher_source:'pretrained',teacher_seed:[0],teacher_ssl:'simclr',
    teacher_path:path.join(bench,'full_ssl_checkpoints_1000ep_3seeds'),feature_batch:128};
function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const files=fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f).concat([
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json','OT-SSL-DD/method1.py',
        'OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py','OT-SSL-DD/utils.py','OT-SSL-DD/networks.py',
        'Evaluate_benchamrk/mipa_init_ablation/worker.py','Evaluate_benchamrk/mipa_init_ablation/plan.js',
        'Evaluate_benchamrk/submit_mipa_init_ablation_vulcan.sh']);
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
    const teacherFile=path.join(bench,'full_ssl_checkpoints_1000ep_3seeds/CIFAR10/ResNet18/full/pt100/simclr/seed_000/encoder.pt');
    const marker=path.join(path.dirname(teacherFile),'ssl_complete.json');assert.ok(read(marker).complete);
    const teacher={path:path.relative(root,teacherFile),file_sha256:hash(teacherFile),marker_sha256:hash(marker)};
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').trim().split('\n').filter(Boolean).map(line=>line.split('|').map(s=>s.trim())));
    const baselineFile=path.join(root,'OT-SSL-DD/ours_syntehtic/CIFAR10/ResNet18BN/pt0.2/res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt');
    assert.ok(fs.statSync(baselineFile).size>0);
    const baseline={method:'kmeans',kmeans_iterations:5,seed:0,synthetic_file:path.relative(root,baselineFile),file_sha256:hash(baselineFile),new_jobs:0};
    let previous;const rates=[];
    for(const line of fs.readFileSync(path.join(root,'OT-SSL-DD/ours_syntehtic/logs/CIFAR10_ResNet18BN_pt0.2.log'),'utf8').split('\n')){
        const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,iteration:+m[3]};
        if(previous&&current.iteration>previous.iteration&&current.time>previous.time&&current.time-previous.time<300)rates.push((current.time-previous.time)/(current.iteration-previous.iteration));
        previous=current;
    }
    const generationRate=p95(rates);
    const reference=path.join(bench,'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/CIFAR10/ResNet18BN/pt0.2/simclr/ssl_seed_000');
    const sslRate=p95(fs.readFileSync(path.join(reference,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(line=>JSON.parse(line).seconds));
    const probeRate=p95([0,1,2].map(seed=>{const r=read(path.join(reference,`evaluations/CIFAR10_lbl1_probe${seed}.json`));assert.ok(r.complete&&r.encoder_unchanged);return r.seconds;}));
    const states={};
    for(const [variant,initialization]of Object.entries(variants)){
        const folder=path.join(root,'OT-SSL-DD/ours_syntehtic_init_ablation/CIFAR10/ResNet18BN/pt0.2',variant);
        const file=path.join(folder,'generation_state.json');states[variant]=fs.existsSync(file)?read(file):null;
        if(states[variant]){
            assert.deepEqual(states[variant].contract,{generation_args,initialization,teacher,code_hashes});
            assert.ok(states[variant].iteration>=0&&states[variant].iteration<=10000);
            if(states[variant].complete){assert.equal(states[variant].iteration,10000);for(const f of ['res_IIC_CIFAR10_ResNet18BN_0.2percent_final.pt','vis_IIC_CIFAR10_ResNet18BN_0.2percent_iter1.png','vis_IIC_CIFAR10_ResNet18BN_0.2percent_iter10000.png'])assert.ok(fs.existsSync(path.join(folder,f)));}
        }
    }
    const groups=[{name:'mipai_random3_gen',account:'aip-boyuwang',variants:['random_seed_000','random_seed_001','random_seed_002']},
        {name:'mipai_kmeans100_gen',account:'aip-yiweilu',variants:['kmeans100_seed_000']}];
    const jobs=[];
    for(const group of groups){
        group.done=group.variants.every(v=>states[v]?.complete);
        if(group.done||queue.has(group.name))continue;
        const remaining=group.variants.reduce((n,v)=>n+10000-(states[v]?.iteration||0),0);
        const initializationSeconds=group.variants.includes('kmeans100_seed_000')&&!states.kmeans100_seed_000?120:0;
        jobs.push({...group,stage:'generation',parents:[],seconds:Math.ceil(remaining*generationRate+initializationSeconds+20+60)});
    }
    for(const [index,variant]of Object.keys(variants).entries()){
        const name='mipai_'+variant+'_eval';if(queue.has(name))continue;
        const folder=path.join(here,'results',variant,'simclr/ssl_seed_000'),summaryFile=path.join(folder,'summary_lbl1.json');
        if(fs.existsSync(summaryFile)){const s=read(summaryFile);assert.ok(s.complete&&s.completed_probes===5&&s.ssl_epochs===2000);assert.deepEqual(s.initialization,variants[variant]);continue;}
        const group=groups.find(g=>g.variants.includes(variant));
        let epoch=0,missing=5;
        const sslMarker=path.join(folder,'ssl_complete.json');
        if(fs.existsSync(sslMarker)){const m=read(sslMarker);assert.ok(m.complete&&m.epochs===2000);epoch=2000;}
        else if(fs.existsSync(path.join(folder,'ssl.jsonl'))){const rows=fs.readFileSync(path.join(folder,'ssl.jsonl'),'utf8').trim().split('\n').filter(Boolean);if(rows.length)epoch=Math.floor(JSON.parse(rows.at(-1)).epoch/500)*500;}
        for(let seed=0;seed<5;seed++){const f=path.join(folder,'evaluations',`CIFAR10_lbl1_probe${seed}.json`);if(fs.existsSync(f)){assert.ok(epoch===2000&&read(f).complete);missing--;}}
        jobs.push({name,stage:'evaluation',variants:[variant],account:index%2?'aip-yiweilu':'aip-boyuwang',
            parents:group.done?[]:[group.name],seconds:Math.ceil((2000-epoch)*sslRate+missing*probeRate+20+60)});
    }
    const known=new Set(queue.keys());let tsv='';
    for(const job of jobs){
        for(const parent of job.parents)assert.ok(known.has(parent),'Missing generation dependency: '+parent);
        assert.ok(job.seconds>0&&job.seconds<3*3600);
        const config={...job,generation_args,teacher,code_hashes,buffer_seconds:60,
            evaluation:{method:'simclr',ssl_epochs:2000,ssl_seeds:[0],probe_epochs:100,probe_seeds:[0,1,2,3,4],labels:[1]}};
        const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(config,null,2)+'\n',{flag:'wx'});
        tsv+=[job.name,job.stage,job.account,job.seconds,job.parents.join(',')||'-',file].join('\t')+'\n';known.add(job.name);
    }
    assert.deepEqual(groups[0].variants.length,3);assert.equal(groups[1].variants.length,1);
    assert.equal(Object.keys(variants).length,4);assert.ok(Object.values(variants).every(v=>v.kmeans_iterations!==5));
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const coverage={generation_jobs:jobs.filter(j=>j.stage==='generation').length,evaluation_jobs:jobs.filter(j=>j.stage==='evaluation').length,
        variants,generation_args,evaluation:{ssl_epochs:2000,ssl_seeds:[0],probe_epochs:100,probe_seeds:[0,1,2,3,4],labels:[1]},
        existing_five_iteration_reference:baseline,accounts:Object.fromEntries(['aip-boyuwang','aip-yiweilu'].map(a=>[a,jobs.filter(j=>j.account===a).length])),
        timing:{generation_iteration_seconds_p95:generationRate,ssl_epoch_seconds_p95:sslRate,probe_seconds_p95:probeRate,
            estimated_kmeans100_initialization_seconds:120,buffer_seconds:60}};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify(coverage,null,2));
}
if(require.main===module)main(process.argv[2]);
