// Login-safe planning: read existing timing logs; never load torch on login.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(n=>Number.isFinite(n)&&n>0),'Missing runtime measurements');return a.sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];};

function main(){
    const batch=path.resolve(process.argv[2]),queue=new Map(fs.readFileSync(process.argv[3],'utf8').trim().split('\n').filter(Boolean).map(l=>l.trim().split('|').map(s=>s.trim())));
    assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const files=[...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py','OT-SSL-DD/method1.py','OT-SSL-DD/utils.py',
        'OT-SSL-DD/networks.py','OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py',
        'Evaluate_benchamrk/protocol.json','Evaluate_benchamrk/eval_jobs/interactive_ours.py',
        'Evaluate_benchamrk/eval_jobs/worker.py','Evaluate_benchamrk/random_teacher_ablation/worker.py',
        'Evaluate_benchamrk/random_teacher_ablation/plan.js','Evaluate_benchamrk/submit_cifar10_random_network_ablation_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
    const jobs=[],timing=[];
    for(const size of [0.2,1]){
        const iterations=size===0.2?10000:8000,tag=String(size).replace('.','p'),rates=[];
        let prev;
        for(const line of fs.readFileSync(path.join(root,`OT-SSL-DD/ours_syntehtic/logs/CIFAR10_ResNet18BN_pt${size}.log`),'utf8').split('\n')){
            const m=line.match(/^\[([^\]]+)\] iter = (\d+)/);if(!m)continue;
            const t=Date.parse(m[1].replace(' ','T')+'Z'),i=+m[2];
            if(prev&&i>prev.i&&i-prev.i<=100&&t>prev.t)rates.push((t-prev.t)/1000/(i-prev.i));prev={t,i};
        }
        const generation_args={dataset:'CIFAR10',percentage:size,model:'ResNet18BN',seed:0,Iteration:iterations,
            lr_img:0.001,batch_real:1024,temperature:0.2,disp_weight:0,pair_weight:0,num_random_networks:1,
            random_models:'ResNet18BN',dsa:true,distill_aug_strategy:'color_crop_cutout_flip_scale_rotate',
            distill_aug_mode:'M',teacher_source:'fresh',teacher_seed:[0],teacher_ssl:'simclr',
            teacher_path:path.join(bench,'full_ssl_checkpoints_1000ep_3seeds'),feature_batch:128};
        const sslRates=[],probeRates={};
        for(const kind of ['main','downstream'])for(const seed of [0,1]){
            const dir=path.join(bench,`ours_eval_results_simclr2k_barlow10k_2ssl3probes_${kind}/CIFAR10/ResNet18BN/pt${size}/simclr/ssl_seed_${String(seed).padStart(3,'0')}`);
            const log=path.join(dir,'ssl.jsonl');
            if(kind==='main'&&fs.existsSync(log))for(const line of fs.readFileSync(log,'utf8').trim().split('\n').slice(-200)){
                const r=JSON.parse(line);if(r.seconds>0)sslRates.push(r.seconds);
            }
            const evals=path.join(dir,'evaluations');if(!fs.existsSync(evals))continue;
            for(const f of fs.readdirSync(evals).filter(f=>/^(CIFAR10|CIFAR100|TinyImageNet)_.*\.json$/.test(f))){
                const r=read(path.join(evals,f));if(r.complete&&r.seconds>0)(probeRates[r.expected.target+'|'+r.expected.label_percentage]??=[]).push(r.seconds);
            }
        }
        const genRate=p95(rates),sslRate=p95(sslRates);
        let probeSeconds=0;for(const target of ['CIFAR10','CIFAR100','TinyImageNet'])for(const label of [1,5])probeSeconds+=3*p95(probeRates[target+'|'+label]||[]);
        // Fresh CPU initialization is not in the old frozen-teacher logs; estimate 0.07 s per new network.
        const genSeconds=Math.ceil(iterations*(genRate+0.07)+23+60);
        const common={size,generation_args,code_hashes,buffer_seconds:60};
        const gen=`ab_rnet_C10_p${tag}_gen`;
        jobs.push({...common,name:gen,stage:'generation',seed:0,account:'aip-yiweilu',seconds:genSeconds,dependencies:[]});
        for(const seed of [0,1]){
            const ssl=`ab_rnet_C10_p${tag}_ssl_s${seed}`;
            jobs.push({...common,name:ssl,stage:'ssl',seed,account:'aip-boyuwang',seconds:Math.ceil(2000*sslRate+60),dependencies:[gen]});
            jobs.push({...common,name:`ab_rnet_C10_p${tag}_probe_s${seed}`,stage:'probe',seed,account:'aip-boyuwang',seconds:Math.ceil(probeSeconds+60),dependencies:[ssl]});
        }
        timing.push({size,generation_seconds_per_iteration_p95:genRate,estimated_new_network_seconds:0.07,
            estimated_initialization_seconds:23,ssl_seconds_per_epoch_p95:sslRate,probe_seconds_per_ssl_seed:probeSeconds});
    }
    assert.equal(jobs.length,10);assert.equal(jobs.filter(j=>j.account==='aip-boyuwang').length,8);
    assert.equal(jobs.filter(j=>j.stage==='ssl').length,4);assert.equal(jobs.filter(j=>j.stage==='probe').length*18,72);
    const done=new Set(),names=new Set();
    for(const task of jobs){
        assert.ok(!names.has(task.name));for(const dep of task.dependencies)assert.ok(names.has(dep));names.add(task.name);
        const status=path.join(here,'status',task.name+'.json');if(!fs.existsSync(status))continue;
        const r=read(status);assert.deepEqual(r.code_hashes,code_hashes,'Ablation code changed; do not mix runs');
        if(r.complete&&Object.entries(r.artifacts).every(([f,sha])=>fs.existsSync(path.join(root,f))&&hash(path.join(root,f))===sha))done.add(task.name);
    }
    let tsv='';const submitted=[];
    for(const task of jobs){
        if(done.has(task.name)||queue.has(task.name))continue;
        task.dependencies=task.dependencies.filter(n=>!done.has(n));
        const f=path.join(batch,task.name+'.json');fs.writeFileSync(f,JSON.stringify(task,null,2)+'\n',{flag:'wx'});
        tsv+=[task.name,task.account,task.seconds,task.dependencies.join(',')||'-',f].join('\t')+'\n';submitted.push(task);
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const report={total_experiment_jobs:10,jobs_to_submit:submitted.length,already_complete:[...done],already_queued:[...queue.keys()].filter(n=>names.has(n)),
        new_wang_jobs:submitted.filter(j=>j.account==='aip-boyuwang').length,new_lu_jobs:submitted.filter(j=>j.account==='aip-yiweilu').length,
        synthetic_sets:2,ssl_encoders:4,probes:72,cells:12,ssl_epochs:2000,probe_epochs:100,buffer_seconds:60,timing};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
}
if(require.main===module)main();
