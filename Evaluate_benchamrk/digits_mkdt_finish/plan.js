// Login-safe planning: metadata, logs and scalar checkpoint headers only.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict'),cp=require('child_process');
const {epochHeader}=require('../cross_arch_ablation/plan');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(v=>Number.isFinite(v)&&v>0));a.sort((a,b)=>a-b);return a[Math.ceil(.95*a.length)-1];};
function iteration(file){
    const b=cp.execFileSync('unzip',['-p',file,'*/data.pkl'],{maxBuffer:4*1024*1024});
    const at=b.indexOf(Buffer.from('iteration'));assert.ok(at>=0&&at<65536);
    let i=at+9;if(b[i]===0x71)i+=2;else if(b[i]===0x72)i+=5;else if(b[i]===0x94)i++;
    if(b[i]===0x4b)return b[i+1];if(b[i]===0x4d)return b.readUInt16LE(i+1);
    if(b[i]===0x4a)return b.readInt32LE(i+1);throw Error('Unknown iteration encoding');
}
function main(batch){
    batch=path.resolve(batch);assert.equal(path.dirname(batch),path.join(HERE,'batches'));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const digits=read(path.join(BENCH,'digits_mkdt/recipe.json')),fashion=read(path.join(BENCH,'fashionmnist/mkdt/recipe.json'));
    for(const recipe of [digits,fashion])for(const [file,sha]of Object.entries(recipe.code_hashes))assert.equal(hash(path.join(ROOT,file)),sha,'Prerequisite implementation changed: '+file);
    const extra=['distill.py','Evaluate_benchamrk/digits_mkdt_finish/worker.py','Evaluate_benchamrk/digits_mkdt_finish/plan.js',
        'Evaluate_benchamrk/submit_digits_mkdt_finish_vulcan.sh','Evaluate_benchamrk/digits_subset_simclr2k/worker.py',
        'Evaluate_benchamrk/mkdt_02_jobs/check_resume.py','Evaluate_benchamrk/digits_mkdt/recipe.json',
        'Evaluate_benchamrk/fashionmnist/mkdt/recipe.json'];
    const files=[...new Set([...Object.keys(digits.code_hashes),...Object.keys(fashion.code_hashes),...extra])];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const jobs=[],active=[],parents={};
    function add(name,stage,seconds,dependencies=[],rest={}){
        if(queue.has(name)){active.push({name,id:queue.get(name)});return name;}
        seconds=Math.ceil(seconds);assert.ok(seconds>0&&seconds<604800);
        const job={name,stage,seconds,dependencies,account:stage==='prerequisite'?'aip-yiweilu':'aip-boyuwang',
            buffer_seconds:60,startup_seconds:120,code_hashes,...rest};
        const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        jobs.push({...job,manifest});return name;
    }
    const root=path.join(BENCH,'digits_mkdt'),teacher=path.join(root,'teachers/MNIST'),timing=read(path.join(root,'timing.json')).MNIST;
    let parent=null;
    if(!fs.existsSync(path.join(teacher,'complete.json'))){
        const cfg=read(path.join(teacher,'config.json'));
        const epoch=epochHeader(path.join(teacher,'training.pt'),{signature:cfg.identity.recipe_sha256});
        assert.ok(epoch>=1&&epoch<=1000);
        const rates=fs.readFileSync(path.join(teacher,'ssl.jsonl'),'utf8').trim().split('\n').slice(-100).map(l=>JSON.parse(l).seconds);
        parent=add('dmk_MNIST_teacher','prerequisite',(1000-epoch)*p95(rates)+180,[],{phase:'teacher',start:0,source:'MNIST',resume_epoch:epoch,gpu:true});
    }
    if(!fs.existsSync(path.join(root,'target_rep/barlow_twins/MNIST_target_rep_train.json')))
        parent=add('dmk_MNIST_features','prerequisite',timing.features,parent?[parent]:[],{phase:'features',start:0,source:'MNIST',gpu:true});
    const shards=[];
    for(let start=0;start<100;start+=5){
        if(fs.existsSync(path.join(root,'work/MNIST',`shard_${String(start).padStart(3,'0')}.json`)))continue;
        shards.push(add('dmk_MNIST_experts_'+start,'prerequisite',timing.experts,parent?[parent]:[],{phase:'experts',start,source:'MNIST',gpu:true}));
    }
    if(!fs.existsSync(path.join(root,'work/MNIST/init_complete.json')))
        parents.MNIST=add('dmk_MNIST_init','prerequisite',timing.init,shards.length?shards:parent?[parent]:[],{phase:'init',start:0,source:'MNIST',gpu:false});
    for(const [source,work,name]of [['SVHN','digits_mkdt/work/SVHN','dmk_SVHN_init'],['FashionMNIST','fashionmnist/mkdt/work/FashionMNIST','fashion_init']]){
        if(fs.existsSync(path.join(BENCH,work,'init_complete.json')))parents[source]=null;
        else{assert.ok(queue.has(name),'Missing active MKDT prerequisite: '+name);parents[source]=name;}
    }
    let checkParent=null;const checked=path.join(HERE,'checked.json');
    if(!fs.existsSync(checked)||read(checked).worker_sha256!==code_hashes['Evaluate_benchamrk/digits_mkdt_finish/worker.py'])
        checkParent=add('dmkf_check','check',180,[],{gpu:true});
    const costs=read(path.join(ROOT,'CVPR_tables/efficiency/generation_cost_sources.json')).mkdt.datasets;
    const smallRate=Math.max(...costs.CIFAR10.generation['0.2'].chunk_seconds,...costs.CIFAR100.generation['0.2'].chunk_seconds)/1000;
    const largeRate=Math.max(...['CIFAR10','CIFAR100'].flatMap(s=>[1,2].map(p=>costs[s].generation[p].seconds/5000)))*256/250;
    for(const source of ['MNIST','FashionMNIST','SVHN'])for(const subset of [.2,1,2]){
        const recipe=source==='FashionMNIST'?fashion:digits,count=Math.floor(recipe.datasets[source].train_count*subset/100);
        const dir=path.join(HERE,'synthetic',source,'ConvNet','pt'+subset),done=path.join(dir,'complete.json'),state=path.join(dir,'training.pt');
        const epoch=fs.existsSync(state)?iteration(state):0;assert.ok(epoch>=0&&epoch<=5000);
        const config=path.join(dir,'generation_config.json');
        if(fs.existsSync(config))assert.deepEqual(read(config).code_hashes,code_hashes,'Generation implementation changed');
        let previous=parents[source]||null;
        if(!fs.existsSync(done))for(let stop=1000;stop<=5000;stop+=1000){
            if(epoch>=stop&&stop<5000)continue;
            const start=Math.max(epoch,stop-1000),rate=subset===.2?smallRate*count/100:largeRate;
            previous=add(`dmkf_${source}_p${String(subset).replace('.','p')}_g${stop}`,'generation',
                Math.max(0,stop-start)*rate+180,[...new Set([previous,checkParent].filter(Boolean))],{source,subset,stop,gpu:true});
        }else{assert.ok(read(done).complete&&read(done).iteration===5000);previous=null;}
        const out=path.join(HERE,'results',source,'ResNet18','pt'+subset,'simclr/ssl_seed_000');
        if(fs.existsSync(path.join(out,'complete.json'))){assert.ok(read(path.join(out,'complete.json')).complete);continue;}
        const baseline=path.join(BENCH,source==='FashionMNIST'?'fashionmnist/subset':'digits_subset_simclr2k',
            'results',source,'ResNet18/random','pt'+subset,'simclr/ssl_seed_000');
        const records=fs.readFileSync(path.join(baseline,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(JSON.parse);
        let sslEpoch=0,probeSeconds=0;const cfg=path.join(out,'config.json'),sslDone=path.join(out,'ssl_complete.json');
        if(fs.existsSync(cfg))assert.deepEqual(read(cfg).signature.driver_code_hashes,code_hashes);
        if(fs.existsSync(sslDone)){assert.ok(read(sslDone).complete);assert.equal(read(sslDone).epochs,2000);sslEpoch=2000;}
        else if(fs.existsSync(path.join(out,'training.pt')))sslEpoch=epochHeader(path.join(out,'training.pt'),read(cfg).identity);
        for(const labels of [1,5])for(let probe=0;probe<5;probe++){
            const name=`${source}_lbl${labels}_probe${probe}.json`,file=path.join(out,'evaluations',name);
            if(fs.existsSync(file)){const p=read(file);assert.ok(p.complete&&p.encoder_unchanged);continue;}
            probeSeconds+=read(path.join(baseline,'evaluations',name)).seconds;
        }
        add(`dmkf_${source}_p${String(subset).replace('.','p')}_ssl`,'evaluation',
            (2000-sslEpoch)*p95(records.map(r=>r.seconds))+probeSeconds+180,
            [...new Set([previous,checkParent].filter(Boolean))],{source,subset,gpu:true});
    }
    const known=new Set(queue.keys());for(const j of jobs){for(const dep of j.dependencies)assert.ok(known.has(dep),'Unresolved dependency '+dep);known.add(j.name);}
    assert.ok(jobs.filter(j=>j.stage==='generation').length<=45);
    assert.ok(jobs.filter(j=>j.stage==='evaluation').length<=9);
    const coverage={synthetic_sets:9,distillation_iterations:5000,chunk_iterations:1000,ssl_epochs:2000,
        encoders:9,linear_epochs:100,probes_per_label_budget:5,label_percentages:[1,5],total_probes:90,
        jobs:jobs.map(({manifest,code_hashes,...j})=>j),active,scope:'main table only'};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.stage,j.account,j.seconds,j.dependencies.join(',')||'-',j.manifest].join('\t')).join('\n')+'\n',{flag:'wx'});
    console.log(JSON.stringify({jobs:jobs.length,prerequisites:jobs.filter(j=>j.stage==='prerequisite').length,
        generation:jobs.filter(j=>j.stage==='generation').length,evaluation:jobs.filter(j=>j.stage==='evaluation').length,
        check:jobs.filter(j=>j.stage==='check').length,active:active.length}));
}
if(require.main===module)main(process.argv[2]);
module.exports={iteration};
