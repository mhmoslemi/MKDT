// Login-safe planning: JSON/logs and the rolling checkpoint's scalar header only.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict'),cp=require('child_process');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(n=>Number.isFinite(n)&&n>0),'Missing measured runtimes');return a.sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];};
function epochHeader(file,identity){
    const b=cp.execFileSync('unzip',['-p',file,'*/data.pkl'],{maxBuffer:4*1024*1024});
    assert.ok(b.includes(Buffer.from(identity.signature)),'Rolling checkpoint signature differs');
    const at=b.indexOf(Buffer.from('epoch'));assert.ok(at>0&&at<1024);
    let i=at+5;if(b[i]===0x71)i+=2;else if(b[i]===0x72)i+=5;else if(b[i]===0x94)i++;
    if(b[i]===0x4b)return b[i+1];if(b[i]===0x4d)return b.readUInt16LE(i+1);
    if(b[i]===0x4a)return b.readInt32LE(i+1);throw Error('Unknown checkpoint epoch encoding');
}
function logs(file){return fs.existsSync(file)?fs.readFileSync(file,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];}
function main(calibration){
    const batch=path.resolve(process.argv[2]);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const queue=new Map(fs.readFileSync(process.argv[3],'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    calibration=calibration||read(path.join(here,'calibration.json'));assert.ok(calibration.complete);
    for(const [file,sha]of Object.entries(calibration.code_hashes))assert.equal(hash(path.join(root,file)),sha,'Preflight code changed: '+file);
    const sharedFiles=[...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),'networks.py','benchmarks/downstream_data.py'];
    const files=[...sharedFiles,
        'Evaluate_benchamrk/protocol.json','Evaluate_benchamrk/eval_jobs/interactive_ours.py',
        ...['models.py','worker.py','plan.js','calibrate.py'].map(f=>'Evaluate_benchamrk/cross_arch_ablation/'+f),
        'Evaluate_benchamrk/submit_cross_arch_ablation_wang_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
    const sharedHashes=Object.fromEntries(sharedFiles.map(f=>[f,code_hashes[f]]));
    const jobs=[],allNames=new Set(),probeTimes={},timings=[],groups=[];
    for(const source of ['CIFAR10','CIFAR100'])for(const size of [0.2,1])for(const kind of ['main','downstream'])for(const seed of [0,1]){
        const evals=path.join(bench,`ours_eval_results_simclr2k_barlow10k_2ssl3probes_${kind}/${source}/ResNet18BN/pt${size}/simclr/ssl_seed_${String(seed).padStart(3,'0')}/evaluations`);
        if(!fs.existsSync(evals))continue;
        for(const f of fs.readdirSync(evals).filter(f=>/^(CIFAR10|CIFAR100|TinyImageNet)_.*\.json$/.test(f))){
            const r=read(path.join(evals,f));if(r.complete&&r.seconds>0)(probeTimes[r.expected.target+'|'+r.expected.label_percentage]??=[]).push(r.seconds);
        }
    }
    let reused=0,newProbes=0;
    for(const source of ['CIFAR10','CIFAR100'])for(const model of ['ResNet34','ResNet50'])for(const kind of ['full','random','ours'])for(const size of (kind==='full'?[100]:[0.2,1])){
        const id=`${source}_${model}_${kind}_p${String(size).replace('.','p')}`,targets=[source,source==='CIFAR10'?'CIFAR100':'CIFAR10','TinyImageNet'];
        groups.push({source,model,kind,size});
        const ref=calibration.models.ResNet18,measured=calibration.models[model];
        const sslLog=kind==='full'?path.join(bench,`full_ssl_checkpoints_1000ep_3seeds/${source}/ResNet18/full/pt100/simclr/seed_000/ssl.jsonl`):
            path.join(bench,`ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/${source}/ResNet18BN/pt${size}/simclr/ssl_seed_000/ssl.jsonl`);
        const oldEpoch=p95(logs(sslLog).slice(-200).map(r=>r.seconds));
        const steps=kind==='full'?195:1,batchSize=size===0.2?'100':'256';
        const epochSeconds=oldEpoch+steps*Math.max(0,measured.ssl_batch_seconds[batchSize]-ref.ssl_batch_seconds[batchSize]);
        timings.push({source,model,kind,size,reference_epoch_seconds:oldEpoch,estimated_epoch_seconds:epochSeconds});
        for(const seed of (kind==='full'?[0]:[0,1])){
            const folder=path.join(here,'results',source,model,kind,`pt${size}`,'simclr',`ssl_seed_${String(seed).padStart(3,'0')}`);
            const config=fs.existsSync(path.join(folder,'config.json'))?read(path.join(folder,'config.json')):null;
            if(config){assert.deepEqual(config.signature.ablation_code_hashes,code_hashes);assert.deepEqual([config.signature.source,config.signature.model,config.signature.selection_method,config.signature.subset,config.identity.seed],[source,model,kind,size,seed]);}
            const markerPath=path.join(folder,'ssl_complete.json'),encoder=path.join(folder,'encoder.pt');let ready=false;
            if(fs.existsSync(markerPath)){
                const marker=read(markerPath);assert.ok(marker.complete&&marker.epochs===(kind==='full'?1000:2000));
                assert.deepEqual(marker.identity,config.identity);assert.equal(hash(encoder),marker.checkpoint_file_sha256);ready=true;
            }
            const rolling=path.join(folder,'training.pt');const epoch=!ready&&fs.existsSync(rolling)?epochHeader(rolling,config.identity):0;
            assert.ok(epoch>=0&&epoch<=(kind==='full'?1000:2000));
            let parent=null;
            if(!ready)for(const end of (kind==='full'?Array.from({length:10},(_,i)=>(i+1)*100):[2000])){
                if(epoch>=end&&end<1000)continue;
                const name=`xa_${id}_s${seed}_e${end}`;allNames.add(name);
                if(!queue.has(name)){
                    const start=Math.max(epoch,kind==='full'?end-100:0),seconds=Math.ceil(Math.max(0,end-start)*epochSeconds+60);
                    jobs.push({name,stage:'ssl',source,model,kind,size,seed,end_epoch:end,seconds,dependencies:parent?[parent]:[],code_hashes,buffer_seconds:60});
                }
                parent=name;
            }
            const probes=kind==='full'?5:2;let seconds=0,missing=0;
            for(const target of targets)for(const labels of [1,5])for(let probe=0;probe<probes;probe++){
                const file=path.join(folder,'evaluations',`${target}_lbl${labels}_probe${probe}.json`);
                if(fs.existsSync(file)){
                    const r=read(file);assert.ok(ready&&r.complete&&r.encoder_unchanged);assert.deepEqual(r.expected.identity,config.identity);
                    assert.equal(r.expected.evaluation.epochs,100);assert.deepEqual(r.expected.evaluation_code_hashes,sharedHashes);
                    reused++;continue;
                }
                const count=(target==='TinyImageNet'?100000:50000)*labels/100;
                const batches=100*Math.ceil(count/256)+Math.ceil(10000/256);
                seconds+=p95(probeTimes[target+'|'+labels]||[])+batches*Math.max(0,measured.probe_batch_seconds-ref.probe_batch_seconds);
                missing++;
            }
            if(missing){
                const name=`xa_${id}_s${seed}_probe`;allNames.add(name);
                if(!queue.has(name))jobs.push({name,stage:'probe',source,model,kind,size,seed,seconds:Math.ceil(seconds+60),dependencies:parent?[parent]:[],code_hashes,missing_probes:missing,buffer_seconds:60});
                newProbes+=missing;
            }
        }
    }
    assert.equal(groups.length,20);assert.equal(groups.filter(g=>g.kind==='full').length,4);
    const encoders=groups.reduce((n,g)=>n+(g.kind==='full'?1:2),0);assert.equal(encoders,36);
    const totalProbes=groups.reduce((n,g)=>n+(g.kind==='full'?30:24),0);assert.equal(totalProbes,504);
    const synthetic=Object.fromEntries(Object.entries(calibration.synthetics).map(([f,sha])=>{assert.equal(hash(path.join(root,f)),sha);return[f,sha];}));
    let tsv='';const seen=new Set(queue.keys());
    for(const job of jobs){
        for(const dep of job.dependencies)assert.ok(seen.has(dep),'Unresolved dependency: '+dep);seen.add(job.name);
        if(job.kind==='ours'){
            job.synthetic_file=`OT-SSL-DD/ours_syntehtic/${job.source}/ResNet18BN/pt${job.size}/res_IIC_${job.source}_ResNet18BN_${job.size}percent_final.pt`;
            job.synthetic_sha256=synthetic[job.synthetic_file];assert.ok(job.synthetic_sha256);
        }
        const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        tsv+=[job.name,job.seconds,job.dependencies.join(',')||'-',file].join('\t')+'\n';
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const coverage={ssl_encoders:encoders,total_probes:totalProbes,table_cells:120,jobs_to_submit:jobs.length,
        ssl_jobs:jobs.filter(j=>j.stage==='ssl').length,probe_jobs:jobs.filter(j=>j.stage==='probe').length,
        full_training_segments:jobs.filter(j=>j.stage==='ssl'&&j.kind==='full').length,new_probes:newProbes,reused_probes:reused,
        account:'aip-boyuwang',full_epochs:1000,subset_epochs:2000,probe_epochs:100,buffer_seconds:60,timings};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n');
    console.log(JSON.stringify({...coverage,timings:undefined},null,2));
}
if(require.main===module)main();
module.exports={epochHeader,main};
