// Read JSON, logs and the scalar epoch header only; never import Python on login.
const fs=require('fs'),path=require('path'),assert=require('assert/strict'),crypto=require('crypto'),cp=require('child_process');
const {rows,key}=require('./tables'),{pairs}=require('./plan');
const here=__dirname,bench=path.dirname(here),repo=path.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8')),exists=f=>fs.existsSync(f)&&fs.statSync(f).size>0;
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const definitions=()=>[...read(path.join(here,'cells.json')),...read(path.join(here,'remaining_cells.json'))];
const excluded=c=>c.subset===2&&c.ssl==='barlowtwins';
const seedTag=g=>['four_ssl','ten_ssl'].includes(g.scheme)?'seed_':'ssl_seed_';
function epochHeader(file,identity){
    const bytes=cp.execFileSync('unzip',['-p',file,'*/data.pkl'],{maxBuffer:4*1024*1024});
    assert.ok(bytes.includes(Buffer.from(identity.signature)),'Rolling checkpoint signature changed');
    const at=bytes.indexOf(Buffer.from('epoch'));assert.ok(at>0&&at<1024,'Unexpected rolling-state header');
    let i=at+5;if(bytes[i]===0x71)i+=2;else if(bytes[i]===0x72)i+=5;else if(bytes[i]===0x94)i++;
    if(bytes[i]===0x4b)return bytes[i+1];if(bytes[i]===0x4d)return bytes.readUInt16LE(i+1);
    if(bytes[i]===0x4a)return bytes.readInt32LE(i+1);throw Error('Unsupported scalar epoch encoding');
}
function logRows(file){
    if(!exists(file))return [];const lines=fs.readFileSync(file,'utf8').split('\n');
    return lines.flatMap((s,i)=>{if(!s.trim())return [];try{return [JSON.parse(s)];}catch(e){if(i===lines.length-1)return [];throw e;}});
}
const p95=a=>{assert.ok(a.length,'No measured runtime');return [...a].sort((x,y)=>x-y)[Math.ceil(a.length*.95)-1];};
function balance(jobs,account){
    const counts={'aip-boyuwang':0,'aip-yiweilu':0},seconds={'aip-boyuwang':0,'aip-yiweilu':0};
    if(account){assert.ok(account in counts);for(const j of jobs){j.account=account;counts[account]++;seconds[account]+=j.seconds;}return {counts,seconds};}
    const limits={'aip-boyuwang':Math.ceil(jobs.length/2),'aip-yiweilu':Math.floor(jobs.length/2)};
    for(const j of [...jobs].sort((a,b)=>b.seconds-a.seconds||a.name.localeCompare(b.name))){
        const a=Object.keys(counts).filter(a=>counts[a]<limits[a]).sort((a,b)=>seconds[a]-seconds[b]||counts[a]-counts[b])[0];
        j.account=a;counts[a]++;seconds[a]+=j.seconds;
    }
    assert.ok(Math.abs(counts['aip-boyuwang']-counts['aip-yiweilu'])<=1);return {counts,seconds};
}
function main(){
    const batch=path.resolve(process.argv[2]),synthetic=path.resolve(process.argv[3]);
    assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const queue=read(process.argv[4]),reference=read(path.join(bench,'protocol.json'));
    const hashes=Object.fromEntries([...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py'].map(f=>[f,hash(path.join(repo,f))]));
    const blanks=['table.tex','cifar10_downstream.tex','cifar100_downstream.tex','tinyimagenet_simclr.tex','tinyimagenet_barlowtwins.tex'].flatMap(f=>
        rows(f,fs.readFileSync(path.join(repo,'CVPR_tables',f),'utf8')).flatMap(r=>r.cells))
        .filter(c=>c.model==='ResNet18'&&c.value==='');
    const groups=definitions(),lookup=new Map();
    if(process.env.TABLE_GROUP)assert.ok(groups.some(g=>g.id===process.env.TABLE_GROUP),'Unknown table group');
    for(const g of groups)for(const [target,labels]of g.cells){const k=key({...g,target,labels});assert.ok(!lookup.has(k));lookup.set(k,g);}
    const selected=blanks.filter(c=>!excluded(c)&&(!process.env.TABLE_GROUP||lookup.get(key(c))?.id===process.env.TABLE_GROUP));
    for(const c of selected)assert.ok(lookup.has(key(c)),'Unmapped blank; do not silently skip: '+key(c));
    const probeTimes=new Map(),sslTimes=new Map(),jobs=[],blocked=[],encoders=new Map(),report=[];
    for(const g of groups)for(const seed of new Set(pairs(g.scheme).map(p=>p[0]))){
        const tag=seedTag(g),dir=path.join(bench,g.checkpoint_group,tag+String(seed).padStart(3,'0'));
        const values=logRows(path.join(dir,'ssl.jsonl')).slice(-200).filter(r=>r.seconds>0).map(r=>r.seconds);
        const k=[g.source,g.subset,g.ssl].join('|');sslTimes.set(k,[...(sslTimes.get(k)||[]),...values]);
        const evaluation=path.join(bench,g.output_group,tag+String(seed).padStart(3,'0'),'evaluations');
        if(fs.existsSync(evaluation))for(const f of fs.readdirSync(evaluation).filter(f=>f.endsWith('.json'))){
            const r=read(path.join(evaluation,f));if(!r.complete||!(r.seconds>0))continue;
            const k=[g.source==='TinyImageNet'?64:32,r.expected.target,r.expected.label_percentage].join('|');
            probeTimes.set(k,[...(probeTimes.get(k)||[]),r.seconds]);
        }
    }
    // Random timings supply the same-size Barlow recipe when an Ours run is new.
    for(const source of ['CIFAR10','CIFAR100','TinyImageNet'])for(const p of[0.2,1,2])for(const ssl of['simclr','barlowtwins']){
        const dir=path.join(bench,'random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k',`${source}/ResNet18/random/pt${p}/${ssl}/seed_000`);
        const k=[source,p,ssl].join('|'),rates=logRows(path.join(dir,'ssl.jsonl')).slice(-200).filter(r=>r.seconds>0).map(r=>r.seconds);
        sslTimes.set(k,[...(sslTimes.get(k)||[]),...rates]);
    }
    let reused=0,missing=0;
    for(const cell of selected){
        const g=lookup.get(key(cell)),tag=seedTag(g);
        const name=`pp_${g.id}_${cell.target}_l${cell.labels}`;
        if(queue[name]){report.push({cell:key(cell),status:'queued',job:queue[name]});continue;}
        const sourceFile=g.kind==='ours'?path.join(repo,'OT-SSL-DD/ours_syntehtic',g.source,'ResNet18BN',`pt${g.subset}`,
            `res_IIC_${g.source}_ResNet18BN_${g.subset}percent_final.pt`):
            path.join(synthetic,'ConvNet',g.source,`${g.source==='TinyImageNet'?'Tiny':g.source}_${g.subset}pct/images_${g.source==='TinyImageNet'?10000:5000}.pt`);
        if(g.hold){blocked.push({cell:key(cell),reason:'user_hold_until_synced_data_confirmed',path:sourceFile});continue;}
        const generation=g.generation_job_name||(g.kind==='mkdt'&&g.subset===0.2?`mk02_${g.source}_dist_${g.source==='TinyImageNet'?10000:5000}`:null);
        const seeds=[...new Set(pairs(g.scheme).map(p=>p[0]))],needsWeights=seeds.some(seed=>!exists(path.join(bench,g.checkpoint_group,tag+String(seed).padStart(3,'0'),'encoder.pt')));
        if(needsWeights&&!exists(sourceFile)&&!queue[generation]){blocked.push({cell:key(cell),reason:'missing_synthetic',path:sourceFile});continue;}
        const dependencies=[],samples=[],directory=`partial_probe_results/${g.id}/${cell.target}_lbl${cell.labels}`;
        let count=0;
        for(const [ssl_seed,probe_seed]of pairs(g.scheme)){
            const folder=path.join(bench,g.checkpoint_group,tag+String(ssl_seed).padStart(3,'0')),id=g.checkpoint_group+'|'+ssl_seed;
            if(!encoders.has(id)){
                const config=exists(path.join(folder,'config.json'))?read(path.join(folder,'config.json')):null;
                const marker=exists(path.join(folder,'ssl_complete.json'))?read(path.join(folder,'ssl_complete.json')):null;
                if(config){const s=config.signature;assert.deepEqual([s.source,s.model,s.selection_method,s.subset,s.ssl,config.identity.seed,s.protocol.ssl.subset_epochs],
                    [g.source,'ResNet18',g.kind,g.subset,g.ssl,ssl_seed,g.epochs]);assert.deepEqual(s.code_hashes,hashes);assert.deepEqual(s.protocol.evaluation,reference.evaluation);}
                if(marker){assert.ok(marker.complete&&marker.epochs===g.epochs);assert.deepEqual(marker.identity,config.identity);}
                const ready=exists(path.join(folder,'encoder.pt'))&&config&&marker,trainName=`rtssl_${g.training_group_id||g.id}_s${ssl_seed}`;
                if(!ready){
                    assert.ok(g.kind!=='random'&&(g.source==='TinyImageNet'||g.kind==='mkdt'&&g.subset===0.2),'No unapproved SSL retraining');
                    const epoch=exists(path.join(folder,'training.pt'))?epochHeader(path.join(folder,'training.pt'),config.identity):0;
                    assert.ok(epoch>=0&&epoch<=g.epochs);
                    const seconds=Math.ceil((g.epochs-epoch)*p95(sslTimes.get([g.source,g.subset,g.ssl].join('|'))||[])+60);
                    if(!queue[trainName])jobs.push({name:trainName,stage:'ssl',group_id:g.id,ssl_seed,remaining:true,
                        source_file:sourceFile,source_file_sha256:!queue[generation]&&exists(sourceFile)?hash(sourceFile):null,resume_epoch_estimate:epoch,seconds,buffer_seconds:60,
                        dependencies:queue[generation]?[generation]:[],code_hashes:hashes});
                }
                encoders.set(id,{folder,config,marker,dependency:ready?null:trainName});
            }
            const {config,marker,dependency}=encoders.get(id);if(dependency)dependencies.push(dependency);
            const output=path.join(bench,g.output_group,tag+String(ssl_seed).padStart(3,'0'));
            const suffix=['four_ssl','ten_ssl'].includes(g.scheme)&&probe_seed===ssl_seed?'':`_probe${probe_seed}`;
            const canonical=path.join(output,'evaluations',`${cell.target}_lbl${cell.labels}${suffix}.json`);
            // These legacy file-metadata fingerprints differ on Vulcan; retain originals and re-probe all six.
            const reprobe=g.scheme==='four_ssl'&&['Dogs','Flowers'].includes(cell.target);
            const destination=!reprobe&&(exists(canonical)||['two_ssl','ten_ssl'].includes(g.scheme))?canonical:
                path.join(bench,directory,`ssl_seed_${ssl_seed}_probe_${probe_seed}.json`);
            if(exists(destination)){
                const r=read(destination),e=r.expected;assert.ok(r.complete&&r.encoder_unchanged&&r.encoder_sha256===marker.encoder_sha256);
                assert.deepEqual(e.identity,config.identity);assert.deepEqual(e.evaluation,reference.evaluation);assert.deepEqual(e.evaluation_code_hashes,hashes);
                assert.deepEqual([e.source,e.target,e.seed,e.label_percentage],[g.source,cell.target,probe_seed,cell.labels]);reused++;
            }else count++;
            samples.push({ssl_seed,probe_seed,checkpoint_dir:path.relative(bench,folder),config_sha256:config?hash(path.join(folder,'config.json')):null,
                encoder_sha256:marker?.encoder_sha256||null,result_file:path.relative(bench,destination)});
        }
        const rates=probeTimes.get([cell.source==='TinyImageNet'?64:32,cell.target,cell.labels].join('|'))||[];
        const seconds=Math.ceil(count*p95(rates)+60);
        jobs.push({name,stage:'probe',group_id:g.id,cell,cell_key:key(cell),remaining:true,ssl_epochs:g.epochs,scheme:g.scheme,
            repeated_encoders:['one_ssl','four_ssl'].includes(g.scheme),samples,summary_file:directory+'/summary.json',code_hashes:hashes,
            seconds,buffer_seconds:60,dependencies:[...new Set(dependencies)],missing_probes:count,reprobe_legacy_target:g.scheme==='four_ssl'&&['Dogs','Flowers'].includes(cell.target)});
        missing+=count;report.push({cell:key(cell),status:'planned',job:name,missing_probes:count});
    }
    const accounts=balance(jobs,process.env.TABLE_ACCOUNT);let tsv='';
    for(const j of jobs){const f=path.join(batch,j.name+'.json');fs.writeFileSync(f,JSON.stringify(j,null,2)+'\n',{flag:'wx'});
        tsv+=[j.name,j.stage,j.account,j.seconds,j.dependencies.join(',')||'-',f].join('\t')+'\n';}
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const summary={blank_cells:blanks.length,excluded_cells:blanks.filter(excluded).length,eligible_cells:selected.length,
        planned_cells:report.filter(r=>r.status==='planned').length,blocked_cells:blocked.length,jobs:jobs.length,
        ssl_jobs:jobs.filter(j=>j.stage==='ssl').length,probe_jobs:jobs.filter(j=>j.stage==='probe').length,new_probes:missing,
        reused_probes:reused,accounts,buffer_seconds:60,blocked,report};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(summary,null,2)+'\n');
    console.log('PLAN:',JSON.stringify({...summary,blocked:undefined,report:undefined}));
    for(const b of blocked)console.log('BLOCKED:',b.cell,b.path);
}
module.exports={epochHeader,balance,excluded,definitions};
if(require.main===module)main();
