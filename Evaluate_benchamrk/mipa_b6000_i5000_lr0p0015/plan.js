// Five new synthetic sets, five saved encoders and six probes per label budget.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench),ot=path.join(root,'OT-SSL-DD');
const syntheticRoot=path.join(ot,'ours_syntehtic_b6000_i5000_lr0p0015');
const settings=[['CIFAR10',2],['CIFAR100',2],['TinyImageNet',.2],['TinyImageNet',1],['TinyImageNet',2]];
const accounts=['aip-boyuwang','aip-yiweilu'];
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(n=>Number.isFinite(n)&&n>0),'Missing runtime measurements');return a.sort((a,b)=>a-b)[Math.ceil(.95*a.length)-1];};
function stages(source,subset){
    const base=`mipa6k_${source}_p${String(subset).replace('.','p')}`;
    return [1000,2000,3000,4000,5000].map(stop_iteration=>({name:base+'_g'+stop_iteration,stage:'generation',stop_iteration}))
        .concat([{name:base+'_ssl2k',stage:'ssl'},{name:base+'_lbl1',stage:'probe',labels:1},{name:base+'_lbl5',stage:'probe',labels:5}])
        .map((s,index)=>({...s,source,subset,account:accounts[index%2],parents:index===0?[]:[base+(index<5?'_g'+index*1000:index===5?'_g5000':'_ssl2k')]}));
}
function check(){
    const jobs=settings.flatMap(([s,p])=>stages(s,p));assert.equal(jobs.length,40);assert.equal(new Set(jobs.map(j=>j.name)).size,40);
    for(const a of accounts)assert.equal(jobs.filter(j=>j.account===a).length,20);
    assert.equal(jobs.filter(j=>j.stage==='generation').length,25);assert.equal(jobs.filter(j=>j.stage==='ssl').length,5);assert.equal(jobs.filter(j=>j.stage==='probe').length,10);
    const seen=new Set();for(const j of jobs){for(const parent of j.parents)assert.ok(seen.has(parent));seen.add(j.name);}
    assert.ok(syntheticRoot!==path.join(ot,'ours_syntehtic'));
    console.log('PASS: 5 separate synthetic sets; 25 continuing generation segments; 5 SSL jobs; 10 six-probe jobs; 20 Wang / 20 Lu.');
}
function generationTiming(source,subset){
    const file=path.join(ot,`ours_syntehtic/logs/${source}_ResNet18BN_pt${subset}.log`),lines=fs.readFileSync(file,'utf8').split('\n');
    let previous;const rates=[];
    for(const line of lines){const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
        const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,iteration:Number(m[3])};
        if(previous&&current.iteration-previous.iteration===25&&current.time>previous.time&&current.time-previous.time<300)rates.push((current.time-previous.time)/25);
        previous=current;
    }
    const count=(source==='TinyImageNet'?100000:50000)*subset/100;
    const measured=p95(rates),scaled=measured*(6000+count)/(1024+count);
    // Measured initialization gaps in these logs, after accounting for their UTC/local-hour offset.
    const initialization={'CIFAR10_2':21,'CIFAR100_2':17,'TinyImageNet_0.2':47,'TinyImageNet_1':82,'TinyImageNet_2':126}[source+'_'+subset];
    return {old_log:path.relative(root,file),samples:rates.length,old_batch:1024,new_batch:6000,
        old_iteration_seconds_p95:measured,estimated_iteration_seconds:scaled,initialization_seconds:initialization,
        estimate:'Scale frozen-feature passes by (6000 + synthetic count)/(1024 + synthetic count); approximate.'};
}
function main(batch){
    check();batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const shared=fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f).concat(['networks.py','benchmarks/downstream_data.py']);
    const files=shared.concat(['Evaluate_benchamrk/protocol.json','OT-SSL-DD/method1.py','OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py','OT-SSL-DD/utils.py','OT-SSL-DD/networks.py',
        'Evaluate_benchamrk/mipa_b6000_i5000_lr0p0015/worker.py','Evaluate_benchamrk/mipa_b6000_i5000_lr0p0015/plan.js','Evaluate_benchamrk/submit_mipa_b6000_i5000_lr0p0015_vulcan.sh']);
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const all=[],timings=[],done=new Set();
    const startup=read(path.join(bench,'cifar_small_simclr2k_1ssl5probes/batches/bundled_20261027T181030Z_YGcQKH/bundled_coverage.json')).startup_measurement.seconds;
    for(const [source,subset]of settings){
        const teacherSeed=source==='TinyImageNet'?1:0;
        const teacherDir=path.join(bench,'full_ssl_checkpoints_1000ep_3seeds',source,'ResNet18/full/pt100/simclr',`seed_${String(teacherSeed).padStart(3,'0')}`);
        assert.ok(fs.statSync(path.join(teacherDir,'encoder.pt')).size>0);assert.ok(read(path.join(teacherDir,'ssl_complete.json')).complete);
        const teacher={seed:teacherSeed,checkpoint:path.relative(root,path.join(teacherDir,'encoder.pt')),marker_sha256:hash(path.join(teacherDir,'ssl_complete.json'))};
        const timing=generationTiming(source,subset),genFolder=path.join(syntheticRoot,source,'ResNet18BN','pt'+subset);
        const statePath=path.join(genFolder,'generation_state.json'),state=fs.existsSync(statePath)?read(statePath):null;
        if(state)assert.deepEqual(state.code_hashes,code_hashes,'Generation code changed');
        const sslFolder=path.join(here,'results',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
        const markerPath=path.join(sslFolder,'ssl_complete.json'),marker=fs.existsSync(markerPath)?read(markerPath):null;
        if(marker){assert.ok(marker.complete&&marker.epochs===2000);assert.deepEqual(read(path.join(sslFolder,'config.json')).signature.driver_code_hashes,code_hashes);}
        const measured=path.join(bench,'subset_simclr2k_1ssl5probes/results',source,'ResNet18/random','pt'+subset,'simclr/ssl_seed_000');
        const sslRate=p95(fs.readFileSync(path.join(measured,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds));
        const probeRates={};for(const labels of [1,5]){
            const rates=[];for(let seed=0;seed<5;seed++){
                const file=path.join(measured,'evaluations',`${source}_lbl${labels}_probe${seed}.json`);
                if(fs.existsSync(file)){const r=read(file);if(r.complete&&r.expected.evaluation.epochs===100&&r.seconds>0)rates.push(r.seconds);}
            }
            if(!rates.length)for(let seed=0;seed<6;seed++){
                const file=path.join(bench,`kmeans_eval_results_interactive_6seeds/${source}/ResNet18/kmeans/pt${subset}/simclr/seed_${String(seed).padStart(3,'0')}/evaluations/${source}_lbl${labels}.json`);
                if(fs.existsSync(file)){const r=read(file);if(r.complete&&r.expected.evaluation.epochs===100&&r.seconds>0)rates.push(r.seconds);}
            }
            probeRates[labels]=p95(rates);
        }
        for(const stage of stages(source,subset)){
            let finished=false,seconds,missing;
            if(stage.stage==='generation'){
                finished=state&&state.iteration>=stage.stop_iteration;
                const start=stage.stop_iteration-1000,remaining=1000-Math.max(0,(state?.iteration||0)-start);
                seconds=Math.ceil(Math.max(0,remaining)*timing.estimated_iteration_seconds+(stage.stop_iteration===1000?timing.initialization_seconds:startup)+60);
            }else if(stage.stage==='ssl'){
                finished=Boolean(marker);seconds=Math.ceil(2000*sslRate+startup+60);
            }else{
                const summaryPath=path.join(sslFolder,`summary_lbl${stage.labels}.json`);
                if(fs.existsSync(summaryPath)){const s=read(summaryPath);finished=s.complete&&s.completed_probes===6&&s.ssl_epochs===2000;}
                missing=0;for(let seed=0;seed<6;seed++){
                    const file=path.join(sslFolder,'evaluations',`${source}_lbl${stage.labels}_probe${seed}.json`);
                    if(!fs.existsSync(file))missing++;else{const r=read(file);assert.ok(marker&&r.complete&&r.encoder_unchanged&&r.encoder_sha256===marker.encoder_sha256);}
                }
                seconds=Math.ceil(missing*probeRates[stage.labels]+startup+60);
            }
            if(finished){done.add(stage.name);continue;}
            all.push({...stage,seconds,buffer_seconds:60,missing_probes:missing,teacher,code_hashes});
        }
        timings.push({source,subset,generation:timing,ssl_epoch_seconds_p95:sslRate,probe_seconds_p95:probeRates});
    }
    const known=new Set([...done,...queue.keys()]);let tsv='';const submitted=[];
    for(const job of all){
        if(queue.has(job.name)){known.add(job.name);continue;}
        job.parents=job.parents.filter(parent=>!done.has(parent));for(const parent of job.parents)assert.ok(known.has(parent),'Missing stage dependency: '+parent);
        assert.ok(job.seconds>0&&job.seconds<3*3600,'Unexpected long stage; inspect its runtime estimate');
        const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        tsv+=[job.name,job.stage,job.account,job.seconds,job.parents.join(',')||'-',file].join('\t')+'\n';submitted.push(job);known.add(job.name);
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const coverage={synthetic_sets:5,generation_iterations:5000,generation_batch:6000,generation_lr:0.0015,model:'ResNet18BN',
        generation_jobs:submitted.filter(j=>j.stage==='generation').length,ssl_jobs:submitted.filter(j=>j.stage==='ssl').length,
        probe_jobs:submitted.filter(j=>j.stage==='probe').length,ssl_epochs:2000,ssl_seeds:[0],probe_epochs:100,probe_seeds:[0,1,2,3,4,5],labels:[1,5],
        expected_total_probes:60,jobs:submitted.length,accounts:Object.fromEntries(accounts.map(a=>[a,submitted.filter(j=>j.account===a).length])),
        minimum_seconds:submitted.length?Math.min(...submitted.map(j=>j.seconds)):0,maximum_seconds:submitted.length?Math.max(...submitted.map(j=>j.seconds)):0,
        synthetic_root:syntheticRoot,result_root:path.join(here,'results'),table_updates:false,timings};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n');
    console.log(JSON.stringify({...coverage,timings:undefined},null,2));
}
function stoppedStages(source,subset){
    const base=`mipa6ks_${source}_p${String(subset).replace('.','p')}`;
    return [{name:base+'_ssl2k',stage:'ssl',parents:[]},
        ...[1,5].map(labels=>({name:base+'_lbl'+labels,stage:'probe',labels,parents:[base+'_ssl2k']}))];
}
function stopped(batch){
    check();batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const original=path.join(here,'batches/batch_20261028T003926Z_OjF5mJ'),old=read(path.join(original,'coverage.json'));
    const template=read(path.join(original,'mipa6k_CIFAR10_p2_ssl2k.json'));
    const code_hashes=Object.fromEntries(Object.keys(template.code_hashes).map(f=>[f,hash(path.join(root,f))]));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    assert.ok(![...queue.keys()].some(name=>name.startsWith('mipa6k_')),'Stop the original larger-batch generation/evaluation chain first');
    const startup=read(path.join(bench,'cifar_small_simclr2k_1ssl5probes/batches/bundled_20261027T181030Z_YGcQKH/bundled_coverage.json')).startup_measurement.seconds;
    const planned=[],accepted_sets=[];let sequence=0,finished_probes=0;
    for(const [source,subset]of settings){
        if(source==='CIFAR100'){
            const dir=path.join(here,'results/CIFAR100/ResNet18BN/pt2/simclr/ssl_seed_000');
            for(const labels of [1,5]){const s=read(path.join(dir,`summary_lbl${labels}.json`));assert.ok(s.complete&&s.completed_probes===6&&s.ssl_epochs===2000);}
            continue;
        }
        const gen=path.join(syntheticRoot,source,'ResNet18BN','pt'+subset),state=read(path.join(gen,'generation_state.json'));
        assert.ok(!state.complete&&state.iteration>0&&state.iteration<5000);
        const input=path.join(gen,`res_IIC_${source}_ResNet18BN_${subset}percent_final.pt`);
        const accepted_generation={iteration:state.iteration,file_sha256:hash(input),generation_code_hashes:state.code_hashes};
        accepted_sets.push({source,subset,...accepted_generation});
        const originalJob=read(path.join(original,`mipa6k_${source}_p${String(subset).replace('.','p')}_ssl2k.json`));
        const timing=old.timings.find(t=>t.source===source&&t.subset===subset),dir=path.join(here,'stopped_results',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
        const markerPath=path.join(dir,'ssl_complete.json');let sslDone=false;
        if(fs.existsSync(markerPath)){
            const m=read(markerPath),c=read(path.join(dir,'config.json'));
            assert.ok(m.complete&&m.epochs===2000);assert.deepEqual(c.signature.accepted_generation,accepted_generation);assert.deepEqual(c.signature.driver_code_hashes,code_hashes);sslDone=true;
        }
        for(const stage of stoppedStages(source,subset)){
            const account=accounts[sequence++%2];let seconds,missing;
            if(stage.stage==='ssl'){
                if(sslDone||queue.has(stage.name))continue;
                seconds=Math.ceil(startup+60+2000*timing.ssl_epoch_seconds_p95);
            }else{
                const summaryPath=path.join(dir,`summary_lbl${stage.labels}.json`);
                if(fs.existsSync(summaryPath)){const s=read(summaryPath);assert.deepEqual(s.accepted_generation,accepted_generation);if(s.complete&&s.completed_probes===6){finished_probes+=6;continue;}}
                if(queue.has(stage.name))continue;
                missing=0;for(let seed=0;seed<6;seed++){
                    const f=path.join(dir,'evaluations',`${source}_lbl${stage.labels}_probe${seed}.json`);
                    if(!fs.existsSync(f))missing++;else{const r=read(f);assert.ok(sslDone&&r.complete&&r.encoder_unchanged);finished_probes++;}
                }
                seconds=Math.ceil(startup+60+missing*timing.probe_seconds_p95[stage.labels]);
            }
            planned.push({...originalJob,...stage,source,subset,account,seconds,code_hashes,accepted_generation,
                parents:stage.stage==='probe'&&sslDone?[]:stage.parents,missing_probes:missing});
        }
    }
    assert.equal(sequence,12);assert.equal(accepted_sets.length,4);
    const known=new Set(queue.keys());let tsv='';
    for(const job of planned){
        assert.ok(job.stage==='ssl'||job.stage==='probe');assert.ok(job.seconds>0&&job.seconds<3*3600);
        for(const parent of job.parents)assert.ok(known.has(parent),'Missing SSL dependency');known.add(job.name);
        const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        tsv+=[job.name,job.stage,job.account,job.seconds,job.parents.join(',')||'-',file].join('\t')+'\n';
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const coverage={mode:'evaluate_explicitly_stopped_generations',accepted_sets,completed_CIFAR100_preserved:true,
        jobs:planned.length,ssl_jobs:planned.filter(j=>j.stage==='ssl').length,probe_jobs:planned.filter(j=>j.stage==='probe').length,
        probes_to_run:planned.reduce((n,j)=>n+(j.missing_probes||0),0),completed_probes_skipped:finished_probes,
        ssl_epochs:2000,ssl_seeds:[0],linear_epochs:100,probe_seeds:[0,1,2,3,4,5],labels:[1,5],buffer_seconds:60,
        accounts:Object.fromEntries(accounts.map(a=>[a,planned.filter(j=>j.account===a).length])),table_updates:false};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n');
    console.log(JSON.stringify({...coverage,accepted_sets:accepted_sets.map(({source,subset,iteration})=>({source,subset,iteration}))},null,2));
}
if(require.main===module){
    if(process.argv[2]==='--check'){
        check();const jobs=settings.filter(([s])=>s!=='CIFAR100').flatMap(([s,p])=>stoppedStages(s,p));
        assert.equal(jobs.length,12);assert.equal(jobs.filter(j=>j.stage==='ssl').length,4);assert.equal(jobs.filter(j=>j.stage==='probe').length,8);
        console.log('PASS: early-stop evaluation is four SSL encoders + eight six-probe jobs; no generation.');
    }else if(process.argv[3]==='--eval-stopped')stopped(process.argv[2]);
    else main(process.argv[2]);
}
