// Login-safe planning: existing JSON, log timings, and checkpoint scalar headers only.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const {epochHeader}=require('../cross_arch_ablation/plan');
const HERE=__dirname,BENCH=path.dirname(HERE),ROOT=path.dirname(BENCH);
const SOURCES=['CIFAR10','CIFAR100','SVHN','FashionMNIST'],SIZES=[.2,1,2];
const METRICS=['l1','l2','mmd','mmd-dist','dm','energy','whitened-cosine','sinkhorn'];
const COUNTS={CIFAR10:50000,CIFAR100:50000,SVHN:73257,FashionMNIST:60000};
const PARAMS={l1:{},l2:{},mmd:{mmd_sigma:1},'mmd-dist':{mmd_sigma:1,mmd_views:2},dm:{dm_views:2},
    energy:{energy_views:2},'whitened-cosine':{whitening_reg:.001},sinkhorn:{sinkhorn_views:2,sinkhorn_epsilon:.1,sinkhorn_iterations:50}};
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(n=>Number.isFinite(n)&&n>0),'Missing measured timing');return a.sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];};
const records=f=>fs.readFileSync(f,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);

function generationArgs(source,subset){
    const folder=source==='SVHN'?'digits_simclr/checkpoints':source==='FashionMNIST'?'fashionmnist/full/checkpoints':'full_ssl_checkpoints_1000ep_3seeds';
    return {dataset:source,percentage:subset,model:'ResNet18BN',seed:0,Iteration:10000,lr_img:.002,batch_real:128,
        temperature:.2,disp_weight:0,pair_weight:0,num_random_networks:1,random_models:'ResNet18BN',dsa:true,
        distill_aug_strategy:'color_crop_cutout_flip_scale_rotate',distill_aug_mode:'M',teacher_source:'pretrained',
        teacher_seed:[0],teacher_ssl:'simclr',teacher_path:path.join(BENCH,folder),feature_batch:128};
}

function generationRate(source,subset,count){
    if(subset===.2&&(source==='SVHN'||source==='FashionMNIST')){
        const calibration=read(path.join(BENCH,'mipa_random128/timing.json'));
        return {seconds:calibration.units.CIFAR10_p0p2.generation_seconds_p95*count/100,
                origin:'measured batch-128 CIFAR10 timing, scaled by synthetic count; digit 0.2% logs used batch 1024'};
    }
    const digits=source==='SVHN'||source==='FashionMNIST';
    const dir=path.join(BENCH,digits?'digits_mipa/logs':'mipa_random128/logs');
    const prefix=(digits?'digm_':'r128_')+source+'_p'+String(subset).replace('.','p')+'_gen-';
    const rates=[],files=fs.readdirSync(dir).filter(f=>f.startsWith(prefix)&&f.endsWith('.out'));
    for(const file of files){let previous;
        for(const line of fs.readFileSync(path.join(dir,file),'utf8').split('\n')){
            const m=line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);if(!m)continue;
            const current={time:Date.parse(m[1]+'T'+m[2]+'Z')/1000,i:+m[3]};
            if(previous&&current.i>previous.i&&current.time>previous.time&&current.time-previous.time<600)
                rates.push((current.time-previous.time)/(current.i-previous.i));
            previous=current;
        }
    }
    return {seconds:p95(rates),origin:files.map(f=>path.relative(ROOT,path.join(dir,f)))};
}

function main(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(HERE,'batches')+path.sep));
    fs.copyFileSync(path.join(ROOT,'OT-SSL-DD/method1.py'),path.join(batch,'method1.py'),fs.constants.COPYFILE_EXCL);
    const source=fs.readFileSync(path.join(batch,'method1.py'),'utf8');
    const found=source.match(/parser.add_argument\('--similarity', choices=\[([^\]]+)\]/);assert.ok(found);
    const available=[...found[1].matchAll(/'([^']+)'/g)].map(m=>m[1]);
    assert.deepEqual(available.filter(x=>x!=='cosine').sort(),[...METRICS].sort());
    const files=[...fs.readdirSync(BENCH).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        ...['method1.py','eval.py','utils.py','networks.py','dd_teachers.py','dm_dsa.py'].map(f=>'OT-SSL-DD/'+f),
        'Evaluate_benchamrk/mipa_random128/worker.py','Evaluate_benchamrk/digits_simclr/datasets.py',
        'Evaluate_benchamrk/fashionmnist/dataset.py','Evaluate_benchamrk/cross_arch_ablation/plan.js',
        'Evaluate_benchamrk/mipa_similarity_ablation/worker.py','Evaluate_benchamrk/mipa_similarity_ablation/plan.js',
        'Evaluate_benchamrk/submit_mipa_similarity_ablation_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(ROOT,f))]));
    const queue=new Set(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').split('\n').filter(Boolean).map(l=>l.split('|')[0].trim()));
    const jobs=[],skipped=[],preflight=new Map();let index=0;
    for(const source of SOURCES){
        const args=generationArgs(source,.2),teacherFolder=path.join(args.teacher_path,source,'ResNet18/full/pt100/simclr/seed_000');
        const teacherFile=path.join(teacherFolder,'encoder.pt'),teacherMarker=path.join(teacherFolder,'ssl_complete.json');
        const marker=read(teacherMarker),teacherConfig=read(path.join(teacherFolder,'config.json'));
        assert.ok(marker.complete&&marker.epochs===1000);assert.equal(teacherConfig.identity.seed,0);
        assert.deepEqual(marker.identity,teacherConfig.identity);assert.equal(hash(teacherFile),marker.checkpoint_file_sha256);
        const teacher={path:path.relative(ROOT,teacherFile),file_sha256:marker.checkpoint_file_sha256,marker_sha256:hash(teacherMarker)};
        const teacherEpoch=p95(records(path.join(teacherFolder,'ssl.jsonl')).slice(-50).map(r=>r.seconds));
        const preparedFile=source==='SVHN'?path.join(BENCH,'digits_simclr/prepared.json'):source==='FashionMNIST'?path.join(BENCH,'fashionmnist/full/prepared.json'):null;
        const prepared=preparedFile?read(preparedFile):null;
        for(const subset of SIZES){
            const oldRoot=source==='SVHN'?'digits_subset_simclr2k':source==='FashionMNIST'?'fashionmnist/subset':subset===2?'subset_simclr2k_1ssl5probes':'cifar_small_simclr2k_1ssl5probes';
            const baseline=path.join(BENCH,oldRoot,'results',source,'ResNet18/random','pt'+subset,'simclr/ssl_seed_000');
            const origin=path.join(baseline,'config.json'),old=read(origin),indices=old.identity.selection.indices,count=Math.floor(COUNTS[source]*subset/100);
            assert.equal(old.identity.seed,0);assert.equal(old.identity.selection_method,'random');
            assert.deepEqual([old.signature.source,old.signature.subset],[source,subset]);
            assert.equal(indices.length,count);assert.equal(new Set(indices).size,count);
            assert.ok(indices.every(i=>Number.isInteger(i)&&i>=0&&i<COUNTS[source]));
            if(prepared)assert.equal(old.signature.source_fingerprint,prepared.datasets[source].train_fingerprint);
            const selection={method:'saved_random_subset',seed:0,config:path.relative(ROOT,origin),config_sha256:hash(origin),
                source_fingerprint:old.signature.source_fingerprint,selection_sha256:old.identity.selection.selection_sha256,indices};
            const generation=generationRate(source,subset,count);
            const sslRate=p95(records(path.join(baseline,'ssl.jsonl')).slice(-200).map(r=>r.seconds));
            const probeRates=Object.fromEntries([1,5].map(lbl=>[lbl,p95([0,1,2,3,4].map(seed=>read(path.join(baseline,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`)).seconds))]));
            for(const metric of METRICS){
                const account=(++index)%5===0?'aip-yiweilu':'aip-boyuwang';
                const name='mips_'+source+'_p'+String(subset).replace('.','p')+'_'+metric;
                if(queue.has(name)){skipped.push({name,reason:'active'});continue;}
                const unit={name,source,subset,similarity:metric,metric_parameters:PARAMS[metric],generation_args:generationArgs(source,subset),
                    selection,teacher,code_hashes,account,evaluation:{ssl:'simclr',ssl_epochs:2000,ssl_seed:0,linear_epochs:100,
                        probe_seeds:[0,1,2,3,4],main_labels:[1,5]}};
                if(preparedFile)Object.assign(unit,{prepared_file:path.relative(ROOT,preparedFile),prepared_sha256:hash(preparedFile)});
                const contract={generation_args:unit.generation_args,selection,teacher,code_hashes,similarity:metric,metric_parameters:PARAMS[metric]};
                const gen=path.join(ROOT,'OT-SSL-DD/ours_syntehtic_similarity_lr0p002_b128_i10k',source,'ResNet18BN','pt'+subset,metric);
                const out=path.join(HERE,'results',source,'ResNet18BN','pt'+subset,metric,'simclr/ssl_seed_000');
                let iteration=0,epoch=0,missing={1:5,5:5};
                const state=path.join(gen,'generation_state.json');
                if(fs.existsSync(state)){const s=read(state);assert.deepEqual(s.contract,contract);iteration=s.iteration;assert.ok(iteration>=0&&iteration<=10000);}
                const configFile=path.join(out,'config.json'),markerFile=path.join(out,'ssl_complete.json');
                const config=fs.existsSync(configFile)?read(configFile):null;
                if(config){assert.deepEqual(config.signature.generation_args,unit.generation_args);assert.deepEqual(config.signature.selection,selection);assert.deepEqual(config.signature.driver_code_hashes,code_hashes);}
                if(fs.existsSync(markerFile)){const m=read(markerFile);assert.ok(m.complete&&m.epochs===2000);assert.deepEqual(m.identity,config.identity);epoch=2000;}
                else if(fs.existsSync(path.join(out,'training.pt'))){assert.ok(config);epoch=epochHeader(path.join(out,'training.pt'),config.identity);}
                assert.ok(epoch>=0&&epoch<=2000);
                for(const lbl of [1,5])for(let seed=0;seed<5;seed++){
                    const file=path.join(out,'evaluations',`${source}_lbl${lbl}_probe${seed}.json`);
                    if(fs.existsSync(file)){const r=read(file);assert.ok(r.complete&&r.encoder_unchanged&&epoch===2000);assert.deepEqual(r.expected.identity,config.identity);missing[lbl]--;}
                }
                if(iteration===10000&&epoch===2000&&!missing[1]&&!missing[5]){skipped.push({name,reason:'complete'});continue;}
                const factor=['mmd-dist','dm','energy','sinkhorn'].includes(metric)?2:1;
                const whitening=metric==='whitened-cosine'&&!fs.existsSync(state)?teacherEpoch:0;
                const seconds=Math.ceil((10000-iteration)*generation.seconds*factor+(2000-epoch)*sslRate+
                    missing[1]*probeRates[1]+missing[5]*probeRates[5]+whitening+20+60);
                assert.ok(seconds>0&&seconds<604800);
                const job={...unit,seconds,startup_seconds:20,buffer_seconds:60,timing:{approximate:true,
                    generation_seconds_per_iteration:generation.seconds,generation_origin:generation.origin,metric_cost_factor:factor,
                    ssl_seconds_per_epoch_p95:sslRate,probe_seconds_p95:probeRates,whitening_setup_seconds:whitening,
                    note:'Existing batch-128 timings; two-sample similarities estimated at twice the old generation cost.'}};
                const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
                jobs.push({...job,manifest});if(!preflight.has(source))preflight.set(source,manifest);
            }
        }
    }
    assert.equal(index,96);assert.equal(jobs.length+skipped.length,96);
    const coverage={expected_jobs:96,metrics:METRICS,datasets:SOURCES,sizes:SIZES,synthetic_sets:96,ssl_encoders:96,
        expected_probes:960,labels:[1,5],probes_per_label:5,distillation_iterations:10000,distillation_lr:.002,
        distillation_batch:128,ssl_epochs:2000,probe_epochs:100,teacher_seed:0,initialization_seed:0,
        jobs:jobs.length,skipped,accounts:{'aip-boyuwang':jobs.filter(j=>j.account==='aip-boyuwang').length,
        'aip-yiweilu':jobs.filter(j=>j.account==='aip-yiweilu').length},
        walltimes:jobs.map(j=>({name:j.name,account:j.account,seconds:j.seconds})),tables_modified:false};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n',{flag:'wx'});
    fs.writeFileSync(path.join(batch,'jobs.tsv'),jobs.map(j=>[j.name,j.account,j.seconds,j.manifest].join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    fs.writeFileSync(path.join(batch,'preflight.tsv'),[...preflight].map(([source,file])=>[source,file].join('\t')).join('\n')+'\n',{flag:'wx'});
    console.log(JSON.stringify({jobs:jobs.length,skipped:skipped.length,accounts:coverage.accounts,
        shortest_minutes:jobs.length?Math.ceil(Math.min(...jobs.map(j=>j.seconds))/60):0,
        longest_minutes:jobs.length?Math.ceil(Math.max(...jobs.map(j=>j.seconds))/60):0}));
}
if(require.main===module)main(process.argv[2]);
