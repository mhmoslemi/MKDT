const fs=require('fs'),p=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const HERE=__dirname,BENCH=p.dirname(HERE),ROOT=p.dirname(BENCH);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=values=>{
    assert.ok(values.length&&values.every(value=>Number.isFinite(value)&&value>0));
    return values.sort((a,b)=>a-b)[Math.ceil(.95*values.length)-1];
};
const rounded=seconds=>Math.ceil(seconds/60)*60;

function main(run,data){
    run=fs.realpathSync(run);data=fs.realpathSync(data);
    assert.ok(run.startsWith(HERE+p.sep));
    const calibration=read(p.join(BENCH,'mipa_random128/timing.json'));
    assert.equal(calibration.gpu,'NVIDIA L40S');
    const codeFiles=fs.readdirSync(BENCH).filter(name=>name.endsWith('.py')).map(name=>p.join('Evaluate_benchamrk',name));
    codeFiles.push('networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        'OT-SSL-DD/method1.py','OT-SSL-DD/eval.py','OT-SSL-DD/utils.py','OT-SSL-DD/networks.py',
        'OT-SSL-DD/dd_teachers.py','OT-SSL-DD/dm_dsa.py','Evaluate_benchamrk/mipa_multilayer/worker.py',
        'Evaluate_benchamrk/iic_init/distill_main_aux.py','Evaluate_benchamrk/iic_init/main_aux_plan.js',
        'Evaluate_benchamrk/submit_iic_main_aux_k4_vulcan.sh');
    const code_hashes=Object.fromEntries([...new Set(codeFiles)].sort().map(file=>[file,hash(p.join(ROOT,file))]));
    const protocol=read(p.join(BENCH,'protocol.json'));
    assert.equal(protocol.ssl.batch_size,256);assert.equal(protocol.ssl.optimizer,'lars');
    assert.equal(protocol.methods.simclr.lr,.4);assert.equal(protocol.methods.simclr.temperature,.2);
    assert.deepEqual(protocol.methods.simclr.projector,[2048,256]);assert.equal(protocol.evaluation.epochs,100);
    assert.equal(protocol.evaluation.lr,.1);assert.equal(protocol.evaluation.weight_decay,0);
    fs.mkdirSync(p.join(run,'jobs'),{recursive:true});
    let index=0;
    for(const source of ['CIFAR10','CIFAR100']){
        const iic_run=p.join(HERE,'runs',source+'_ResNet18_e150_lu_killarney');
        const configFile=p.join(iic_run,'config.json'),completeFile=p.join(iic_run,'complete.json');
        const config=read(configFile),complete=read(completeFile),clusters=p.join(iic_run,'clusters.pt');
        assert.equal(config.dataset,source);assert.equal(config.epochs,150);
        assert.equal(complete.complete,true);assert.equal(complete.epochs,150);assert.equal(complete.teacher_unchanged,true);
        assert.equal(hash(clusters),complete.outputs['clusters.pt'],'Saved clustering changed: '+clusters);
        const teacher=config.teacher.path,marker=p.join(p.dirname(teacher),'ssl_complete.json');
        assert.equal(hash(teacher),config.teacher.file_sha256);assert.equal(hash(marker),config.teacher.marker_sha256);
        const input_hashes=Object.fromEntries([configFile,completeFile,clusters,teacher,marker].map(file=>[file,hash(file)]));
        const required=source==='CIFAR10'?['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch','batches.meta'].map(f=>p.join(data,'cifar-10-batches-py',f)):
            ['train','test','meta'].map(f=>p.join(data,'cifar-100-python',f));
        for(const file of required)assert.ok(fs.existsSync(file)&&fs.statSync(file).isFile(),'Dataset missing: '+file);
        for(const percentage of [.2,1,2]){
            const tag=String(percentage).replace('.','p'),account=index++%2===0?'aip-boyuwang':'aip-yiweilu';
            const old=p.join(BENCH,'mipa_random128/results',source,'ResNet18BN','pt'+percentage,'simclr/ssl_seed_000');
            const rates=fs.readFileSync(p.join(old,'ssl.jsonl'),'utf8').trim().split('\n').map(line=>JSON.parse(line).seconds);
            let probeSeconds=0;
            for(const labels of [1,5]){
                const times=[];
                for(let seed=0;seed<5;seed++){
                    const result=read(p.join(old,'evaluations',source+'_lbl'+labels+'_probe'+seed+'.json'));
                    assert.equal(result.complete,true);times.push(result.seconds);
                }
                probeSeconds+=5*p95(times);
            }
            const rate=calibration.units[source+'_p'+tag].generation_seconds_p95;
            assert.ok(rate>0&&Number.isFinite(rate));
            // Four-view cost is extrapolated from measured one-view updates, not a new hardware benchmark.
            const generation_seconds=rounded(4*rate*10000+120);
            const evaluation_seconds=rounded(2000*p95(rates)+probeSeconds+120);
            const save_path=p.join(ROOT,'OT-SSL-DD/ours_syntehtic_iic_main_aux_k4_10k');
            const manifest=p.join(run,'jobs',source+'_p'+tag+'.json');
            const job={source,percentage,account,iic_run,data_path:data,save_path,Iteration:10000,cosine_views:4,
                lr_img:.001,batch_real:128,temperature:.2,teacher_seed:0,ssl_epochs:2000,
                ssl_seed:0,probe_seeds:[0,1,2,3,4],labels:[1,5],linear_epochs:100,
                code_hashes,input_hashes,generation_seconds,evaluation_seconds,
                timing_basis:'Generation: four times measured one-view p95 update cost; evaluation: completed matching SSL/probe p95 times; each stage adds 120 seconds for setup/buffer'};
            if(fs.existsSync(manifest))assert.deepEqual(read(manifest),job,'Existing submission settings changed: '+manifest);
            else fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
            const job_sha=hash(manifest),output=p.join(save_path,source,'ResNet18BN','pt'+percentage);
            const result=p.join(output,'res_IIC_'+source+'_ResNet18BN_'+percentage+'percent_final.pt');
            const genMarker=p.join(output,'generation_complete.json');
            let generation_done=false,evaluation_done=false;
            if(fs.existsSync(genMarker)){
                const saved=read(genMarker);
                assert.equal(saved.complete,true);assert.equal(saved.iteration,10000);assert.equal(saved.cosine_views,4);
                assert.equal(saved.job_file_sha256,job_sha);assert.equal(saved.checkpoint_file_sha256,hash(result));
                generation_done=true;
            }
            const evaluation=p.join(output,'evaluation/simclr/ssl_seed_000'),evalMarker=p.join(evaluation,'complete.json');
            if(fs.existsSync(evalMarker)){
                const saved=read(evalMarker);
                assert.equal(saved.complete,true);assert.equal(saved.job_file_sha256,job_sha);
                assert.equal(saved.ssl_epochs,2000);assert.deepEqual(saved.probe_seeds,[0,1,2,3,4]);
                assert.equal(saved.synthetic_file_sha256,hash(result));
                assert.equal(saved.encoder_file_sha256,hash(p.join(evaluation,'encoder.pt')));
                for(const labels of [1,5]){
                    const cell=read(p.join(evaluation,'summary_'+source+'_lbl'+labels+'.json'));
                    assert.equal(cell.complete,true);assert.equal(cell.completed_probes,5);
                }
                evaluation_done=true;
            }
            console.log([source,tag,account,generation_seconds,evaluation_seconds,manifest,+generation_done,+evaluation_done].join('\t'));
        }
    }
    assert.equal(index,6);
}

if(require.main===module)main(process.argv[2],process.argv[3]);
