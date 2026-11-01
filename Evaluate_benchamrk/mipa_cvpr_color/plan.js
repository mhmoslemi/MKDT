const fs = require('fs'), path = require('path'), crypto = require('crypto'), assert = require('assert/strict');
const here = __dirname, bench = path.dirname(here), root = path.dirname(bench);
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const p95 = values => { assert.ok(values.length && values.every(x => Number.isFinite(x) && x > 0)); return values.sort((a,b) => a-b)[Math.ceil(.95*values.length)-1]; };
const manifestFile = path.join(here, 'manifest.json');
if (!fs.existsSync(manifestFile)) {
    const snapshot = path.join(here, 'snapshot');
    fs.mkdirSync(snapshot, {recursive:true});
    for (const [from,to] of [[path.join(root,'OT-SSL-DD/method1_cvpr.py'),'method1_cvpr.py'],[path.join(here,'worker.py'),'worker.py']]) {
        fs.copyFileSync(from,path.join(snapshot,to),fs.constants.COPYFILE_EXCL);
    }
    const dependencies = ['OT-SSL-DD/eval.py','OT-SSL-DD/dm_dsa.py','OT-SSL-DD/dd_teachers.py','OT-SSL-DD/utils.py','OT-SSL-DD/networks.py',
        'networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/common.py','Evaluate_benchamrk/data.py',
        'Evaluate_benchamrk/train_ssl.py','Evaluate_benchamrk/evaluate.py','Evaluate_benchamrk/protocol.json',
        'Evaluate_benchamrk/digits_simclr/datasets.py','Evaluate_benchamrk/digits_simclr/prepared.json','Evaluate_benchamrk/fashionmnist/dataset.py'];
    const code_hashes = Object.fromEntries([...dependencies.map(f=>path.join(root,f)),path.join(snapshot,'method1_cvpr.py'),path.join(snapshot,'worker.py')].map(f=>[f,hash(f)]));
    const measured = read(path.join(bench,'mipa_random128/timing.json')).units;
    const jobs = [];
    for (const source of ['CIFAR10','CIFAR100','TinyImageNet','SVHN']) for (const subset of [.2,1,2]) {
        const teacher_seed = source === 'TinyImageNet' ? 1 : 0;
        const teacher = path.join(bench,source==='SVHN'?'digits_simclr/checkpoints':'full_ssl_checkpoints_1000ep_3seeds',source,'ResNet18/full/pt100/simclr',`seed_00${teacher_seed}/encoder.pt`);
        assert.ok(read(path.join(path.dirname(teacher),'ssl_complete.json')).complete);
        const generation_args = {dataset:source,percentage:subset,seed:0,Iteration:({'.2':10000,'0.2':10000,'1':8000,'2':6000})[subset],
            lr_img:.0015,batch_real:128,temperature:.2,beta:1,prototype_pool_size:0,distill_aug_strategy:'color_cutout_scale',feature_batch:128,single_channel:false};
        const name = `${source}_pt${String(subset).replace('.','p')}`;
        const synthetic_dir = path.join(root,'OT-SSL-DD/ours_syntehtic_cvpr_color',source,'ResNet18BN','pt'+subset);
        const evaluation_dir = path.join(here,'results',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
        const previous = source === 'SVHN'
            ? path.join(bench,'mipa_multilayer/batches/SVHN_cos8_pt'+subset+'_20261030T16310'+(subset===2?'4Z_vASPvI':subset===1?'3Z_PIzgna':'3Z_jBEyg6'),'results/SVHN/ResNet18BN','pt'+subset,'simclr/ssl_seed_000')
            : path.join(bench,'mipa_random128/results',source,'ResNet18BN','pt'+subset,'simclr/ssl_seed_000');
        const sslLines = fs.readFileSync(path.join(previous,'ssl.jsonl'),'utf8').trim().split('\n').slice(-200).map(JSON.parse);
        const sslRate = p95(sslLines.map(row=>row.seconds));
        const probeSeconds = [1,5].map(labels => p95(Array.from({length:5},(_,seed)=>read(path.join(previous,'evaluations',`${source}_lbl${labels}_probe${seed}.json`)).seconds)));
        const generationRate = source === 'SVHN' ? ({'.2':.07,'0.2':.07,'1':.22,'2':.4})[subset] : measured[`${source}_p${String(subset).replace('.','p')}`].generation_seconds_p95 * 1.3;
        const job = {name,account:'aip-boyuwang',generation_args,teacher,teacher_seed,teacher_sha256:hash(teacher),code_hashes,
            data_root:'/scratch/mmoslem3/data',synthetic_dir,evaluation_dir,
            generation_seconds:Math.ceil((generationRate*generation_args.Iteration+120)/60)*60,
            evaluation_seconds:Math.ceil((2000*sslRate*1.15+5*(probeSeconds[0]+probeSeconds[1])*1.15+120)/60)*60,
            memory:source==='TinyImageNet'?'8G':'6G',timing:{generation_seconds_per_iteration:generationRate,ssl_seconds_per_epoch:sslRate,probe_seconds:probeSeconds}};
        const file = path.join(snapshot,name+'.json');
        fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        jobs.push({...job,manifest:file});
    }
    fs.writeFileSync(manifestFile,JSON.stringify({created:new Date().toISOString(),jobs},null,2)+'\n',{flag:'wx'});
}
const jobs = read(manifestFile).jobs;
assert.equal(jobs.length,12);
assert.equal(jobs.filter(j=>j.account==='aip-boyuwang').length,12);
for (const job of jobs) {
    const {manifest,...workerJob} = job;
    assert.deepEqual(read(manifest),workerJob,'Worker manifest differs: '+manifest);
    for (const [file,expected] of Object.entries(job.code_hashes)) assert.equal(hash(file),expected,'Changed dependency: '+file);
    assert.equal(hash(job.teacher),job.teacher_sha256);
    const genDone = path.join(job.synthetic_dir,'complete.json'), evalDone = path.join(job.evaluation_dir,'complete.json');
    if (fs.existsSync(evalDone) && read(evalDone).complete) continue;
    const generationDone = fs.existsSync(genDone) && read(genDone).complete;
    process.stdout.write([job.name,job.account,job.memory,generationDone?0:job.generation_seconds,job.evaluation_seconds,job.manifest].join('\t')+'\n');
}
