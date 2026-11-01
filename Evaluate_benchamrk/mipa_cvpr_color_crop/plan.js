const fs = require('fs'), path = require('path'), crypto = require('crypto'), assert = require('assert/strict');
const here = __dirname, bench = path.dirname(here), root = path.dirname(bench), original = path.join(bench,'mipa_cvpr_color');
const read = file => JSON.parse(fs.readFileSync(file,'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifestFile = path.join(here,'manifest.json'), snapshot = path.join(here,'snapshot');
const roundMinute = seconds => Math.ceil(seconds/60)*60;
if (!fs.existsSync(manifestFile)) {
    const previous = read(path.join(original,'manifest.json')).jobs;
    assert.equal(previous.length,12);
    const oldMethod = fs.readFileSync(path.join(original,'snapshot/method1_cvpr.py'),'utf8');
    const oldWorker = fs.readFileSync(path.join(original,'snapshot/worker.py'),'utf8');
    assert.equal(fs.readFileSync(path.join(snapshot,'method1_cvpr.py'),'utf8'),oldMethod.replace("    args.distill_aug_strategy = 'color_cutout_scale'","    args.distill_aug_strategy = 'color_crop_cutout_scale'"));
    assert.equal(fs.readFileSync(path.join(snapshot,'worker.py'),'utf8'),oldWorker.replace("saved['iic_params']['strategy'] == 'color_cutout_scale'","saved['iic_params']['strategy'] == 'color_crop_cutout_scale'").replace("experiment='mipa_cvpr_color'","experiment='mipa_cvpr_color_crop'"));
    // Measured p95 seconds/iteration from the previous batch's synced generation logs.
    const rates = {CIFAR10:[.06,.17,.43], CIFAR100:[.06,.17,.25], TinyImageNet:[.26,.96,1.87], SVHN:[.09,.23,.38]};
    const jobs = previous.map(old => {
        for (const [file,expected] of Object.entries(old.code_hashes)) assert.equal(hash(file),expected,'Changed source dependency: '+file);
        assert.equal(hash(old.teacher),old.teacher_sha256,'Teacher changed');
        const generation_args = {...old.generation_args,distill_aug_strategy:'color_crop_cutout_scale'};
        const {dataset,percentage,Iteration} = generation_args;
        const code_hashes = Object.fromEntries(Object.entries(old.code_hashes).map(([file,expected]) => {
            if (path.dirname(file) !== path.join(original,'snapshot')) return [file,expected];
            const replacement = path.join(snapshot,path.basename(file));
            return [replacement,hash(replacement)];
        }));
        const {manifest:oldManifest,...oldJob} = old;
        const rate = rates[dataset][[.2,1,2].indexOf(percentage)];
        assert.ok(Number.isFinite(rate));
        const job = {...oldJob,account:'aip-boyuwang',generation_args,code_hashes,
            synthetic_dir:path.join(root,'OT-SSL-DD/ours_syntehtic_cvpr_color_crop',dataset,'ResNet18BN','pt'+percentage),
            evaluation_dir:path.join(here,'results',dataset,'ResNet18BN','pt'+percentage,'simclr/ssl_seed_000'),
            generation_seconds:Math.max(old.generation_seconds,roundMinute(Iteration*rate*1.25+120)),
            evaluation_seconds:roundMinute(old.evaluation_seconds*1.5),
            timing:{...old.timing,generation_seconds_per_iteration:rate,
                basis:'Synced cvprg log p95 with 25% allowance and 120s buffer; evaluation limit increased 50% after observed probe timeouts'}};
        assert.notEqual(job.synthetic_dir,old.synthetic_dir);
        assert.notEqual(job.evaluation_dir,old.evaluation_dir);
        const manifest = path.join(snapshot,job.name+'.json');
        if (fs.existsSync(manifest)) assert.deepEqual(read(manifest),job);
        else fs.writeFileSync(manifest,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        return {...job,manifest};
    });
    fs.writeFileSync(manifestFile,JSON.stringify({created:new Date().toISOString(),jobs},null,2)+'\n',{flag:'wx'});
}
const jobs = read(manifestFile).jobs;
assert.equal(jobs.length,12);
for (const job of jobs) {
    assert.equal(job.account,'aip-boyuwang');
    assert.equal(job.generation_args.distill_aug_strategy,'color_crop_cutout_scale');
    const {manifest,...workerJob} = job;
    assert.deepEqual(read(manifest),workerJob);
    for (const [file,expected] of Object.entries(job.code_hashes)) assert.equal(hash(file),expected,'Changed dependency: '+file);
    assert.equal(hash(job.teacher),job.teacher_sha256);
    const genDone = path.join(job.synthetic_dir,'complete.json'), evalDone = path.join(job.evaluation_dir,'complete.json');
    if (fs.existsSync(evalDone) && read(evalDone).complete) continue;
    const generationDone = fs.existsSync(genDone) && read(genDone).complete;
    process.stdout.write([job.name,job.account,job.memory,generationDone?0:job.generation_seconds,job.evaluation_seconds,manifest].join('\t')+'\n');
}
