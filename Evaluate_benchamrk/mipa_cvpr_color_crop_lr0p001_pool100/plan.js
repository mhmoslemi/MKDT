const fs = require('fs'), path = require('path'), crypto = require('crypto'), assert = require('assert/strict');
const here = __dirname, bench = path.dirname(here), root = path.dirname(bench), original = path.join(bench,'mipa_cvpr_color_crop_lr0p001');
const read = file => JSON.parse(fs.readFileSync(file,'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifestFile = path.join(here,'manifest.json'), snapshot = path.join(here,'snapshot');
if (!fs.existsSync(manifestFile)) {
    const previous = read(path.join(original,'manifest.json')).jobs;
    assert.equal(previous.length,12);
    assert.equal(hash(path.join(snapshot,'method1_cvpr.py')),hash(path.join(original,'snapshot/method1_cvpr.py')));
    assert.equal(fs.readFileSync(path.join(snapshot,'worker.py'),'utf8'),fs.readFileSync(path.join(original,'snapshot/worker.py'),'utf8').replace("experiment='mipa_cvpr_color_crop_lr0p001'","experiment='mipa_cvpr_color_crop_lr0p001_pool100'"));
    const jobs = previous.map(old => {
        for (const [file,expected] of Object.entries(old.code_hashes)) assert.equal(hash(file),expected,'Changed source dependency: '+file);
        assert.equal(hash(old.teacher),old.teacher_sha256,'Teacher changed');
        const generation_args = {...old.generation_args,prototype_pool_size:100};
        const {dataset,percentage} = generation_args;
        const code_hashes = Object.fromEntries(Object.entries(old.code_hashes).map(([file,expected]) => {
            if (path.dirname(file) !== path.join(original,'snapshot')) return [file,expected];
            const replacement = path.join(snapshot,path.basename(file));
            return [replacement,hash(replacement)];
        }));
        const {manifest:oldManifest,...oldJob} = old;
        const job = {...oldJob,account:'aip-yiweilu',gpu:'l40s',generation_args,code_hashes,
            synthetic_dir:path.join(root,'OT-SSL-DD/ours_syntehtic_cvpr_color_crop_lr0p001_pool100',dataset,'ResNet18BN','pt'+percentage),
            evaluation_dir:path.join(here,'results',dataset,'ResNet18BN','pt'+percentage,'simclr/ssl_seed_000')};
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
    assert.equal(job.account,'aip-yiweilu');
    assert.equal(job.gpu,'l40s');
    assert.equal(job.generation_args.lr_img,0.001);
    assert.equal(job.generation_args.prototype_pool_size,100);
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
