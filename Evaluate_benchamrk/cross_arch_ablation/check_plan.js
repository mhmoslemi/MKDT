// Exercise scheduling only with explicit timing fixtures, never submit these tasks.
const fs=require('fs'),path=require('path'),assert=require('assert/strict'),crypto=require('crypto');
const here=__dirname,root=path.dirname(path.dirname(here));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const {main,epochHeader}=require('./plan');
fs.mkdirSync(path.join(here,'batches'),{recursive:true});
const batch=fs.mkdtempSync(path.join(here,'batches','check_only_'));
fs.writeFileSync(path.join(batch,'queue.tsv'),'');
const synthetic={};
for(const source of ['CIFAR10','CIFAR100'])for(const size of [0.2,1]){
    const file=`OT-SSL-DD/ours_syntehtic/${source}/ResNet18BN/pt${size}/res_IIC_${source}_ResNet18BN_${size}percent_final.pt`;
    synthetic[file]=hash(path.join(root,file));
}
const timing={ssl_batch_seconds:{100:0.1,256:0.2},probe_batch_seconds:0.01};
process.argv[2]=batch;process.argv[3]=path.join(batch,'queue.tsv');
main({complete:true,code_hashes:{},synthetics:synthetic,models:{ResNet18:timing,ResNet34:timing,ResNet50:timing}});
const coverage=JSON.parse(fs.readFileSync(path.join(batch,'coverage.json')));
assert.deepEqual([coverage.jobs_to_submit,coverage.ssl_jobs,coverage.probe_jobs,coverage.full_training_segments],[108,72,36,40]);
assert.equal(coverage.ssl_encoders,36);assert.equal(coverage.total_probes,504);assert.equal(coverage.table_cells,120);
const tasks=fs.readdirSync(batch).filter(f=>f.startsWith('xa_')&&f.endsWith('.json')).map(f=>JSON.parse(fs.readFileSync(path.join(batch,f))));
for(const task of tasks){
    assert.equal(task.buffer_seconds,60);assert.ok(task.seconds>=60);
    if(task.kind==='full'&&task.stage==='ssl')assert.ok(task.end_epoch%100===0&&task.end_epoch<=1000);
    if(task.kind!=='full'&&task.stage==='ssl')assert.equal(task.end_epoch,2000);
    if(task.kind==='ours'){assert.ok(task.synthetic_file.includes('/ResNet18BN/'));assert.equal(task.synthetic_sha256,synthetic[task.synthetic_file]);}
}
for(const source of ['cifar10','cifar100']){
    const rows=fs.readFileSync(path.join(root,`CVPR_tables/ablations/cross_arch_${source}.tex`),'utf8').split('\n').filter(l=>l.includes('% crossarch '));
    assert.equal(rows.length,10);assert.equal(new Set(rows.map(l=>l.slice(l.indexOf('% crossarch ')))).size,10);
    for(const row of rows){assert.equal(row.split('&').length,9);const prefix=row.split('&').slice(0,3).join('&').trimEnd();assert.equal((prefix+' & '+Array(6).fill('score').join(' & ')).split('&').length,9);}
}
const old=path.join(root,'Evaluate_benchamrk/full_ssl_checkpoints_1000ep_3seeds/CIFAR10/ResNet18/full/pt100/simclr/seed_000');
assert.equal(epochHeader(old+'/training.pt',JSON.parse(fs.readFileSync(old+'/config.json')).identity),1000);
console.log('PASS: 108-job DAG, 36 encoders, 504 probes, 120 cells, epoch budgets, image paths and resume header. CHECK ONLY; nothing submitted.');
