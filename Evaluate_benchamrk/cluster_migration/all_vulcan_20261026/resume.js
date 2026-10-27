// Retry only the three timed-out cross-architecture chains, using Killarney timings.
const fs=require('fs'),path=require('path'),cp=require('child_process'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.resolve(here,'../..');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const run=(cmd,args)=>cp.execFileSync(cmd,args,{encoding:'utf8'}).trim();
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert.match(run('scontrol',['show','config']),/ClusterName\s*=\s*killarney/);
assert.ok(run('sinfo',['-h','-o','%G','--Node']).includes('gpu:l40s:'));
const jobs=read(path.join(here,'jobs.json')).jobs.filter(j=>j.name.startsWith('xa_'));
assert.equal(jobs.length,27);
const previous=read(path.join(here,'submitted.json'));
const states=new Map(run('sacct',['-X','-j',jobs.map(j=>previous[j.old_job]).join(','),'-n','-P','--format=JobIDRaw,State']).split('\n').filter(Boolean).map(line=>line.split('|')));
const receipt=path.join(here,'resumed.json'),ids=fs.existsSync(receipt)?read(receipt):{};
const queue=run('squeue',['-h','-u','mmoslem3','-o','%i|%j']).split('\n').filter(Boolean).map(line=>line.split('|').map(s=>s.trim()));
const progress=new Map(),minutes=new Map();
for(const job of jobs){
    assert.ok(['TIMEOUT','CANCELLED'].includes(states.get(previous[job.old_job])), 'Previous run is not stopped: '+job.name);
    const queued=queue.find(r=>r[1]===job.name);
    if(queued)assert.equal(queued[0],ids[job.old_job],'Already queued: '+job.name);
    const task=read(job.args[2]);
    assert.equal(task.kind,'full');
    for(const [file,sha]of Object.entries(task.code_hashes))assert.equal(hash(path.join(bench,'..',file)),sha,'Code changed: '+file);
    if(task.stage!=='ssl')continue;
    let start;
    if(job.dependencies.length){
        assert.equal(job.dependencies.length,1);
        start=read(jobs.find(j=>j.old_job===job.dependencies[0]).args[2]).end_epoch;
    }else{
        const folder=path.join(bench,'cross_arch_ablation/results',task.source,task.model,'full/pt100/simclr/ssl_seed_000');
        const saved=read(path.join(folder,'progress.json')),config=read(path.join(folder,'config.json'));
        assert.deepEqual(saved.identity,config.identity);
        assert.equal(config.signature.protocol.ssl.full_epochs,1000);
        assert.ok(fs.statSync(path.join(folder,'training.pt')).size>1000000);
        start=saved.epoch;
        progress.set(job.old_job,start);
    }
    assert.ok(Number.isInteger(start)&&start<task.end_epoch&&start%50===0);
    // Measured ~101/139 seconds per epoch; round upward and add one minute.
    minutes.set(job.old_job,Math.ceil(((task.model==='ResNet34'?103:141)*(task.end_epoch-start)+60)/60));
}
const pending=jobs.filter(j=>!ids[j.old_job]);
while(pending.length){
    const index=pending.findIndex(j=>j.dependencies.every(id=>ids[id]));
    assert.ok(index>=0,'Unresolved dependency');
    const job=pending.splice(index,1)[0],duration=minutes.get(job.old_job);
    const wall=duration===undefined?job.time:Math.floor(duration/60)+':'+String(duration%60).padStart(2,'0')+':00';
    const logs=path.join(bench,'cross_arch_ablation/logs');
    const args=['--parsable','--hold','--job-name='+job.name,'--account='+job.account,
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=6G','--gres=gpu:l40s:1',
        '--time='+wall,'--chdir='+bench,'--output='+path.join(logs,'%x-%j.out'),'--error='+path.join(logs,'%x-%j.err')];
    if(job.dependencies.length)args.push('--dependency=afterok:'+job.dependencies.map(id=>ids[id]).join(':'),'--kill-on-invalid-dep=yes');
    const id=run('sbatch',[...args,job.script,...job.args]).split(';')[0];
    assert.match(id,/^\d+$/);ids[job.old_job]=id;
    fs.writeFileSync(receipt+'.tmp',JSON.stringify(ids,null,2)+'\n');fs.renameSync(receipt+'.tmp',receipt);
    console.log(id+' '+job.account+' '+wall+' '+job.name+(progress.has(job.old_job)?' RESUME epoch '+progress.get(job.old_job):''));
}
for(const id of Object.values(ids)){
    const details=run('scontrol',['show','job',id]);
    if(/Reason=JobHeldUser\b/.test(details))run('scontrol',['release',id]);
}
console.log('Released '+Object.keys(ids).length+' replacement jobs. A2MIM and completed probes were not resubmitted.');
