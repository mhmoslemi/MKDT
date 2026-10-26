// Move the captured queue, preserving accounts/resources and remapping dependencies.
const fs=require('fs'),path=require('path'),cp=require('child_process'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.resolve(here,'../..');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const run=(cmd,args)=>cp.execFileSync(cmd,args,{encoding:'utf8'}).trim();
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const plan=read(path.join(here,'jobs.json'));
assert.match(run('scontrol',['show','config']),/ClusterName\s*=\s*killarney/);
assert.ok(run('sinfo',['-h','-o','%G','--Node']).includes('gpu:l40s:'));
assert.equal(plan.jobs.length,30);
for(const [file,checksum]of Object.entries(plan.checkpoints))assert.equal(hash(path.join(bench,file)),checksum,'Resume checkpoint differs: '+file);
for(const job of plan.jobs){
    assert.ok(['aip-boyuwang','aip-yiweilu'].includes(job.account));
    assert.ok(fs.existsSync(job.script));
    if(job.name.startsWith('xa_')){
        const manifest=read(job.args[2]);
        for(const [file,checksum]of Object.entries(manifest.code_hashes))assert.equal(hash(path.join(bench,'..',file)),checksum,'Code differs: '+file);
    }
}
const receipt=path.join(here,'submitted.json'),ids=fs.existsSync(receipt)?read(receipt):{};
const queue=run('squeue',['-h','-u','mmoslem3','-o','%i|%j']).split('\n').filter(Boolean).map(s=>s.split('|').map(x=>x.trim()));
for(const job of plan.jobs){const queued=queue.find(r=>r[1]===job.name);if(queued)assert.equal(queued[0],ids[job.old_job],'Job already queued outside this migration: '+job.name);}
const remaining=plan.jobs.filter(j=>!ids[j.old_job]);
while(remaining.length){
    const index=remaining.findIndex(j=>j.dependencies.every(id=>ids[id]));
    assert.ok(index>=0,'Unresolved dependency');
    const job=remaining.splice(index,1)[0];
    const args=['--parsable','--hold','--job-name='+job.name,'--account='+job.account,
        '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=6G','--gres=gpu:l40s:1',
        '--time='+job.time,'--chdir='+bench];
    const logs=path.join(bench,job.name.startsWith('a2mim')?'a2mim_teacher/logs':'cross_arch_ablation/logs');
    fs.mkdirSync(logs,{recursive:true});
    args.push('--output='+path.join(logs,'%x-%j.out'),'--error='+path.join(logs,'%x-%j.err'));
    if(job.signal)args.push('--signal='+job.signal);
    if(job.dependencies.length)args.push('--dependency=afterok:'+job.dependencies.map(id=>ids[id]).join(':'),'--kill-on-invalid-dep=yes');
    const id=run('sbatch',[...args,job.script,...job.args]).split(';')[0];
    assert.match(id,/^\d+$/);ids[job.old_job]=id;
    fs.writeFileSync(receipt+'.tmp',JSON.stringify(ids,null,2)+'\n');fs.renameSync(receipt+'.tmp',receipt);
    console.log(job.old_job+' -> '+id+' '+job.account+' '+job.name);
}
console.log('ALL '+Object.keys(ids).length+' REPLACEMENTS SUBMITTED ON HOLD. Cancel the original Vulcan IDs before releasing.');
