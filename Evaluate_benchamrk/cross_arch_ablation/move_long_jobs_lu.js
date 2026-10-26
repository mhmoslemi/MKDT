// Change only Slurm accounting/dependencies; keep the fingerprinted training code intact.
const fs=require('fs'),path=require('path'),cp=require('child_process'),assert=require('assert/strict');
const here=__dirname,bench=path.dirname(here);
const run=(cmd,args)=>cp.execFileSync(cmd,args,{encoding:'utf8'}).trim();
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const seconds=s=>{const [day,time]=s.includes('-')?s.split('-'):['0',s];const parts=time.split(':').map(Number);if(parts.length===2)parts.unshift(0);assert.equal(parts.length,3);return +day*86400+parts[0]*3600+parts[1]*60+parts[2];};
const parent=s=>s==='-'?null:(s.match(/--dependency=afterok:(\d+)/)||[])[1];
function queue(){
    return new Map(run('squeue',['-h','-u',run('id',['-un']),'-o','%i|%a|%j|%T|%l|%E']).split('\n').filter(Boolean).map(line=>{
        const [id,account,name,state,time,dependency]=line.split('|').map(s=>s.trim());return [id,{id,account,name,state,time,dependency}];}));
}
function selfCheck(){
    assert.equal(seconds('03:00:00'),10800);assert.ok(!(seconds('03:00:00')>10800));
    assert.ok(seconds('03:03:00')>10800);assert.equal(seconds('1-00:00:00'),86400);
    assert.equal(seconds('49:00'),2940);assert.equal(seconds('7:00'),420);
    assert.equal(parent('--dependency=afterok:123 --kill-on-invalid-dep=yes'),'123');assert.equal(parent('-'),null);
    const mapping=new Map([['123','456']]);assert.equal(mapping.get(parent('--dependency=afterok:123')),'456');
}
function main(){
    selfCheck();
    const batch=path.resolve(process.argv[2]);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const apply=process.argv[3]==='--apply';assert.ok(apply||process.argv.length===3);
    const rows=fs.readFileSync(path.join(batch,'submitted.tsv'),'utf8').trim().split('\n').slice(1).map(line=>{
        const [name,id,account,time,dependency]=line.split('\t');return {name,id,account,time,dependency,parent:parent(dependency)};});
    assert.equal(rows.length,108);assert.equal(new Set(rows.map(r=>r.id)).size,108);
    const before=queue(),targets=rows.filter(r=>before.has(r.id)&&seconds(before.get(r.id).time)>10800&&before.get(r.id).account==='aip-boyuwang');
    const selected=new Set(targets.map(r=>r.id));
    const successors=rows.filter(r=>!selected.has(r.id)&&selected.has(r.parent)&&before.has(r.id));
    for(const row of [...targets,...successors]){
        const actual=before.get(row.id);assert.equal(actual.name,row.name);assert.equal(actual.state,'PENDING','Job started; inspect before moving: '+row.id);
        assert.equal(actual.account,'aip-boyuwang');assert.ok(row.name.startsWith('xa_'));
        assert.equal(actual.dependency==='(null)'?null:actual.dependency.match(/^afterok:(\d+)\(/)?.[1],row.parent);
        const task=read(path.join(batch,row.name+'.json'));assert.equal(task.name,row.name);
        const detail=run('scontrol',['show','job','-o',row.id]);
        for(const expected of ['NumCPUs=1','MinMemoryNode=6G','gres/gpu:l40s:1','WorkDir='+bench,path.join(batch,row.name+'.json')])assert.ok(detail.includes(expected),'Unexpected original resources/command: '+expected);
    }
    console.log(JSON.stringify({affected:targets.length,rewire_shorter_probes:successors.length,unchanged:rows.filter(r=>before.has(r.id)&&!selected.has(r.id)).length,apply},null,2));
    if(!apply||!targets.length)return;
    const migration=path.join(batch,'reassigned_lu.json');assert.ok(!fs.existsSync(migration),'Migration receipt already exists; inspect it before retrying');
    const report={phase:'planned',targets,successors,replacements:[],rewired:[],held:[]};
    const save=()=>fs.writeFileSync(migration,JSON.stringify(report,null,2)+'\n');save();
    const map=new Map();
    try{
        for(const row of targets){run('scontrol',['hold',row.id]);report.held.push(row.id);save();}
        report.phase='originals_held';save();
        for(const row of targets){
            const dependency=row.parent?(map.get(row.parent)||row.parent):null;
            if(selected.has(row.parent))assert.ok(map.has(row.parent),'Receipt is not topologically ordered');
            const args=['--parsable','--hold','--job-name='+row.name,'--account=aip-yiweilu','--nodes=1','--ntasks=1',
                '--cpus-per-task=1','--mem=6G','--gres=gpu:l40s:1','--time='+row.time,'--chdir='+bench,
                '--output='+path.join(here,'logs/%x-%j.out'),'--error='+path.join(here,'logs/%x-%j.err')];
            if(dependency)args.push('--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes');
            args.push(path.join(bench,'submit_cross_arch_ablation_wang_vulcan.sh'),'--worker',bench,
                path.join(batch,row.name+'.json'),'/home/mmoslem3/ENV','/scratch/mmoslem3/data');
            const id=run('sbatch',args).split(';')[0];assert.match(id,/^\d+$/);map.set(row.id,id);
            report.replacements.push({...row,new_id:id,new_account:'aip-yiweilu',new_parent:dependency});save();
        }
        report.phase='replacements_held';save();
        for(const row of successors){
            const replacement=map.get(row.parent);assert.ok(replacement);
            run('scontrol',['update','JobId='+row.id,'Dependency=afterok:'+replacement]);
            report.rewired.push({id:row.id,old_parent:row.parent,new_parent:replacement});save();
        }
        report.phase='dependencies_rewired';save();
        for(const r of [...report.replacements,...report.rewired]){
            const id=r.new_id||r.id,expected=r.new_parent,detail=run('scontrol',['show','job','-o',id]);
            if(expected)assert.ok(detail.includes('Dependency=afterok:'+expected+'('),'Dependency was not updated: '+id);
        }
        run('scancel',targets.map(r=>r.id));report.phase='originals_cancelled';save();
        for(const r of report.replacements)run('scontrol',['release',r.new_id]);
        report.phase='complete';save();
        const after=queue();
        for(const r of report.replacements){assert.ok(!after.has(r.id),'Original job still active: '+r.id);assert.equal(after.get(r.new_id)?.account,'aip-yiweilu');}
        for(const r of rows.filter(r=>before.has(r.id)&&!selected.has(r.id))){assert.equal(after.get(r.id)?.account,'aip-boyuwang','Shorter job unexpectedly changed: '+r.id);}
        const tsv='name\told_job\tnew_job\taccount\twalltime\n'+report.replacements.map(r=>[r.name,r.id,r.new_id,'aip-yiweilu',r.time].join('\t')).join('\n')+'\n';
        fs.writeFileSync(path.join(batch,'reassigned_lu.tsv'),tsv);
        console.log('COMPLETE: moved '+targets.length+' jobs to Lu; '+successors.length+' shorter-job dependencies repaired. Receipt: '+migration);
    }catch(error){console.error('Stopped in phase '+report.phase+'. Exact held/replacement IDs are saved in '+migration);throw error;}
}
if(require.main===module)main();
