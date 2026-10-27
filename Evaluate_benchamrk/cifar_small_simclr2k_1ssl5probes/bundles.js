// Bundle the unchanged probe manifests; only scheduling and the final buffer differ.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict'),cp=require('child_process');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length,'No completed probe startup measurements');return a.sort((a,b)=>a-b)[Math.ceil(.95*a.length)-1];};
function startupSeconds(){
    const previous=path.join(bench,'subset_simclr2k_1ssl5probes/batches/batch_20261027T165104Z_vraqso');
    const rows=fs.readFileSync(path.join(previous,'submitted.tsv'),'utf8').trim().split('\n').slice(1).map(l=>l.split('\t')).filter(r=>r[1]==='probe'&&/^s2k5_CIFAR/.test(r[0]));
    const accounting=cp.execFileSync('sacct',['-X','-j',rows.map(r=>r[2]).join(','),'-nP','--format=JobIDRaw,JobName%80,State,ElapsedRaw'],{encoding:'utf8'});
    const samples=[];
    for(const line of accounting.trim().split('\n')){
        const [id,name,state,elapsed]=line.split('|').map(s=>s.trim());
        if(state!=='COMPLETED'||!rows.some(r=>r[2]===id&&r[0]===name))continue;
        const t=read(path.join(previous,name+'.json'));
        const file=path.join(bench,'subset_simclr2k_1ssl5probes/results',t.source,'ResNet18',t.kind,'pt'+t.subset,'simclr/ssl_seed_000/evaluations',`${t.source}_lbl${t.labels}_probe${t.probe_seed}.json`);
        if(!fs.existsSync(file))continue;
        const result=read(file),seconds=Number(elapsed)-result.seconds;
        if(result.complete&&Number.isFinite(seconds)&&seconds>=0)samples.push({job:id,seconds});
    }
    return {seconds:p95(samples.map(s=>s.seconds)),samples};
}
function plan(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const queue=fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim()));
    const filtered=queue.filter(r=>!/^s2k5c_.*_l[15]_p[0-4]$/.test(r[0]));
    fs.writeFileSync(path.join(batch,'encoder_queue.tsv'),filtered.map(r=>r.slice(0,2).join('|')).join('\n')+'\n');
    cp.execFileSync(process.execPath,[path.join(here,'plan.js'),batch,path.join(batch,'encoder_queue.tsv')],{stdio:'pipe'});
    const rows=fs.readFileSync(path.join(batch,'jobs.tsv'),'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('\t'));
    const jobs=rows.filter(r=>r[1]==='ssl'),groups=new Map(),overhead=startupSeconds();
    const code_hashes=Object.fromEntries([path.relative(root,__filename),'Evaluate_benchamrk/submit_cifar_small_simclr2k_bundled_wang_vulcan.sh'].map(f=>[f,hash(path.join(root,f))]));
    for(const r of rows.filter(r=>r[1]==='probe')){
        const task=read(r[4]);assert.equal(task.stage,'probe');
        assert.ok(['CIFAR10','CIFAR100'].includes(task.source)&&[.2,1].includes(task.subset));
        const key=task.source+'|'+task.subset;if(!groups.has(key))groups.set(key,[]);
        groups.get(key).push({manifest:r[4],sha256:hash(r[4]),task});
    }
    const bundles=[];
    for(const [key,probes]of groups){
        const [source,size]=key.split('|'),subset=Number(size),name=`s2k5c_${source}_p${size.replace('.','p')}_probes20`;
        assert.ok(probes.length>0&&probes.length<=20);
        assert.equal(new Set(probes.map(p=>[p.task.kind,p.task.labels,p.task.probe_seed].join('|'))).size,probes.length);
        if(queue.some(r=>r[0]===name))continue;
        const dependencies=[...new Set(probes.flatMap(p=>p.task.dependencies))];assert.ok(dependencies.length<=2);
        const seconds=Math.ceil(probes.reduce((total,p)=>total+p.task.seconds-p.task.buffer_seconds+overhead.seconds,0)+60);
        const manifest=path.join(batch,name+'.json');
        const bundle={name,source,subset,stage:'probe_bundle',probes:probes.map(({task,...p})=>p),dependencies,seconds,buffer_seconds:60,startup_measurement:overhead,code_hashes};
        fs.writeFileSync(manifest,JSON.stringify(bundle,null,2)+'\n',{flag:'wx'});
        jobs.push([name,'probe_bundle',String(seconds),dependencies.join(',')||'-',manifest]);bundles.push(bundle);
    }
    assert.ok(bundles.length<=4&&bundles.reduce((n,b)=>n+b.probes.length,0)<=80);
    fs.writeFileSync(path.join(batch,'bundled_jobs.tsv'),jobs.map(r=>r.join('\t')).join('\n')+(jobs.length?'\n':''),{flag:'wx'});
    fs.writeFileSync(path.join(batch,'bundled_coverage.json'),JSON.stringify({account:'aip-boyuwang',ssl_jobs_to_submit:jobs.filter(r=>r[1]==='ssl').length,probe_bundle_jobs:bundles.length,probes:bundles.reduce((n,b)=>n+b.probes.length,0),maximum_probes_per_bundle:20,buffer_seconds_per_bundle:60,startup_measurement:overhead,bundles},null,2)+'\n');
    console.log(`PLAN: ${jobs.filter(r=>r[1]==='ssl').length} new encoders, ${bundles.length} probe bundles, ${bundles.reduce((n,b)=>n+b.probes.length,0)} probes; one 60-second buffer per bundle.`);
}
function run(file,data){
    assert.ok(process.env.SLURM_JOB_ID,'Use a GPU Slurm job');
    file=path.resolve(file);assert.ok(file.startsWith(path.join(here,'batches')+path.sep));
    const bundle=read(file);assert.equal(bundle.stage,'probe_bundle');assert.equal(bundle.buffer_seconds,60);
    assert.ok(bundle.probes.length>0&&bundle.probes.length<=20);
    for(const [f,sha]of Object.entries(bundle.code_hashes))assert.equal(hash(path.join(root,f)),sha,f);
    for(const probe of bundle.probes){
        assert.ok(path.resolve(probe.manifest).startsWith(path.join(here,'batches')+path.sep));
        assert.equal(hash(probe.manifest),probe.sha256,'Probe manifest changed');
        const task=read(probe.manifest);assert.equal(task.stage,'probe');assert.equal(task.source,bundle.source);assert.equal(task.subset,bundle.subset);
        console.log('BUNDLED PROBE:',task.name);
        const result=cp.spawnSync('python',['-u',path.join(here,'worker.py'),probe.manifest,data],{stdio:'inherit',cwd:bench});
        if(result.error)throw result.error;assert.equal(result.status,0,'Probe failed; completed probe files remain reusable');
    }
    console.log('BUNDLE COMPLETE:',bundle.name);
}
if(process.argv[2]==='--run')run(process.argv[3],process.argv[4]);else plan(process.argv[2]);
