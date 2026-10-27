// Login-safe plan using exact saved subsets and measured Vulcan K-means timings.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const {rows,replacement,eligible}=require('./tables');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const configurations=[['CIFAR10',2],['CIFAR100',2],['TinyImageNet',0.2],['TinyImageNet',1],['TinyImageNet',2]].flatMap(([source,subset])=>['random','kmeans'].map(kind=>({source,subset,kind})));
const p95=a=>{assert.ok(a.length&&a.every(x=>Number.isFinite(x)&&x>0),'No valid runtime samples');return a.sort((a,b)=>a-b)[Math.ceil(a.length*.95)-1];};
function check(){
    const file=path.join(root,'CVPR_tables/table.tex'),original=fs.readFileSync(file,'utf8');
    const before=rows(original);assert.equal(configurations.length,10);let tested=0;
    for(const c of configurations)for(const labels of [1,5]){
        assert.ok(eligible(c.source,c.subset));
        const summary={...c,labels,complete:true,ssl_epochs:2000,linear_epochs:100,ssl_seed:0,std_ddof:1,
            samples:[0,1,2,3,4].map(probe_seed=>({probe_seed,accuracy_percent:1})),tex_value:'\\dsacc{1.00}{0.00}'};
        const next=rows(replacement(original,summary));assert.equal(next.length,9);
        for(let i=0;i<9;i++)for(let j=0;j<12;j++){
            const target=before[i].source===c.source&&before[i].subset===c.subset;
            const index=(labels===1?2:7)+(c.kind==='random'?0:1);
            const clean=s=>s.replace(/\\bestacc(?=\{)/g,'\\dsacc');
            if(target&&j===index)assert.ok(next[i].tokens[j].includes('\\dsacc{1.00}{0.00}'));
            else assert.equal(clean(next[i].tokens[j]),clean(before[i].tokens[j]),'Unrelated table number changed');
        }
        tested++;
    }
    assert.equal(tested,20);assert.ok(!eligible('CIFAR10',0.2));
    console.log('PASS: 10 encoder configurations, 100 probe jobs, 20 table-only cell replacements; other numbers preserved.');
}
function main(){
    check();const batch=path.resolve(process.argv[2]);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const queue=new Map(fs.readFileSync(process.argv[3],'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const shared=[...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),'networks.py','benchmarks/downstream_data.py'];
    const files=[...shared,'Evaluate_benchamrk/protocol.json',...['worker.py','tables.js','plan.js'].map(f=>'Evaluate_benchamrk/subset_simclr2k_1ssl5probes/'+f),
        'Evaluate_benchamrk/submit_subset_simclr2k_1ssl5probes_wang_vulcan.sh'];
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
    const sharedHashes=Object.fromEntries(shared.map(f=>[f,code_hashes[f]]));
    const jobs=[],timings=[],groups=[],reference=read(path.join(bench,'protocol.json'));
    for(const c of configurations){
        const oldRoot=c.kind==='random'?'random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k':'kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds';
        const selection_config=`${oldRoot}/${c.source}/ResNet18/${c.kind}/pt${c.subset}/simclr/seed_000/config.json`,old=read(path.join(bench,selection_config));
        assert.deepEqual([old.signature.source,old.signature.model,old.signature.selection_method,old.signature.subset,old.signature.ssl,old.identity.seed],[c.source,'ResNet18',c.kind,c.subset,'simclr',0]);
        assert.deepEqual(old.signature.code_hashes,sharedHashes);assert.deepEqual(old.signature.protocol.evaluation,reference.evaluation);
        assert.deepEqual(old.signature.protocol.methods.simclr,reference.methods.simclr);
        const indices=old.identity.selection.indices,count=(c.source==='TinyImageNet'?100000:50000)*c.subset/100;
        assert.equal(indices.length,count);assert.equal(new Set(indices).size,count);
        assert.ok(indices.every(i=>Number.isInteger(i)&&i>=0&&i<(c.source==='TinyImageNet'?100000:50000)));
        const measured=path.join(bench,`kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds/${c.source}/ResNet18/kmeans/pt${c.subset}/simclr/seed_000/ssl.jsonl`);
        const epochSeconds=p95(fs.readFileSync(measured,'utf8').trim().split('\n').slice(-200).map(l=>JSON.parse(l).seconds));
        const probeTimes={};
        for(const labels of [1,5]){
            const values=[];for(let seed=0;seed<6;seed++){
                const file=path.join(bench,`kmeans_eval_results_interactive_6seeds/${c.source}/ResNet18/kmeans/pt${c.subset}/simclr/seed_${String(seed).padStart(3,'0')}/evaluations/${c.source}_lbl${labels}.json`);
                if(!fs.existsSync(file))continue;const r=read(file);if(r.complete&&r.expected.evaluation.epochs===100&&r.seconds>0)values.push(r.seconds);
            }
            probeTimes[labels]=p95(values);
        }
        const id=`s2k5_${c.source}_${c.kind}_p${String(c.subset).replace('.','p')}`;
        const common={...c,selection_config,selection_config_sha256:hash(path.join(bench,selection_config)),code_hashes,buffer_seconds:30};
        const folder=path.join(here,'results',c.source,'ResNet18',c.kind,`pt${c.subset}`,'simclr/ssl_seed_000');
        const marker=fs.existsSync(path.join(folder,'ssl_complete.json'))?read(path.join(folder,'ssl_complete.json')):null;
        if(marker){assert.ok(marker.complete&&marker.epochs===2000);const config=read(path.join(folder,'config.json'));assert.deepEqual(config.signature.driver_code_hashes,code_hashes);assert.deepEqual(marker.identity,config.identity);assert.equal(hash(path.join(folder,'encoder.pt')),marker.checkpoint_file_sha256);}
        const sslName=id+'_ssl';
        if(!marker&&!queue.has(sslName))jobs.push({...common,name:sslName,stage:'ssl',seconds:Math.ceil(2000*epochSeconds+30),dependencies:[]});
        for(const labels of [1,5])for(let probe_seed=0;probe_seed<5;probe_seed++){
            const name=id+`_l${labels}_p${probe_seed}`,file=path.join(folder,`evaluations/${c.source}_lbl${labels}_probe${probe_seed}.json`);
            if(fs.existsSync(file)){const r=read(file);assert.ok(marker&&r.complete&&r.encoder_unchanged&&r.encoder_sha256===marker.encoder_sha256);assert.deepEqual(r.expected.identity,marker.identity);assert.deepEqual(r.expected.evaluation,reference.evaluation);assert.deepEqual(r.expected.evaluation_code_hashes,sharedHashes);continue;}
            if(!queue.has(name))jobs.push({...common,name,stage:'probe',labels,probe_seed,seconds:Math.ceil(probeTimes[labels]+30),dependencies:marker?[]:[sslName]});
        }
        groups.push({...c,selection_config,selection_config_sha256:common.selection_config_sha256,selection_seed:0,ssl_seed:0,ssl_epochs:2000,probe_seeds:[0,1,2,3,4],label_percentages:[1,5],table:'CVPR_tables/table.tex'});
        timings.push({...c,ssl_epoch_seconds_p95:epochSeconds,probe_seconds_p95:probeTimes,measurement:'Vulcan K-means / same BN architecture, data size and benchmark recipe',buffer_seconds:30});
    }
    const seen=new Set(queue.keys());let tsv='';
    for(const job of jobs){for(const dep of job.dependencies)assert.ok(seen.has(dep),'Missing encoder dependency: '+dep);assert.ok(!seen.has(job.name));seen.add(job.name);
        const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});tsv+=[job.name,job.stage,job.seconds,job.dependencies.join(',')||'-',file].join('\t')+'\n';}
    assert.equal(groups.length,10);assert.ok(jobs.filter(j=>j.stage==='ssl').length<=10&&jobs.filter(j=>j.stage==='probe').length<=100);
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const report={expected_encoder_jobs:10,expected_probe_jobs:100,expected_table_cells:20,jobs_to_submit:jobs.length,
        ssl_jobs:jobs.filter(j=>j.stage==='ssl').length,probe_jobs:jobs.filter(j=>j.stage==='probe').length,account:'aip-boyuwang',
        ssl_epochs:2000,probe_epochs:100,buffer_seconds:30,table_scope:'CVPR_tables/table.tex',groups,timings};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({...report,groups:undefined,timings:undefined},null,2));
}
if(require.main===module){if(process.argv[2]==='--check')check();else main();}
