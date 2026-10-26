// Only frozen encoders and saved JSONs are inspected on the login node.
const fs=require('fs'),path=require('path'),assert=require('assert/strict'),crypto=require('crypto');
const {rows,key}=require('./tables');
const here=__dirname,bench=path.dirname(here),repo=path.dirname(bench);
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const exists=f=>fs.existsSync(f)&&fs.statSync(f).size>0;
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
function pairs(scheme){
    if(scheme==='one_ssl')return Array.from({length:6},(_,p)=>[0,p]);
    if(scheme==='ten_ssl')return Array.from({length:10},(_,s)=>[s,s]);
    if(scheme==='four_ssl')return [[0,0],[1,1],[2,2],[4,4],[0,3],[1,5]];
    assert.equal(scheme,'two_ssl');return [0,1].flatMap(s=>[0,1,2].map(p=>[s,p]));
}
function main(){
    const batch=path.resolve(process.argv[2]);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const queue=new Set((process.argv[3]||'').split('\n'));
    const hashes=Object.fromEntries([...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py'].map(f=>[f,hash(path.join(repo,f))]));
    const reference=read(path.join(bench,'protocol.json')).evaluation;
    let tsv='',jobs=0,newProbes=0,reused=0,skipped=0;
    for(const group of read(path.join(here,'cells.json'))){
        const tag=group.scheme==='four_ssl'?'seed_':'ssl_seed_';
        const encoders=new Map();
        for(const seed of new Set(pairs(group.scheme).map(p=>p[0]))){
            const folder=path.join(bench,group.checkpoint_group,tag+String(seed).padStart(3,'0'));
            assert.ok(exists(path.join(folder,'config.json'))&&exists(path.join(folder,'ssl_complete.json')),'Missing completed encoder metadata: '+folder);
            const config=read(path.join(folder,'config.json')),marker=read(path.join(folder,'ssl_complete.json')),s=config.signature;
            assert.deepEqual([s.source,s.model,s.selection_method,s.subset,s.ssl,config.identity.seed,s.protocol.ssl.subset_epochs],
                [group.source,'ResNet18',group.kind,group.subset,group.ssl,seed,group.epochs]);
            assert.deepEqual(s.code_hashes,hashes);assert.deepEqual(s.protocol.evaluation,reference);
            assert.ok(marker.complete&&marker.epochs===group.epochs);assert.deepEqual(marker.identity,config.identity);
            assert.ok(exists(path.join(folder,'encoder.pt'))||exists(path.join(folder,'training.pt')),'No saved final weights: '+folder);
            encoders.set(seed,{folder,config,marker});
        }
        for(const [target,labels] of group.cells){
            const cell={source:group.source,kind:group.kind,subset:group.subset,ssl:group.ssl,target,labels};
            const table=target===group.source?'CVPR_tables/table.tex':`CVPR_tables/${group.source.toLowerCase()}_downstream.tex`;
            const entry=rows(table,fs.readFileSync(path.join(repo,table),'utf8')).flatMap(r=>r.cells).find(c=>c.model==='ResNet18'&&key(c)===key(cell));
            assert.ok(entry,'Missing table cell: '+key(cell));
            if(entry.value!==''){skipped++;continue;}
            const name=`pp_${group.id}_${target}_l${labels}`;
            if(queue.has(name)){skipped++;continue;}
            const directory=`partial_probe_results/${group.id}/${target}_lbl${labels}`,samples=[],times=[];
            let missing=0;
            for(const [ssl_seed,probe_seed] of pairs(group.scheme)){
                const {folder,config,marker}=encoders.get(ssl_seed);
                const output=path.join(bench,group.output_group,tag+String(ssl_seed).padStart(3,'0'));
                const suffix=group.scheme==='four_ssl'&&probe_seed===ssl_seed?'':`_probe${probe_seed}`;
                const canonical=path.join(output,'evaluations',`${target}_lbl${labels}${suffix}.json`);
                const destination=exists(canonical)||group.scheme==='two_ssl'?canonical:
                    path.join(bench,directory,`ssl_seed_${ssl_seed}_probe_${probe_seed}.json`);
                const sample={ssl_seed,probe_seed,checkpoint_dir:path.relative(bench,folder),
                    config_sha256:hash(path.join(folder,'config.json')),encoder_sha256:marker.encoder_sha256,
                    result_file:path.relative(bench,destination)};
                if(exists(destination)){
                    const r=read(destination),e=r.expected;
                    assert.ok(r.complete&&r.encoder_unchanged&&r.encoder_sha256===marker.encoder_sha256);
                    assert.deepEqual(e.identity,config.identity);assert.deepEqual(e.evaluation_code_hashes,hashes);
                    assert.deepEqual([e.source,e.target,e.label_percentage,e.seed],[group.source,target,labels,probe_seed]);
                    assert.deepEqual(e.evaluation,reference);
                    assert.ok(Number.isFinite(r.seconds)&&r.seconds>0);times.push(r.seconds);reused++;
                }else missing++;
                samples.push(sample);
            }
            assert.ok(times.length,'Expected existing partial probes: '+key(cell));times.sort((a,b)=>a-b);
            const seconds=Math.ceil(missing*times[Math.ceil(times.length*.95)-1]+60);
            const job={name,group_id:group.id,cell,cell_key:key(cell),ssl_epochs:group.epochs,
                scheme:group.scheme,repeated_encoders:group.scheme!=='two_ssl',samples,
                summary_file:directory+'/summary.json',code_hashes:hashes,seconds,buffer_seconds:60};
            const file=path.join(batch,name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
            tsv+=[name,seconds,missing,file].join('\t')+'\n';jobs++;newProbes+=missing;
        }
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    console.log('PLAN:',JSON.stringify({jobs,new_probes:newProbes,reused_probes:reused,skipped_cells:skipped,ssl_training:false,buffer_seconds:60}));
}
module.exports={pairs};
if(require.main===module)main();
