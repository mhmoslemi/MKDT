// Probe existing 2k encoders; only the unified downstream table is replaced.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const table=path.join(root,'CVPR_tables/cifar_downstream.tex');
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const sources=['CIFAR10','CIFAR100','TinyImageNet'];
const targets=source=>sources.filter(s=>s!==source).concat(['Aircraft','CUB2011','Dogs']);
const eligible=c=>sources.includes(c.source)&&[.2,1].includes(c.subset)&&['random','kmeans'].includes(c.kind)&&[1,5].includes(c.labels)&&targets(c.source).includes(c.target)&&!(c.source==='CIFAR10'&&c.subset===1&&c.labels===5);
function rows(text){
    let labels,subset;const result=[];
    for(const [lineIndex,line]of text.split('\n').entries()){
        const label=line.match(/\\multirow\{9\}\{\*\}\{([15])\\%\}/);if(label)labels=Number(label[1]);
        const tokens=line.split('&');if(tokens.length!==18||!labels)continue;
        const size=tokens[1].match(/\\multirow\{4\}\{\*\}\{(0\.2|1)\}/);if(size)subset=Number(size[1]);
        const method={'Random':'random','K-means':'kmeans','MKDT':'mkdt','Ours':'ours','Full Data':'full'}[tokens[2].trim()];
        if(!method)continue;
        result.push({labels,subset:method==='full'?100:subset,kind:method,lineIndex,tokens});
    }
    assert.equal(result.length,18,'Unified downstream layout changed');
    assert.equal(new Set(result.map(r=>[r.labels,r.subset,r.kind].join('|'))).size,18);
    return result;
}
function cells(text){return rows(text).filter(r=>['random','kmeans'].includes(r.kind)).flatMap(r=>sources.flatMap(source=>targets(source).map(target=>({source,target,subset:r.subset,kind:r.kind,labels:r.labels})))).filter(eligible);}
function folder(c){return path.join(here,'results',c.source,'ResNet18',c.kind,'pt'+c.subset,'simclr/ssl_seed_000');}
function encoder(c){return path.join(bench,c.source==='TinyImageNet'?'subset_simclr2k_1ssl5probes':'cifar_small_simclr2k_1ssl5probes','results',c.source,'ResNet18',c.kind,'pt'+c.subset,'simclr/ssl_seed_000');}
const probeFile=(c,seed)=>path.join(folder(c),'evaluations',`${c.target}_lbl${c.labels}_probe${seed}.json`);
function valid(result,c,seed,config,marker,shared){
    const e=result.expected;assert.ok(result.complete&&result.encoder_unchanged);
    assert.equal(result.encoder_sha256,marker.encoder_sha256);assert.deepEqual(e.identity,config.identity);
    assert.deepEqual([e.source,e.target,e.label_percentage,e.seed],[c.source,c.target,c.labels,seed]);
    assert.deepEqual(e.evaluation,config.signature.protocol.evaluation);assert.deepEqual(e.evaluation_code_hashes,shared);
    assert.ok(Number.isFinite(result.accuracy_percent)&&result.accuracy_percent>=0&&result.accuracy_percent<=100);
}
function replacement(text,s){
    assert.ok(eligible(s)&&s.complete&&s.ssl_epochs===2000&&s.linear_epochs===100&&s.std_ddof===1&&s.ssl_seed===0);
    assert.deepEqual(s.samples.map(p=>p.probe_seed),[0,1,2,3,4]);
    assert.match(s.tex_value,/^\\dsacc\{\d+\.\d\}\{\d+\.\d\}$/);
    const parsed=rows(text),row=parsed.find(r=>r.labels===s.labels&&r.subset===s.subset&&r.kind===s.kind);
    const col=3+sources.indexOf(s.source)*5+targets(s.source).indexOf(s.target),lines=text.split('\n');
    const score=/\\(?:dsacc|bestacc)\{([\d.]+)\}\{([\d.]+)\}/;
    assert.ok(score.test(row.tokens[col]));row.tokens[col]=row.tokens[col].replace(score,s.tex_value);
    const peers=parsed.filter(r=>r.labels===s.labels&&r.subset===s.subset&&r.kind!=='full');assert.equal(peers.length,4);
    const means=peers.map(r=>{const m=r.tokens[col].match(score);assert.ok(m);return +m[1];}),best=Math.max(...means);
    peers.forEach((r,i)=>{r.tokens[col]=r.tokens[col].replace(/\\(?:dsacc|bestacc)(?=\{)/,means[i]===best?'\\bestacc':'\\dsacc');lines[r.lineIndex]=r.tokens.join('&');});
    const tag=`% Downstream SimCLR 2k replacement: ${s.cell_key};`;
    const note=tag+' encoder seed=0; probe seeds=0,1,2,3,4; 100-epoch frozen probes; results=Evaluate_benchamrk/downstream_simclr2k_1ssl5probes.';
    const old=lines.findIndex(l=>l.startsWith(tag));if(old>=0)lines[old]=note;else{if(lines.at(-1)==='')lines.pop();lines.push(note,'');}
    return lines.join('\n');
}
function check(){
    const text=fs.readFileSync(table,'utf8'),before=rows(text),all=cells(text);assert.equal(all.length,110);
    assert.equal(new Set(all.map(c=>JSON.stringify(c))).size,110);
    for(const c of all){
        const s={...c,complete:true,ssl_epochs:2000,linear_epochs:100,std_ddof:1,ssl_seed:0,samples:[0,1,2,3,4].map(probe_seed=>({probe_seed})),tex_value:'\\dsacc{99.9}{0.0}',cell_key:JSON.stringify(c)};
        const after=rows(replacement(text,s)),col=3+sources.indexOf(c.source)*5+targets(c.source).indexOf(c.target);
        for(let i=0;i<before.length;i++)for(let j=0;j<18;j++){
            const clean=t=>t.replace(/\\bestacc(?=\{)/g,'\\dsacc');
            if(before[i].kind===c.kind&&before[i].subset===c.subset&&before[i].labels===c.labels&&j===col)assert.ok(clean(after[i].tokens[j]).includes(s.tex_value));
            else assert.equal(clean(after[i].tokens[j]),clean(before[i].tokens[j]),'Unrelated table value changed');
        }
    }
    assert.ok(!eligible({source:'CIFAR10',subset:1,kind:'random',labels:5,target:'Dogs'}));
    assert.ok(!eligible({source:'CIFAR10',subset:1,kind:'kmeans',labels:5,target:'Dogs'}));
    assert.ok(!eligible({source:'CIFAR100',subset:2,kind:'random',labels:1,target:'Dogs'}));
    console.log('PASS: 110 downstream cells / 550 probes; both old-result exceptions and all unrelated values protected.');
}
function plan(batch){
    batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const sharedFiles=fs.readdirSync(bench).filter(f=>f.endsWith('.py')).sort().map(f=>'Evaluate_benchamrk/'+f).concat(['networks.py','benchmarks/downstream_data.py']);
    const shared=Object.fromEntries(sharedFiles.map(f=>[f,hash(path.join(root,f))]));
    const code_hashes=Object.fromEntries([...sharedFiles,'Evaluate_benchamrk/downstream_simclr2k_1ssl5probes/jobs.js','Evaluate_benchamrk/downstream_simclr2k_1ssl5probes/worker.py','Evaluate_benchamrk/submit_downstream_simclr2k_5probes_wang_vulcan.sh'].map(f=>[f,hash(path.join(root,f))]));
    const text=fs.readFileSync(table,'utf8'),parsed=rows(text),all=cells(text);
    assert.equal(all.length,110);
    const groups=new Map(),timings=[],blocked=[];let probes=0,done=0;
    const startup=read(path.join(bench,'cifar_small_simclr2k_1ssl5probes/batches/bundled_20261027T181030Z_YGcQKH/bundled_coverage.json')).startup_measurement.seconds;
    for(const c of all){
        const dir=encoder(c);if(!['config.json','ssl_complete.json','encoder.pt'].every(f=>fs.existsSync(path.join(dir,f)))){blocked.push(c);continue;}
        const config=read(path.join(dir,'config.json')),marker=read(path.join(dir,'ssl_complete.json')),sig=config.signature;
        assert.ok(marker.complete&&marker.epochs===2000);assert.deepEqual(marker.identity,config.identity);assert.deepEqual(sig.code_hashes,shared);
        assert.deepEqual([sig.source,sig.model,sig.selection_method,sig.subset,sig.ssl,config.identity.seed],[c.source,'ResNet18',c.kind,c.subset,'simclr',0]);
        assert.equal(sig.protocol.ssl.subset_epochs,2000);assert.equal(sig.protocol.evaluation.epochs,100);
        const missing=[];for(let seed=0;seed<5;seed++){const f=probeFile(c,seed);if(!fs.existsSync(f))missing.push(seed);else{valid(read(f),c,seed,config,marker,shared);done++;}}
        if(!missing.length){
            const summary=path.join(folder(c),`summary_${c.target}_lbl${c.labels}.json`);
            if(fs.existsSync(summary)){
                const s=read(summary),row=parsed.find(r=>r.labels===c.labels&&r.subset===c.subset&&r.kind===c.kind);
                const col=3+sources.indexOf(c.source)*5+targets(c.source).indexOf(c.target);
                if(s.complete&&row.tokens[col].replace(/\\bestacc(?=\{)/g,'\\dsacc').includes(s.tex_value))continue;
            }
        }
        const rates=[];for(let seed=0;seed<6;seed++){
            const f=path.join(bench,`kmeans_eval_results_interactive_6seeds/${c.source}/ResNet18/kmeans/pt${c.subset}/simclr/seed_${String(seed).padStart(3,'0')}/evaluations/${c.target}_lbl${c.labels}.json`);
            if(!fs.existsSync(f))continue;const r=read(f);if(r.complete&&r.seconds>0&&r.expected.evaluation.epochs===100)rates.push(r.seconds);
        }
        assert.ok(rates.length,'No matching completed downstream runtime: '+JSON.stringify(c));rates.sort((a,b)=>a-b);const seconds=rates[Math.ceil(.95*rates.length)-1];
        const cell={...c,checkpoint_dir:path.relative(bench,dir),output_dir:path.relative(bench,folder(c)),config_sha256:hash(path.join(dir,'config.json')),marker_sha256:hash(path.join(dir,'ssl_complete.json')),missing_probe_seeds:missing,probe_seconds_p95:seconds};
        const key=[c.source,c.kind,c.subset].join('|');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(cell);probes+=missing.length;
        timings.push({...c,seconds,samples:rates.length});
    }
    let tsv='',jobs=0;const queue=new Set(fs.readFileSync(path.join(batch,'queue.txt'),'utf8').trim().split('\n').map(s=>s.trim()));
    for(const [key,group]of groups){
        // Stable full-cell positions keep reruns from duplicating a still-queued bundle.
        const positions=all.filter(c=>[c.source,c.kind,c.subset].join('|')===key);
        for(let offset=0;offset<positions.length;offset+=4){
            const chunk=positions.slice(offset,offset+4),selected=group.filter(c=>chunk.some(p=>p.target===c.target&&p.labels===c.labels));if(!selected.length)continue;
            const [source,kind,size]=key.split('|'),name=`d2k5_${source}_${kind}_p${size.replace('.','p')}_b${offset/4}`;
            if(queue.has(name))continue;
            const seconds=Math.ceil(selected.reduce((n,c)=>n+c.missing_probe_seeds.length*(c.probe_seconds_p95+startup),0)+60);
            const manifest=path.join(batch,name+'.json');fs.writeFileSync(manifest,JSON.stringify({name,stage:'downstream_probe_bundle',cells:selected,seconds,buffer_seconds:60,code_hashes},null,2)+'\n',{flag:'wx'});
            tsv+=[name,seconds,selected.reduce((n,c)=>n+c.missing_probe_seeds.length,0),manifest].join('\t')+'\n';jobs++;
        }
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify({expected_cells:110,expected_probes:550,completed_probes:done,missing_probes:probes,jobs,blocked,account:'aip-boyuwang',ssl_jobs:0,probe_epochs:100,maximum_probes_per_bundle:20,buffer_seconds:60,timings},null,2)+'\n');
    console.log(`PLAN: ${jobs} probe-only jobs; ${probes} missing probes; ${done} already done; ${blocked.length} cells blocked by missing encoders.`);
}
module.exports={rows,cells,eligible,replacement};
if(require.main===module){
    if(process.argv[2]==='--check')check();
    else if(process.argv[2]==='--update'){
        const file=path.resolve(process.argv[3]);assert.ok(file.startsWith(path.join(here,'results')+path.sep));
        const old=fs.readFileSync(table,'utf8'),next=replacement(old,read(file));if(next!==old){const temporary=table+'.'+process.pid+'.tmp';fs.writeFileSync(temporary,next);fs.chmodSync(temporary,fs.statSync(table).mode&0o777);fs.renameSync(temporary,table);}
        console.log('DOWNSTREAM TABLE UPDATED:',read(file).cell_key);
    }else plan(process.argv[2]);
}
