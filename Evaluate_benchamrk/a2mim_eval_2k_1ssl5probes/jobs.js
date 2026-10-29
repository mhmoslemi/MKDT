// Fill only the A²MIM block; completed Barlow results are never modified here.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const here=__dirname,bench=path.dirname(here),root=path.dirname(bench);
const table=path.join(root,'CVPR_tables/ablations/ssl_methods_combined.tex');
const sources=['CIFAR10','CIFAR100'],accounts=['aip-boyuwang','aip-yiweilu'];
const targets=s=>[s,s==='CIFAR10'?'CIFAR100':'CIFAR10','TinyImageNet'];
const read=f=>JSON.parse(fs.readFileSync(f,'utf8'));
const hash=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const p95=a=>{assert.ok(a.length&&a.every(v=>Number.isFinite(v)&&v>0),'No matching runtime observations');return a.sort((a,b)=>a-b)[Math.ceil(.95*a.length)-1];};
const folder=u=>path.join(here,'results',u.source,'ResNet18',u.kind,'pt'+u.subset,'a2mim/ssl_seed_000');
function rows(text){
    const result=[];for(const [index,line]of text.split('\n').entries()){
        const m=line.match(/% sslcompare method=(\w+) kind=(\w+) pt=([\d.]+) lbl=([15])/);if(!m)continue;
        const cut=line.indexOf('\\\\ % sslcompare ');assert.ok(cut>=0,'Unexpected row terminator');
        const tokens=line.slice(0,cut).split('&');assert.equal(tokens.length,10);
        result.push({index,ssl:m[1],kind:m[2],subset:Number(m[3]),labels:Number(m[4]),tokens,comment:line.slice(cut+3)});
    }
    assert.equal(result.length,28);return result;
}
function replacement(text,s){
    assert.ok(s.complete&&s.ssl==='a2mim'&&s.model==='ResNet18'&&s.ssl_seed===0&&s.std_ddof===1&&s.completed_probes===5);
    assert.ok(sources.includes(s.source)&&targets(s.source).includes(s.target)&&[1,5].includes(s.labels));
    assert.ok(['full','random','mkdt','ours'].includes(s.kind));assert.equal(s.ssl_epochs,s.kind==='full'?300:2000);
    assert.deepEqual(s.samples.map(p=>p.probe_seed),[0,1,2,3,4]);assert.ok(s.samples.every(p=>Number.isFinite(p.accuracy_percent)&&p.accuracy_percent>=0&&p.accuracy_percent<=100));
    assert.match(s.tex_value,/^\\dsacc\{\d+\.\d{2}\}\{\d+\.\d{2}\}$/);
    const parsed=rows(text),row=parsed.find(r=>r.ssl==='a2mim'&&r.kind===s.kind&&r.subset===s.subset&&r.labels===s.labels);assert.ok(row);
    const col=4+sources.indexOf(s.source)*3+targets(s.source).indexOf(s.target),score=/\\(?:dsacc|bestacc)\{([\d.]+)\}\{([\d.]+)\}/;
    if(score.test(row.tokens[col]))row.tokens[col]=row.tokens[col].replace(score,s.tex_value);
    else row.tokens[col]=row.tokens[col].trim()+s.tex_value;
    const lines=text.split('\n');
    const peers=s.kind==='full'?[row]:parsed.filter(r=>r.ssl==='a2mim'&&r.subset===s.subset&&r.labels===s.labels);
    if(s.kind!=='full'){
        assert.equal(peers.length,3);const values=peers.map(r=>r.tokens[col].match(score));
        const best=values.every(Boolean)?Math.max(...values.map(m=>+m[1])):null;
        peers.forEach((r,i)=>r.tokens[col]=r.tokens[col].replace(/\\(?:dsacc|bestacc)(?=\{)/,best!==null&&Number(values[i][1])===best?'\\bestacc':'\\dsacc'));
    }
    for(const r of peers)lines[r.index]=r.tokens.join('&')+'\\\\ '+r.comment;
    const tag=`% A2MIM result: ${s.cell_key};`,note=tag+` SSL epochs=${s.ssl_epochs}; encoder seed=0; five 100-epoch frozen probes, seeds=0,1,2,3,4; mean and sample std; results=Evaluate_benchamrk/a2mim_eval_2k_1ssl5probes.`;
    const previous=lines.findIndex(l=>l.startsWith(tag));if(previous>=0)lines[previous]=note;else{if(lines.at(-1)==='')lines.pop();lines.push(note,'');}
    return lines.join('\n');
}
function units(){return sources.flatMap(source=>[{source,kind:'full',subset:100},...[.2,1].flatMap(subset=>['random','mkdt','ours'].map(kind=>({source,kind,subset})))]);}
function check(){
    const original=fs.readFileSync(table,'utf8'),before=rows(original);let count=0;
    for(const u of units())for(const target of targets(u.source))for(const labels of [1,5]){
        const s={...u,target,labels,complete:true,ssl:'a2mim',model:'ResNet18',ssl_seed:0,std_ddof:1,completed_probes:5,ssl_epochs:u.kind==='full'?300:2000,
            samples:[0,1,2,3,4].map(probe_seed=>({probe_seed,accuracy_percent:99.99})),tex_value:'\\dsacc{99.99}{0.00}',cell_key:[u.source,u.kind,u.subset,target,labels].join('|')};
        const after=rows(replacement(original,s)),col=4+sources.indexOf(u.source)*3+targets(u.source).indexOf(target);
        for(let i=0;i<before.length;i++)for(let j=0;j<10;j++){
            const clean=t=>t.replace(/\\bestacc(?=\{)/g,'\\dsacc'),r=before[i];
            if(r.ssl==='a2mim'&&r.kind===u.kind&&r.subset===u.subset&&r.labels===labels&&j===col)assert.ok(clean(after[i].tokens[j]).includes(s.tex_value));
            else assert.equal(clean(after[i].tokens[j]).trim(),clean(before[i].tokens[j]).trim(),'Unrelated table data changed');
        }
        count++;
    }
    assert.equal(count,84);assert.equal(units().length,14);console.log('PASS: 84 A2MIM cells / 420 probes; Barlow block and other values protected.');
}
function origin(u){
    if(u.kind==='full')return path.join(bench,'a2mim_teacher/checkpoints',u.source,'ResNet18/full/pt100/a2mim/seed_000/config.json');
    if(u.kind==='random')return path.join(bench,`cifar_small_simclr2k_1ssl5probes/results/${u.source}/ResNet18/random/pt${u.subset}/simclr/ssl_seed_000/config.json`);
    if(u.kind==='ours')return path.join(bench,`ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/${u.source}/ResNet18BN/pt${u.subset}/simclr/ssl_seed_000/config.json`);
    return path.join(bench,u.subset===.2?'mkdt_eval_results_simclr2k_2ssl3probes':u.source.toLowerCase()+'_main_results_simclr2k3k_barlow10k_2ssl3probes',u.source,'ResNet18/mkdt','pt'+u.subset,'barlowtwins/ssl_seed_000/config.json');
}
function plan(batch){
    check();batch=path.resolve(batch);assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const files=fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f).concat(['networks.py','benchmarks/downstream_data.py','Evaluate_benchamrk/protocol.json',
        'Evaluate_benchamrk/a2mim_teacher/model.py','Evaluate_benchamrk/a2mim_teacher/train.py','Evaluate_benchamrk/a2mim_teacher/sources.json',
        'Evaluate_benchamrk/a2mim_eval_2k_1ssl5probes/train_subset.py','Evaluate_benchamrk/a2mim_eval_2k_1ssl5probes/worker.py','Evaluate_benchamrk/a2mim_eval_2k_1ssl5probes/jobs.js','Evaluate_benchamrk/submit_a2mim_table_2k_1ssl5probes_vulcan.sh']);
    const code_hashes=Object.fromEntries(files.map(f=>[f,hash(path.join(root,f))]));
    const queue=new Map(fs.readFileSync(path.join(batch,'queue.tsv'),'utf8').trim().split('\n').filter(Boolean).map(l=>l.split('|').map(s=>s.trim())));
    const startup=read(path.join(bench,'cifar_small_simclr2k_1ssl5probes/batches/bundled_20261027T181030Z_YGcQKH/bundled_coverage.json')).startup_measurement.seconds;
    const parsed=rows(fs.readFileSync(table,'utf8'));
    const planned=[],timings=[];let sequence=0,skipped=0;
    for(const bare of units()){
        const cfg=origin(bare),old=read(cfg),u={...bare,origin_config:path.relative(root,cfg),origin_config_sha256:hash(cfg),code_hashes};
        const dir=folder(u),name=`a25_${u.source}_${u.kind}_p${String(u.subset).replace('.','p')}`;
        if(u.kind==='full'){
            const m=read(path.join(path.dirname(cfg),'ssl_complete.json'));assert.ok(m.complete&&m.epochs===300&&m.method==='a2mim');assert.ok(fs.statSync(path.join(path.dirname(cfg),'encoder.pt')).size>0);
        }else if(u.kind==='random'){
            const indices=old.identity.selection.indices;assert.equal(indices.length,50000*u.subset/100);assert.equal(new Set(indices).size,indices.length);
        }else{
            const p=path.resolve(old.signature.synthetic.path);assert.ok(p.startsWith(root+path.sep));assert.ok(fs.statSync(p).size>0);u.synthetic=path.relative(root,p);
            if(u.kind==='mkdt')assert.equal(hash(p),old.signature.synthetic.file_sha256,'MKDT image file differs from its previously evaluated input');
        }
        let sslDone=u.kind==='full';
        if(u.kind!=='full'){
            const account=accounts[sequence++%2],marker=path.join(dir,'ssl_complete.json');
            if(fs.existsSync(marker)){const m=read(marker);assert.ok(m.complete&&m.epochs===2000);assert.deepEqual(read(path.join(dir,'config.json')).signature.driver_code_hashes,code_hashes);sslDone=true;}
            if(!sslDone&&!queue.has(name+'_ssl')){
                const teacher=path.join(bench,'a2mim_teacher/checkpoints',u.source,'ResNet18/full/pt100/a2mim/seed_000/ssl.jsonl');
                const rate=p95(fs.readFileSync(teacher,'utf8').trim().split('\n').slice(-100).map(l=>JSON.parse(l).seconds));
                const seconds=Math.ceil(rate*(50000*u.subset/100)/50000*2000+startup+60);
                planned.push({name:name+'_ssl',stage:'ssl',account,seconds,parents:[],unit:u});timings.push({name:name+'_ssl',full_epoch_p95:rate,subset_count:50000*u.subset/100,seconds,estimate:'Scale measured A2MIM full-data time by image count; approximate.'});
            }
        }
        const cells=targets(u.source).flatMap(target=>[1,5].map(labels=>({target,labels})));
        for(let offset=0;offset<cells.length;offset+=4){
            const account=accounts[sequence++%2],bundleName=name+'_p'+offset/4,selected=[],observations=[];let seconds=startup+60,missing=0;
            for(const cell of cells.slice(offset,offset+4)){
                const summary=path.join(dir,`summary_${cell.target}_lbl${cell.labels}.json`);
                if(fs.existsSync(summary)){
                    const s=read(summary),row=parsed.find(r=>r.ssl==='a2mim'&&r.kind===u.kind&&r.subset===u.subset&&r.labels===cell.labels);
                    const col=4+sources.indexOf(u.source)*3+targets(u.source).indexOf(cell.target);
                    if(s.complete&&s.completed_probes===5&&row.tokens[col].replace(/\\bestacc(?=\{)/g,'\\dsacc').includes(s.tex_value)){skipped+=5;continue;}
                }
                selected.push(cell);let count=0;
                for(let seed=0;seed<5;seed++){
                    const f=path.join(dir,'evaluations',`${cell.target}_lbl${cell.labels}_probe${seed}.json`);
                    if(!fs.existsSync(f))count++;else{const r=read(f);assert.ok(r.complete&&r.encoder_unchanged);skipped++;}
                }
                const rates=[];for(let seed=0;seed<6;seed++){
                    const f=path.join(bench,`kmeans_eval_results_interactive_6seeds/${u.source}/ResNet18/kmeans/pt0.2/simclr/seed_${String(seed).padStart(3,'0')}/evaluations/${cell.target}_lbl${cell.labels}.json`);
                    if(fs.existsSync(f)){const r=read(f);if(r.complete&&r.expected.evaluation.epochs===100&&r.seconds>0)rates.push(r.seconds);}
                }
                const rate=p95(rates);seconds+=count*rate;missing+=count;observations.push({...cell,probe_seconds_p95:rate,samples:rates.length});
            }
            if(selected.length&&!queue.has(bundleName)){
                planned.push({name:bundleName,stage:'probe_bundle',account,seconds:Math.ceil(seconds),parents:sslDone?[]:[name+'_ssl'],unit:u,cells:selected,missing_probes:missing});
                timings.push({name:bundleName,seconds:Math.ceil(seconds),observations});
            }
        }
    }
    assert.equal(sequence,40);const known=new Set(queue.keys());let tsv='';
    for(const job of planned){for(const parent of job.parents)assert.ok(known.has(parent),'Missing encoder dependency');known.add(job.name);
        assert.ok(job.seconds>0&&job.seconds<3600);const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        tsv+=[job.name,job.stage,job.account,job.seconds,job.parents.join(',')||'-',file].join('\t')+'\n';}
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const coverage={expected_cells:84,expected_probes:420,full_encoders_reused:2,full_epochs:300,subset_epochs:2000,probe_epochs:100,probe_seeds:[0,1,2,3,4],
        jobs:planned.length,ssl_jobs:planned.filter(j=>j.stage==='ssl').length,probe_bundles:planned.filter(j=>j.stage==='probe_bundle').length,
        probes_to_run:planned.reduce((n,j)=>n+(j.missing_probes||0),0),completed_probes_skipped:skipped,
        accounts:Object.fromEntries(accounts.map(a=>[a,planned.filter(j=>j.account===a).length])),buffer_seconds:60,
        min_seconds:planned.length?Math.min(...planned.map(j=>j.seconds)):0,max_seconds:planned.length?Math.max(...planned.map(j=>j.seconds)):0,
        table:'CVPR_tables/ablations/ssl_methods_combined.tex',timings};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(coverage,null,2)+'\n');console.log(JSON.stringify({...coverage,timings:undefined},null,2));
}
module.exports={rows,replacement,units};
if(require.main===module){
    if(process.argv[2]==='--check')check();
    else if(process.argv[2]==='--update'){
        const file=path.resolve(process.argv[3]);assert.ok(file.startsWith(path.join(here,'results')+path.sep));
        const old=fs.readFileSync(table,'utf8'),next=replacement(old,read(file));if(next!==old){const temporary=table+'.'+process.pid+'.tmp';fs.writeFileSync(temporary,next);fs.chmodSync(temporary,fs.statSync(table).mode&0o777);fs.renameSync(temporary,table);}
        console.log('A2MIM TABLE UPDATED:',read(file).cell_key);
    }else plan(process.argv[2]);
}
