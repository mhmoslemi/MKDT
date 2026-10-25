// Text/JSON-only planning on the login node; tensor loading happens in GPU jobs.
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const crypto = require('crypto');
const here = __dirname, bench = path.dirname(here), repo = path.dirname(bench);
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const exists = file => fs.existsSync(file) && fs.statSync(file).size > 0;
const kinds = {'Random Subset':'random', 'K-means Subset':'kmeans', MKDT:'mkdt', Ours:'ours'};
const sources = ['CIFAR10', 'CIFAR100', 'TinyImageNet'];
const key = c => [c.source,c.kind,c.subset,c.ssl,c.labels].join('|');

function parseTable(text) {
    let labels, subset;
    const cells = [];
    for (const row of text.replace(/(?<!\\)%[^\n]*/g, '').split('\\\\')) {
        let m = row.match(/\\multirow\{14\}\{\*\}\{([15])\\%\}/);
        if (m) labels = Number(m[1]);
        m = row.match(/\\multirow\{4\}\{\*\}\{([.\d]+)\\%\}/);
        if (m) subset = Number(m[1]);
        const method = row.match(/&\s*(Random Subset|K-means Subset|MKDT|Ours)\s*&/);
        if (!method) continue;
        assert.ok([1,5].includes(labels) && [0.2,1,2].includes(subset));
        const values = row.slice(method.index + method[0].length).split('&').flatMap(value => {
            const span = value.trim().match(/^\\multicolumn\{(\d+)\}\{c\}\{\}$/);
            return span ? Array(Number(span[1])).fill('') : [value.trim()];
        });
        assert.equal(values.length, 6, 'Malformed combined main-table row');
        values.forEach((value,index) => cells.push({source:sources[Math.floor(index/2)],
            kind:kinds[method[1]], subset, ssl:index%2 ? 'barlowtwins':'simclr', labels,
            status:!value ? 'blank':value==='--' ? 'omitted':'filled'}));
    }
    assert.equal(cells.length, 144);
    assert.equal(new Set(cells.map(key)).size, cells.length);
    return cells;
}

function eligible(cell) {
    return cell.status === 'blank' && !(cell.subset === 2 && cell.ssl === 'barlowtwins')
        && !(cell.kind === 'mkdt' && cell.subset === 0.2 && cell.source !== 'TinyImageNet');
}

function unitFor(cell, seed) {
    // The other datasets currently have no eligible blanks; do not revive unrelated runs.
    assert.equal(cell.source, 'TinyImageNet', 'New CIFAR blank: inspect its saved schedule before scheduling');
    assert.ok(['random','ours','mkdt'].includes(cell.kind));
    const {source,kind,subset,ssl} = cell;
    const epochs = kind === 'random' ? ({0.2:6000,1:5000,2:4000})[subset]
        : ssl === 'simclr' ? 2000:10000;
    const relative = kind === 'ours' ? `${source}/ResNet18BN/pt${subset}/${ssl}/ssl_seed_${String(seed).padStart(3,'0')}`
        : `${source}/ResNet18/${kind}/pt${subset}/${ssl}/${kind==='random'?'seed':'ssl_seed'}_${String(seed).padStart(3,'0')}`;
    const root = kind === 'random' ? 'random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k'
        : kind === 'ours' ? 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main':'mkdt_eval_results_simclr2k_2ssl3probes';
    return {source,kind,subset,ssl,seed,epochs,model:'ResNet18',scope:'main',parent_dir:null,
        checkpoint_dir:`${root}/${relative}`,
        output_dir:`${kind==='random'?'random_eval_results_interactive_10seeds':root}/${relative}`};
}

function logRows(file) {
    if (!exists(file)) return [];
    const lines = fs.readFileSync(file,'utf8').split('\n');
    return lines.flatMap((line,index) => {
        if (!line.trim()) return [];
        try { return [JSON.parse(line)]; }
        catch(error) { if(index===lines.length-1) return []; throw error; }
    });
}

function main() {
    const batch = path.resolve(process.argv[2]), syntheticRoot = path.resolve(process.argv[3]);
    assert.ok(batch.startsWith(path.join(here,'batches')+path.sep));
    const active = new Set((process.argv[4] || '').split('\n').filter(Boolean));
    const reference = read(path.join(bench,'protocol.json'));
    const hashes = Object.fromEntries([...fs.readdirSync(bench).filter(f=>f.endsWith('.py')).map(f=>'Evaluate_benchamrk/'+f),
        'networks.py','benchmarks/downstream_data.py'].map(f=>[f,crypto.createHash('sha256')
            .update(fs.readFileSync(path.join(repo,f))).digest('hex')]));
    const cells = parseTable(fs.readFileSync(path.join(repo,'CVPR_tables/table.tex'),'utf8')).filter(eligible);
    const units = new Map();
    for (const cell of cells) for(let seed=0;seed<(cell.kind==='random'?10:2);seed++) {
        const unit = unitFor(cell,seed), id = [unit.kind,unit.subset,unit.ssl,seed].join('|');
        if (!units.has(id)) units.set(id,{...unit,labels:[]});
        units.get(id).labels.push(cell.labels);
    }
    function configFor(unit) {
        const file = path.join(bench,unit.checkpoint_dir,'config.json');
        if (!exists(file)) return null;
        const config = read(file), s = config.signature;
        assert.deepEqual([s.source,s.model,s.selection_method,s.subset,s.ssl,config.identity.seed],
            [unit.source,unit.model,unit.kind,unit.subset,unit.ssl,unit.seed],file);
        const protocol = structuredClone(reference);
        protocol.ssl.subset_epochs = unit.epochs;
        if(unit.kind!=='random') {protocol.ssl.checkpoint_every_epochs=500;protocol.subset_seeds=[0,1];}
        assert.deepEqual(s.protocol,protocol,file);
        assert.deepEqual(s.code_hashes,hashes,file);
        const marker = path.join(bench,unit.checkpoint_dir,'ssl_complete.json');
        if(exists(marker)) {
            const m = read(marker);
            assert.ok(m.complete && m.epochs===unit.epochs,marker);
            assert.deepEqual(m.identity,config.identity,marker);
        }
        return config;
    }
    function completeProbe(unit, labels, probe, config) {
        const suffix = unit.kind==='random'?'':`_probe${probe}`;
        const file = path.join(bench,unit.output_dir,'evaluations',`${unit.source}_lbl${labels}${suffix}.json`);
        if(!exists(file)) return false;
        assert.ok(config,'Probe exists without its SSL config: '+file);
        const r=read(file), e=r.expected, marker=read(path.join(bench,unit.checkpoint_dir,'ssl_complete.json'));
        assert.ok(r.complete && r.encoder_unchanged && r.encoder_sha256===marker.encoder_sha256,file);
        assert.ok(Number.isFinite(r.accuracy_percent)&&r.accuracy_percent>=0&&r.accuracy_percent<=100,file);
        assert.deepEqual(e.identity,config.identity,file);
        assert.deepEqual([e.source,e.target,e.label_percentage,e.seed],[unit.source,unit.source,labels,probe],file);
        assert.deepEqual(e.evaluation,reference.evaluation,file);
        assert.deepEqual(e.evaluation_code_hashes,hashes,file);
        return true;
    }
    const probeTimes = new Map(), sslTimes = new Map();
    const p95 = values => {
        assert.ok(values.length,'No measured runtime for this job');
        return values.sort((a,b)=>a-b)[Math.ceil(values.length*.95)-1];
    };
    for(const kind of ['random','ours','mkdt','kmeans']) for(const subset of [0.2,1,2])
        for(const ssl of ['simclr','barlowtwins']) for(let seed=0;seed<(kind==='random'?10:kind==='kmeans'?6:2);seed++) {
            const stem = `TinyImageNet/ResNet18/${kind}/pt${subset}/${ssl}/seed_${String(seed).padStart(3,'0')}`;
            const dirs = kind==='kmeans' ? [path.join(bench,'kmeans_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k_6seeds',stem)]
                : [path.join(bench,unitFor({source:'TinyImageNet',kind,subset,ssl},seed).checkpoint_dir)];
            for(const dir of dirs) {
                const values=logRows(path.join(dir,'ssl.jsonl')).slice(-200).filter(r=>Number.isFinite(r.seconds)&&r.seconds>0).map(r=>r.seconds);
                sslTimes.set([kind,subset,ssl].join('|'),[...(sslTimes.get([kind,subset,ssl].join('|'))||[]),...values]);
            }
            const output = kind==='kmeans' ? path.join(bench,'kmeans_eval_results_interactive_6seeds',stem)
                : path.join(bench,unitFor({source:'TinyImageNet',kind,subset,ssl},seed).output_dir);
            const evaluation = path.join(output,'evaluations');
            if(!fs.existsSync(evaluation)) continue;
            for(const file of fs.readdirSync(evaluation).filter(f=>/^TinyImageNet_lbl[15](?:_probe[012])?\.json$/.test(f))) {
                const r=read(path.join(evaluation,file));
                if(r.complete && Number.isFinite(r.seconds) && r.seconds>0) {
                    const label=r.expected.label_percentage;
                    probeTimes.set(label,[...(probeTimes.get(label)||[]),r.seconds]);
                }
            }
        }
    let tsv='', reused=0, queued=0, missingProbes=0;
    const jobs=[];
    for(const unit of units.values()) {
        const config=configFor(unit), probes=[];
        for(const labels of unit.labels) for(const probe of unit.kind==='random'?[unit.seed]:[0,1,2]) {
            if(completeProbe(unit,labels,probe,config)) reused++; else probes.push({labels,probe});
        }
        if(!probes.length) continue;
        const name=`mainlu_${unit.kind}_${unit.source}_p${String(unit.subset).replace('.','p')}_${unit.ssl}_s${unit.seed}`;
        if(active.has(name)) {queued++;continue;}
        const folder=path.join(bench,unit.checkpoint_dir);
        let epoch=0;
        if(exists(path.join(folder,'encoder.pt'))) epoch=unit.epochs;
        else if(exists(path.join(folder,'training.pt'))) {
            const rows=logRows(path.join(folder,'ssl.jsonl'));
            const interval=config.signature.protocol.ssl.checkpoint_every_epochs;
            epoch=rows.length?Math.floor(rows.at(-1).epoch/interval)*interval:0;
        }
        assert.ok(epoch>=0&&epoch<=unit.epochs);
        if(unit.kind!=='random') {
            const synthetic = unit.kind==='ours' ? path.join(repo,'OT-SSL-DD/ours_syntehtic',unit.source,
                'ResNet18BN',`pt${unit.subset}`,`res_IIC_${unit.source}_ResNet18BN_${unit.subset}percent_final.pt`)
                : path.join(syntheticRoot,'ConvNet',unit.source,`Tiny_${unit.subset}pct/images_10000.pt`);
            assert.ok(exists(synthetic),'Missing synthetic data: '+synthetic);
            if(unit.kind==='mkdt'&&unit.subset===0.2) {
                const marker=read(path.join(path.dirname(synthetic),'complete_0.2.json'));
                assert.ok(marker.complete&&marker.count===200&&marker.iterations===10000);
                if(config) assert.equal(config.signature.synthetic.file_sha256,marker.files['images_10000.pt']);
            }
        }
        let rates=sslTimes.get([unit.kind,unit.subset,unit.ssl].join('|'))||[];
        if(!rates.length) rates=['random','kmeans','ours','mkdt'].flatMap(k=>sslTimes.get([k,unit.subset,unit.ssl].join('|'))||[]);
        const sslSeconds=epoch===unit.epochs?0:(unit.epochs-epoch)*p95(rates);
        const seconds=Math.ceil(sslSeconds+probes.reduce((sum,p)=>sum+p95(probeTimes.get(p.labels)||[]),0)+60);
        const job={name,unit,probes,resume_epoch_estimate:epoch,seconds,buffer_seconds:60};
        const file=path.join(batch,name+'.json');
        fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
        tsv+=[name,seconds,epoch,probes.length,file].join('\t')+'\n';
        missingProbes+=probes.length;jobs.push(job);
    }
    fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv,{flag:'wx'});
    const summary={eligible_blank_cells:cells.length,jobs:jobs.length,missing_probes:missingProbes,
        reused_probes:reused,queued_seeds_skipped:queued,buffer_seconds:60,
        cells,job_times:jobs.map(j=>({name:j.name,seconds:j.seconds,resume_epoch_estimate:j.resume_epoch_estimate}))};
    fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify(summary,null,2)+'\n',{flag:'wx'});
    console.log('PLAN:',JSON.stringify({...summary,cells:undefined,job_times:undefined}));
}

module.exports={parseTable,eligible,unitFor};
if(require.main===module) main();
