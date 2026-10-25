// JSON/text-only planning: no Python imports or tensor reads on the login node.
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const crypto = require('crypto');
const here = __dirname, bench = path.resolve(here, '..'), repo = path.dirname(bench);
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const exists = file => fs.existsSync(file) && fs.statSync(file).size > 0;
function logRows(file) {
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  return lines.map((line, i) => {
    if (!line.trim()) return null;
    try { return JSON.parse(line); } catch (error) { if (i === lines.length - 1) return null; throw error; }
  }).filter(Boolean);
}
const key = c => [c.source, c.kind, c.subset, c.ssl, c.target, c.labels].join('|');
const targets = {
  CIFAR10: ['CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers', 'TinyImageNet'],
  CIFAR100: ['CIFAR10', 'TinyImageNet', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'],
  TinyImageNet: ['CIFAR10', 'CIFAR100', 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'],
};
const tables = {CIFAR10: 'cifar10_downstream.tex', CIFAR100: 'cifar100_downstream.tex', TinyImageNet: 'tinyimagenet_simclr.tex'};
const kinds = {'Random Subset': 'random', 'K-means Subset': 'kmeans', MKDT: 'mkdt', Ours: 'ours'};

function parseTable(source, text) {
  let labels = null, subset = null;
  const cells = [];
  for (const row of text.replace(/(?<!\\)%[^\n]*/g, '').split('\\\\')) {
    let m = row.match(/\\multirow\{14\}\{\*\}\{([15])\\%\}/);
    if (m) labels = Number(m[1]);
    m = row.match(/\\multirow\{4\}\{\*\}\{([.\d]+)\\%\}/);
    if (m) subset = Number(m[1]);
    const method = row.match(/&\s*(Random Subset|K-means Subset|MKDT|Ours)\s*&/);
    if (!method) continue;
    assert.ok([1, 5].includes(labels) && [0.2, 1, 2].includes(subset));
    const columns = source === 'TinyImageNet' ? 6 : 12;
    let values = row.slice(method.index + method[0].length).split('&').map(x => x.trim());
    if (values.length === 1 && /\\multicolumn\{12\}\{c\}\{\}/.test(values[0])) values = Array(columns).fill('');
    assert.equal(values.length, columns, `${source}: malformed ${method[1]} row`);
    values.forEach((value, index) => cells.push({source, kind: kinds[method[1]], subset,
      ssl: source === 'TinyImageNet' || index % 2 === 0 ? 'simclr' : 'barlowtwins',
      target: targets[source][source === 'TinyImageNet' ? index : Math.floor(index / 2)], labels,
      status: !value ? 'blank' : value === '--' ? 'omitted' : 'filled'}));
  }
  assert.equal(cells.length, source === 'TinyImageNet' ? 144 : 288);
  return cells;
}

function unitFor(cell, seed) {
  const {source, kind, subset, ssl} = cell;
  assert.ok(['ours', 'mkdt'].includes(kind), 'No unstarted Random/k-means cells were expected; inspect their missing encoders explicitly.');
  const oneK = kind === 'ours' && subset === 2 && ssl === 'simclr';
  const epochs = ssl === 'barlowtwins' ? 10000 : oneK ? 1000 : 2000;
  const relative = kind === 'ours' ? `${source}/ResNet18BN/pt${subset}/${ssl}/ssl_seed_${String(seed).padStart(3, '0')}`
    : `${source}/ResNet18/mkdt/pt${subset}/${ssl}/ssl_seed_${String(seed).padStart(3, '0')}`;
  const checkpointRoot = kind === 'mkdt' ? 'mkdt_eval_results_simclr2k_2ssl3probes'
    : oneK ? 'ours_eval_results_simclr1k_2ssl3probes_main' : 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main';
  const outputRoot = kind === 'mkdt' ? checkpointRoot
    : oneK ? 'ours_eval_results_simclr1k_2ssl3probes_downstream' : 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_downstream';
  return {source, kind, subset, ssl, seed, epochs, model: 'ResNet18',
    id: `${kind}_${source}_p${String(subset).replace('.', 'p')}_${ssl}_s${seed}`,
    checkpoint_dir: `${checkpointRoot}/${relative}`, output_dir: `${outputRoot}/${relative}`};
}

function progress(cell, hashes) {
  const expected = ['ours', 'mkdt'].includes(cell.kind) ? 6 : cell.kind === 'random' ? 10 : 6;
  let completed = 0, incomplete_files = 0;
  for (let seed = 0; seed < (expected === 6 && ['ours', 'mkdt'].includes(cell.kind) ? 2 : expected); seed++) {
    let dir, configDir;
    if (['ours', 'mkdt'].includes(cell.kind)) {
      const unit = unitFor(cell, seed);
      dir = path.join(bench, unit.output_dir); configDir = path.join(bench, unit.checkpoint_dir);
    } else {
      dir = path.join(bench, `${cell.kind}_eval_results_interactive_${cell.kind === 'random' ? '10' : '6'}seeds`,
        cell.source, 'ResNet18', cell.kind, `pt${cell.subset}`, cell.ssl, `seed_${String(seed).padStart(3, '0')}`);
      configDir = dir;
    }
    for (const probe of ['ours', 'mkdt'].includes(cell.kind) ? [0, 1, 2] : [seed]) {
      const suffix = ['ours', 'mkdt'].includes(cell.kind) ? `_probe${probe}` : '';
      const file = path.join(dir, 'evaluations', `${cell.target}_lbl${cell.labels}${suffix}.json`);
      if (!exists(file)) continue;
      const result = read(file), e = result.expected;
      if (!result.complete) { incomplete_files++; continue; }
      const config = read(path.join(exists(path.join(configDir, 'config.json')) ? configDir : dir, 'config.json'));
      assert.ok(result.encoder_unchanged && Number.isFinite(result.accuracy_percent)
        && result.accuracy_percent >= 0 && result.accuracy_percent <= 100, file);
      assert.deepEqual(e.identity, config.identity);
      assert.deepEqual([e.source, e.target, e.label_percentage, e.seed], [cell.source, cell.target, cell.labels, probe]);
      assert.deepEqual(e.evaluation_code_hashes, hashes);
      assert.equal(e.evaluation.epochs, 100);
      if (['ours', 'mkdt'].includes(cell.kind)) assert.equal(config.signature.protocol.ssl.subset_epochs, unitFor(cell, seed).epochs);
      if (['ours', 'mkdt'].includes(cell.kind)) {
        const marker = read(path.join(configDir, 'ssl_complete.json'));
        assert.ok(marker.complete && result.encoder_sha256 === marker.encoder_sha256, file);
        assert.deepEqual(marker.identity, config.identity);
      }
      completed++;
    }
  }
  assert.ok(completed <= expected);
  return {completed, expected, incomplete_files};
}

function classify(cell, count, approved, skip02) {
  if (cell.status !== 'blank' || (cell.subset === 2 && cell.ssl === 'barlowtwins')) return 'table_excluded';
  if (skip02 && cell.kind === 'mkdt' && cell.subset === 0.2) return 'mkdt02_excluded';
  if (count.completed === count.expected) return 'already_complete';
  if ((count.completed > 0 || count.incomplete_files > 0) && !approved.has(key(cell))) return 'partial_excluded';
  return 'selected';
}

function main() {
  const batch = path.resolve(process.argv[2]);
  assert.ok(batch.startsWith(path.join(here, 'batches') + path.sep));
  const queue = read(process.argv[3]);
  const dry = process.argv.includes('--dry-run'), skip02 = process.argv.includes('--skip-mkdt02');
  const hashes = Object.fromEntries([...fs.readdirSync(bench).filter(x => x.endsWith('.py')).map(x => 'Evaluate_benchamrk/' + x),
    'networks.py', 'benchmarks/downstream_data.py'].sort().map(f => [f,
    crypto.createHash('sha256').update(fs.readFileSync(path.join(repo, f))).digest('hex')]));
  const cells = Object.entries(tables).flatMap(([s, f]) => parseTable(s, fs.readFileSync(path.join(repo, 'CVPR_tables', f), 'utf8')));
  assert.equal(new Set(cells.map(key)).size, cells.length);
  const approvedFile = path.join(here, 'approved_cells.json');
  const approved = new Set(exists(approvedFile) ? read(approvedFile) : []);
  const report = [], selected = [];
  for (const cell of cells) {
    if (cell.status !== 'blank' || (cell.subset === 2 && cell.ssl === 'barlowtwins')) continue;
    const count = progress(cell, hashes), status = classify(cell, count, approved, skip02);
    report.push({...cell, ...count, status});
    if (status === 'selected') { selected.push(cell); approved.add(key(cell)); }
  }
  const queuedProbes = new Set();
  const priorBatches = fs.existsSync(path.join(here, 'batches')) ? fs.readdirSync(path.join(here, 'batches')).sort().reverse() : [];
  for (const name of Object.keys(queue).filter(n => n.startsWith('dfprobe_'))) {
    const file = priorBatches.map(b => path.join(here, 'batches', b, name + '.json')).find(exists);
    assert.ok(file, 'Cannot identify already queued downstream job: ' + name);
    for (const b of read(file).bundles) for (const p of b.probes) queuedProbes.add([b.unit.id, key(b.cell), p].join('|'));
  }
  const probeRates = new Map(), sslRates = new Map();
  const units = new Map();
  for (const cell of selected) for (const seed of [0, 1]) {
    const u = unitFor(cell, seed); units.set(u.id, u);
  }
  const roots = ['random_ssl_checkpoints_cifar10k8k6k_tiny6k5k4k', 'mkdt_eval_results_simclr2k_2ssl3probes',
    'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main', 'ours_eval_results_simclr1k_2ssl3probes_main'];
  for (const source of Object.keys(tables)) for (const subset of [0.2, 1, 2]) for (const ssl of ['simclr', 'barlowtwins']) {
    for (const root of roots) {
      const relative = root.startsWith('random') ? `${source}/ResNet18/random/pt${subset}/${ssl}/seed_000`
        : root.startsWith('mkdt') ? `${source}/ResNet18/mkdt/pt${subset}/${ssl}/ssl_seed_000`
        : `${source}/ResNet18BN/pt${subset}/${ssl}/ssl_seed_000`;
      const file = path.join(bench, root, relative, 'ssl.jsonl');
      if (!exists(file)) continue;
      const values = logRows(file).slice(-200)
        .filter(r => r.seconds > 0 && Number.isFinite(r.seconds)).map(r => r.seconds);
      const k = `${source}|${subset}|${ssl}`;
      sslRates.set(k, [...(sslRates.get(k) || []), ...values]);
    }
    const dir = path.join(bench, 'random_eval_results_interactive_10seeds', source, 'ResNet18/random', `pt${subset}`, ssl, 'seed_000/evaluations');
    if (!fs.existsSync(dir)) continue;
    for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.json'))) {
      const row = read(path.join(dir, file));
      if (!row.complete || !(row.seconds > 0)) continue;
      const k = `${source === 'TinyImageNet' ? 64 : 32}|${row.expected.target}|${row.expected.label_percentage}`;
      probeRates.set(k, [...(probeRates.get(k) || []), row.seconds]);
    }
  }
  const p90 = values => { assert.ok(values && values.length, 'Missing measured runtime'); return [...values].sort((a,b)=>a-b)[Math.ceil(values.length*.9)-1]; };
  for (const [k,v] of probeRates) probeRates.set(k, p90(v));
  for (const [k,v] of sslRates) sslRates.set(k, p90(v));
  const jobs = [], trainNames = new Map(), blocked = [];
  for (const unit of units.values()) {
    const folder = path.join(bench, unit.checkpoint_dir);
    const ready = ['encoder.pt','config.json','ssl_complete.json'].every(f => exists(path.join(folder,f)));
    if (ready) {
      const c=read(path.join(folder,'config.json')),m=read(path.join(folder,'ssl_complete.json'));
      assert.deepEqual([c.signature.source,c.signature.selection_method,c.signature.subset,c.signature.ssl,c.identity.seed,c.signature.protocol.ssl.subset_epochs],
        [unit.source,unit.kind,unit.subset,unit.ssl,unit.seed,unit.epochs]);
      assert.deepEqual(c.signature.code_hashes, hashes); assert.ok(m.complete && m.epochs===unit.epochs); assert.deepEqual(m.identity,c.identity);
      continue;
    }
    let dependencies = [];
    if (unit.kind === 'mkdt' && unit.subset === 0.2) {
      const token = unit.source === 'TinyImageNet' ? 'Tiny' : unit.source;
      const final = token === 'Tiny' ? 10000 : 5000;
      const syntheticRoot = process.env.MKDT_SYNTH_ROOT || path.join(repo, 'benchmarks/mkdt_multidataset/results/batch_20261010T160046Z_Y6dqZS/synthetic');
      const output = path.join(syntheticRoot,'generated/ConvNet',unit.source,`${token}_0.2pct`);
      if (!exists(path.join(output,`images_${final}.pt`)) || !exists(path.join(output,'complete_0.2.json'))
          || !read(path.join(output,'complete_0.2.json')).complete) {
        const name=`mk02_${token}_dist_${final}`;
        if (!queue[name]) blocked.push(name);
        dependencies=[name];
      }
    }
    let epoch = exists(path.join(folder,'encoder.pt')) ? unit.epochs : 0;
    const log = path.join(folder,'ssl.jsonl');
    if (!epoch && exists(path.join(folder,'training.pt')) && exists(log)) {
      const rows=logRows(log);
      epoch=rows.length?Math.floor(rows.at(-1).epoch/500)*500:0;
    }
    assert.ok(epoch<=unit.epochs);
    const rate = sslRates.get(`${unit.source}|${unit.subset}|${unit.ssl}`);
    assert.ok(Number.isFinite(rate), 'No SSL timing for '+unit.id);
    const name='dfssl_'+unit.id;
    trainNames.set(unit.id,name);
    jobs.push({name,stage:'ssl',unit,dependencies,seconds:Math.ceil((unit.epochs-epoch)*rate+150),resume_epoch_estimate:epoch});
  }
  const bundles=[];
  for (const cell of selected) for (const seed of [0,1]) {
    const unit=unitFor(cell,seed),folder=path.join(bench,unit.output_dir,'evaluations');
    const probes=[0,1,2].filter(p=>!exists(path.join(folder,`${cell.target}_lbl${cell.labels}_probe${p}.json`))
      && !queuedProbes.has([unit.id,key(cell),p].join('|')));
    if (!probes.length) continue;
    const rate=probeRates.get(`${cell.source==='TinyImageNet'?64:32}|${cell.target}|${cell.labels}`);
    assert.ok(Number.isFinite(rate),'No probe timing: '+key(cell));
    bundles.push({unit,cell,probes,seconds:probes.length*rate,dependencies:trainNames.has(unit.id)?[trainNames.get(unit.id)]:[]});
  }
  const short=bundles.filter(b=>b.seconds+150<1200), packs=bundles.filter(b=>b.seconds+150>=1200).map(b=>[b]);
  for(let i=0;i<short.length;i+=5)packs.push(short.slice(i,i+5));
  for(const pack of packs){
    const name='dfprobe_'+crypto.createHash('sha256').update(JSON.stringify(pack.map(b=>[b.unit.id,key(b.cell),b.probes]))).digest('hex').slice(0,16);
    jobs.push({name,stage:'probe',bundles:pack,dependencies:[...new Set(pack.flatMap(b=>b.dependencies))],seconds:Math.ceil(pack.reduce((n,b)=>n+b.seconds,150))});
  }
  let tsv='';
  for(const job of jobs){const file=path.join(batch,job.name+'.json');fs.writeFileSync(file,JSON.stringify(job,null,2)+'\n',{flag:'wx'});
    tsv+=[job.name,job.stage,job.seconds,job.dependencies.join(',') || '-',file].join('\t')+'\n';}
  fs.writeFileSync(path.join(batch,'jobs.tsv'),tsv);
  const summary={blank_cells:report.length,selected_cells:selected.length,partial_cells_ignored:report.filter(c=>c.status==='partial_excluded').length,
    complete_cells_skipped:report.filter(c=>c.status==='already_complete').length,mkdt02_cells_excluded:report.filter(c=>c.status==='mkdt02_excluded').length,
    ssl_jobs:jobs.filter(j=>j.stage==='ssl').length,probe_jobs:packs.length,probes:bundles.reduce((n,b)=>n+b.probes.length,0),
    blocked_generation_jobs:[...new Set(blocked)],buffer_seconds:150};
  fs.writeFileSync(path.join(batch,'coverage.json'),JSON.stringify({summary,cells:report},null,2)+'\n');
  console.log('PLAN:',JSON.stringify(summary));
  if(blocked.length && !dry)throw Error('Submit MKDT 0.2% generation first: bash Evaluate_benchamrk/submit_mkdt_02_wang_killarney.sh');
  if (!dry) fs.writeFileSync(approvedFile, JSON.stringify([...approved].sort(), null, 2) + '\n');
}

module.exports={parseTable,classify,key,unitFor};
if(require.main===module)main();
