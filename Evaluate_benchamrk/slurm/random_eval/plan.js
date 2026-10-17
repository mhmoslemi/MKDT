// Small metadata-only planner. All Python, tensor I/O, and probes run in Slurm.
'use strict';
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const read = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const tsv = p => {
  const [head, ...rows] = fs.readFileSync(p, 'utf8').trim().split('\n').map(x => x.split('\t'));
  return rows.map(r => Object.fromEntries(head.map((k, i) => [k, r[i]])));
};
const writeTSV = (p, keys, rows) => fs.writeFileSync(p,
  [keys.join('\t'), ...rows.map(r => keys.map(k => r[k]).join('\t'))].join('\n') + '\n');
const quoted = x => "'" + x.replace(/'/g, "'\\''") + "'";
const groupKey = r => [r.source, r.model, 'random', 'pt' + Number(r.subset), r.ssl].join('/');
const cellKey = r => [groupKey(r), r.target, r.labels].join('/');
const nonempty = p => fs.existsSync(p) && fs.statSync(p).isFile() && fs.statSync(p).size > 0;

function build(bench) {
  const cells = tsv(path.join(bench, 'table_coverage.tsv'))
    .filter(r => r.method === 'random' && r.disabled === 'False');
  if (cells.length !== 324 || new Set(cells.map(cellKey)).size !== 324)
    throw Error('Expected 324 distinct active random-subset table cells');
  const groups = new Map();
  for (const row of cells) {
    if (!['0.2', '1', '2'].includes(row.subset) || !['1', '5'].includes(row.labels))
      throw Error('Unexpected table budget');
    const key = groupKey(row);
    if (!groups.has(key)) groups.set(key, {group: key, source: row.source,
      model: row.model, subset: Number(row.subset), ssl: row.ssl, cells: []});
    groups.get(key).cells.push(row);
  }
  if (groups.size !== 54) throw Error('Expected 54 encoder groups');
  const jobs = Array.from({length: 40}, (_, i) => ({task: String(i + 1).padStart(2, '0'),
    account: i % 10 === 9 ? (process.env.BENCH_SECONDARY_ACCOUNT || 'aip-yiweilu') :
      (process.env.BENCH_ACCOUNT || 'aip-boyuwang'),
    walltime: i < 38 ? (process.env.BENCH_EVAL_TIME || '06:00:00') :
      (process.env.BENCH_EVAL_TAIL_TIME || '12:00:00'),
    wait_for: [], units: [], weight: 0}));
  // Keep both tail groups on the training account, preserving a 36/4 split.
  jobs[39].account = process.env.BENCH_ACCOUNT || 'aip-boyuwang';
  jobs[37].account = process.env.BENCH_SECONDARY_ACCOUNT || 'aip-yiweilu';
  jobs[38].wait_for = ['6334332', '6334333'];
  jobs[39].wait_for = ['6334334', '6334335'];
  const normal = [], units = [];
  // Work estimate is only for balancing, not an accuracy or runtime prediction.
  const trainCounts = {CIFAR10: 50000, CIFAR100: 50000, TinyImageNet: 100000,
    Aircraft: 6667, CUB2011: 5994, Dogs: 12000, Flowers: 1020};
  for (const group of [...groups.values()].sort((a,b) => a.group.localeCompare(b.group))) {
    const expectedCount = group.model === 'ResNet18' ? 14 : 2;
    if (group.cells.length !== expectedCount) throw Error('Wrong target coverage: ' + group.group);
    for (let seed = 0; seed < 10; seed++) {
      const relative = group.group + '/seed_' + String(seed).padStart(3, '0');
      const weight = group.cells.reduce((s, c) => s + trainCounts[c.target] * Number(c.labels) / 100, 0)
        * (group.source === 'TinyImageNet' ? 4 : 1) * (group.model === 'ResNet18' ? 2 : 1);
      const unit = {...group, seed, relative, weight};
      if (group.source === 'TinyImageNet' && group.model === 'ResNet18' && group.subset !== 0.2) {
        const job = jobs[group.subset === 1 ? 38 : 39];
        job.units.push(unit); job.weight += weight;
      } else normal.push(unit);
      units.push(unit);
    }
  }
  normal.sort((a, b) => b.weight - a.weight || a.relative.localeCompare(b.relative));
  for (const unit of normal) {
    const job = jobs.slice(0, 38).reduce((a, b) => a.weight <= b.weight ? a : b);
    job.units.push(unit); job.weight += unit.weight;
  }
  for (const job of jobs) {
    if (!job.units.length) throw Error('Empty job');
    job.units.sort((a, b) => a.relative.localeCompare(b.relative));
    job.probes = job.units.reduce((n, u) => n + u.cells.length, 0);
  }
  if (units.length !== 540 || jobs.reduce((s,j) => s + j.probes, 0) !== 3240)
    throw Error('Incomplete evaluation plan');
  return {version: 1, checkpoint_root: path.resolve(process.env.BENCH_CHECKPOINT_ROOT ||
    path.join(bench, 'random_ssl_checkpoints')), data_root: path.resolve(process.env.BENCH_DATA_ROOT ||
    path.join(bench, '../../data')), venv: path.resolve(process.env.BENCH_VENV || path.join(process.env.HOME, 'ENV')),
    training_cluster: 'killarney', jobs, cells, groups: [...groups.values()]};
}

function prepare(bench) {
  const plan = build(bench), root = path.join(bench, 'slurm/random_eval/batches');
  fs.mkdirSync(root, {recursive: true});
  const batch = fs.mkdtempSync(path.join(root, 'batch_' + new Date().toISOString().replace(/[-:.]/g, '') + '_'));
  for (const folder of ['code/Evaluate_benchamrk', 'code/benchmarks', 'code/CVPR_tables', 'logs', 'jobs', 'results'])
    fs.mkdirSync(path.join(batch, folder), {recursive: true});
  const files = [];
  function copy(source, relative) {
    fs.copyFileSync(source, path.join(batch, relative)); files.push(relative);
  }
  for (const name of fs.readdirSync(bench).filter(n => n.endsWith('.py') ||
    ['protocol.json', 'table_coverage.tsv', 'random_ssl.sh'].includes(n)))
    copy(path.join(bench, name), 'code/Evaluate_benchamrk/' + name);
  copy(path.join(bench, '../networks.py'), 'code/networks.py');
  copy(path.join(bench, '../benchmarks/downstream_data.py'), 'code/benchmarks/downstream_data.py');
  for (const name of fs.readdirSync(path.join(bench, '../CVPR_tables')).filter(n => n.endsWith('.tex')))
    copy(path.join(bench, '../CVPR_tables', name), 'code/CVPR_tables/' + name);
  for (const name of ['plan.js', 'worker.py', 'report.py', 'worker.sh'])
    copy(path.join(bench, 'slurm/random_eval', name), name);
  fs.writeFileSync(path.join(batch, 'plan.json'), JSON.stringify(plan, null, 2) + '\n');
  fs.writeFileSync(path.join(batch, 'environment.sh'), 'BENCH_VENV=' + quoted(plan.venv) + '\n');
  files.push('plan.json', 'environment.sh');
  fs.writeFileSync(path.join(batch, 'bundle.sha256'), files.map(n => sha(path.join(batch, n)) + '  ' + n).join('\n') + '\n');
  writeTSV(path.join(batch, 'jobs.tsv'), ['task', 'account', 'walltime', 'encoders', 'probes', 'afterok'],
    plan.jobs.map(j => ({...j, encoders: j.units.length, afterok: j.wait_for.join(':') || '-'})));
  writeTSV(path.join(batch, 'checklist.tsv'), ['task','table','source','model','subset','ssl','seed','target','labels','checkpoint','result'],
    plan.jobs.flatMap(j => j.units.flatMap(u => u.cells.map(c => ({...c, task: j.task, seed: u.seed,
      checkpoint: path.join(plan.checkpoint_root, u.relative, 'encoder.pt'),
      result: 'results/' + u.relative + '/evaluations/' + c.target + '_lbl' + c.labels + '.json'})))));
  // These are reviewable commands; the launcher resolves finished dependencies
  // and checks all inputs before invoking the same worker.
  for (const job of plan.jobs) {
    fs.writeFileSync(path.join(batch, 'jobs', job.task + '.sh'), '#!/bin/bash\n' +
      '#SBATCH --job-name=random_eval_' + job.task + '\n#SBATCH --account=' + job.account +
      '\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=6G\n' +
      '#SBATCH --gres=gpu:l40s:1\n#SBATCH --time=' + job.walltime + '\n' +
      (job.wait_for.length ? '#SBATCH --dependency=afterok:' + job.wait_for.join(':') +
        '\n#SBATCH --kill-on-invalid-dep=yes\n' : '') +
      '#SBATCH --chdir=' + quoted(batch) + '\n#SBATCH --output=' + quoted(batch + '/logs/%x-%j.out') +
      '\n#SBATCH --error=' + quoted(batch + '/logs/%x-%j.err') +
      '\nexec bash ' + quoted(batch + '/worker.sh') + ' ' + quoted(batch) + ' ' + job.task + '\n');
  }
  console.log(batch);
}

function check(batch, cluster, gpu) {
  const plan = read(path.join(batch, 'plan.json'));
  const bundle = fs.readFileSync(path.join(batch, 'bundle.sha256'), 'utf8').trim().split('\n');
  for (const line of bundle) {
    const [digest, file] = line.split('  ');
    if (sha(path.join(batch, file)) !== digest) throw Error('Batch modified: ' + file);
  }
  if (!nonempty(path.join(plan.venv, 'bin/activate')) || !fs.existsSync(plan.data_root))
    throw Error('Missing virtual environment or data root; prepare with BENCH_VENV/BENCH_DATA_ROOT');
  const types = fs.readFileSync(path.join(batch, 'gres.txt'), 'utf8').match(/gpu:[^:,\s]+:/g) || [];
  if (!types.includes('gpu:' + gpu + ':')) throw Error('GPU type absent from destination sinfo: ' + gpu);
  const accounts = fs.readFileSync(path.join(batch, 'accounts.txt'), 'utf8').split('\n').map(x => x.split('|')[0].trim());
  const queue = new Map(fs.readFileSync(path.join(batch, 'queue.txt'), 'utf8').trim().split('\n').filter(Boolean)
    .map(line => {const [id, name, state] = line.split('|'); return [id, {name, state}];}));
  const previousPath = path.join(batch, 'submitted.tsv');
  const previous = fs.existsSync(previousPath) ? tsv(previousPath) : [];
  const missing = [], submissions = [];
  for (const job of plan.jobs) {
    if (!accounts.includes(job.account)) throw Error('Unavailable account: ' + job.account);
    if (nonempty(path.join(batch, 'results/jobs', job.task + '.json'))) continue;
    if (previous.some(p => p.task === job.task && p.cluster === cluster && queue.has(p.job_id))) continue;
    if (previous.some(p => p.task === job.task && p.cluster !== cluster))
      throw Error('Resume on the same cluster as the original evaluation submission');
    const dependencies = new Set();
    for (const unit of job.units) {
      const folder = path.join(plan.checkpoint_root, unit.relative);
      const ready = ['encoder.pt', 'config.json', 'ssl_complete.json'].every(n => nonempty(path.join(folder, n)));
      if (ready) {
        const marker = read(path.join(folder, 'ssl_complete.json'));
        if (!marker.complete || marker.epochs !== 1000 || marker.seed !== unit.seed ||
          marker.source !== unit.source || marker.model !== unit.model || marker.method !== unit.ssl)
          throw Error('Invalid SSL completion record: ' + folder);
        continue;
      }
      const parent = unit.subset === 1 ? (unit.ssl === 'simclr' ? '6334332' : '6334333') :
        (unit.ssl === 'simclr' ? '6334334' : '6334335');
      const name = 'random_TinyImageNet_ResNet18_' + unit.subset + '_' + unit.ssl;
      const active = queue.get(parent);
      if (job.wait_for.includes(parent) && cluster === plan.training_cluster && active && active.name === name &&
          ['PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING'].includes(active.state)) dependencies.add(parent);
      else missing.push(folder);
    }
    submissions.push({task: job.task, account: job.account, walltime: job.walltime,
      dependency: dependencies.size ? 'afterok:' + [...dependencies].sort().join(':') : '-', gpu});
  }
  if (missing.length) throw Error(`${missing.length} checkpoints are missing without a valid local training dependency. ` +
    `Run on the cluster holding the checkpoints (training: ${plan.training_cluster}). First missing: ${missing[0]}`);
  writeTSV(path.join(batch, 'submit.tsv'), ['task','account','walltime','dependency','gpu'], submissions);
  console.log('Preflight passed: ' + submissions.length + ' jobs to submit.');
}

function main() {
  const [mode, location, cluster, gpu] = process.argv.slice(2);
  if (mode === 'prepare') prepare(path.resolve(location));
  else if (mode === 'check') check(path.resolve(location), cluster, gpu);
  else if (mode === 'preview') {
    const plan = build(path.resolve(location));
    console.log('40 jobs / 540 encoders / 3240 probes / 324 active random table cells');
    console.log('task\taccount\twalltime\tencoders\tprobes\tafterok');
    for (const j of plan.jobs) console.log([j.task,j.account,j.walltime,j.units.length,j.probes,j.wait_for.join(':') || '-'].join('\t'));
  } else throw Error('Use preview BENCH, prepare BENCH, or check BATCH CLUSTER GPU');
}
module.exports = {build, check};
if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
