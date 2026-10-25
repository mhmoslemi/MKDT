// No Slurm calls or Python execution: exercise the submitter against an isolated fake cluster.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');
const repo = path.resolve(__dirname, '../..');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'mkdt02_launcher_'));
const bench = path.join(temp, 'repo/Evaluate_benchamrk');
const script = path.join(bench, 'submit_mkdt_02_wang_killarney.sh');
const write = (name, content) => {
  fs.mkdirSync(path.dirname(name), {recursive: true});
  fs.writeFileSync(name, content);
};
try {
  for (const name of ['buffer.py', 'sort_dset.py', 'distill.py', 'utils.py', 'networks.py', 'reparam_module.py',
    'Evaluate_benchamrk/mkdt_02_jobs/worker.py', 'Evaluate_benchamrk/mkdt_02_jobs/check_resume.py',
    'Evaluate_benchamrk/submit_mkdt_02_wang_killarney.sh']) {
    write(path.join(temp, 'repo', name), fs.readFileSync(path.join(repo, name)));
  }
  const synthetic = path.join(temp, 'synthetic');
  for (const dataset of ['CIFAR10', 'CIFAR100', 'Tiny']) {
    write(path.join(synthetic, 'work/target_rep/barlow_twins', `${dataset}_target_rep_train.pt`), 'fixture');
  }
  for (const marker of ['cifar-10-batches-py/data_batch_1', 'cifar-100-python/train', 'tinyimagenet.pt']) {
    write(path.join(temp, 'data', marker), 'fixture');
  }
  write(path.join(temp, 'venv/bin/activate'), '');
  for (const [name, body] of Object.entries({
    scontrol: "echo 'ClusterName = killarney'",
    sinfo: "echo 'gpu:l40s:4'",
    sacctmgr: "echo 'aip-boyuwang|'",
    squeue: "printf '%s' \"${FAKE_QUEUE:-}\"",
    module: 'exit 0',
    sbatch: 'printf "%s\\n" "$*" >> "$FAKE_CALLS"; n=0; [[ ! -f $FAKE_COUNTER ]] || read -r n < "$FAKE_COUNTER"; n=$((n+1)); echo "$n" > "$FAKE_COUNTER"; echo "$((800000000+n))"',
  })) {
    const filename = path.join(temp, 'bin', name);
    write(filename, `#!/bin/bash\n${body}\n`);
    fs.chmodSync(filename, 0o700);
  }
  const env = {...process.env, PATH: `${temp}/bin:${process.env.PATH}`, SLURM_JOB_ID: '',
    MKDT_SYNTH_ROOT: synthetic, BENCH_DATA_ROOT: `${temp}/data`, BENCH_VENV: `${temp}/venv`,
    FAKE_CALLS: `${temp}/calls`, FAKE_COUNTER: `${temp}/counter`};
  const run = (extra = {}, dry = true) => {
    const result = spawnSync('bash', [script, ...(dry ? ['--dry-run'] : [])], {env: {...env, ...extra}, encoding: 'utf8'});
    assert.strictEqual(result.status, 0, result.stderr + result.stdout);
    const batches = path.join(bench, 'mkdt_02_jobs/runs/batches');
    const batch = fs.readdirSync(batches).map(n => path.join(batches, n))
      .sort((a, b) => fs.statSync(a).mtimeMs - fs.statSync(b).mtimeMs).at(-1);
    return fs.readFileSync(path.join(batch, 'submitted.tsv'), 'utf8').trim().split('\n').slice(1)
      .map(line => line.split('\t'));
  };
  const rows = run();
  assert.strictEqual(rows.length, 83);
  assert.strictEqual(rows.filter(r => r[2] === 'experts').length, 60);
  assert.strictEqual(rows.filter(r => r[2] === 'init').length, 3);
  assert.strictEqual(rows.filter(r => r[2] === 'distill').length, 20);
  for (const dataset of ['CIFAR10', 'CIFAR100', 'Tiny']) {
    const experts = rows.filter(r => r[2] === 'experts' && r[3] === dataset);
    assert.deepStrictEqual(experts.map(r => Number(r[4])), Array.from({length: 20}, (_, i) => i * 5));
    const init = rows.find(r => r[2] === 'init' && r[3] === dataset);
    assert.deepStrictEqual(init[6].split(':'), experts.map(r => r[1]));
    let previous = init[1];
    for (const row of rows.filter(r => r[2] === 'distill' && r[3] === dataset)) {
      assert.strictEqual(row[6], previous);
      previous = row[1];
    }
  }
  assert.strictEqual(run({}, false).length, 83);
  const calls = fs.readFileSync(env.FAKE_CALLS, 'utf8').trim().split('\n');
  assert.strictEqual(calls.length, 83);
  assert.strictEqual(calls.filter(c => c.includes('--gres=gpu:l40s:1')).length, 80);
  assert(calls.every(c => c.includes('--account=aip-boyuwang') && c.includes('--mem=6G')
    && c.includes('--cpus-per-task=1') && !c.includes('--partition') && !c.includes('--array')));
  const work = path.join(bench, 'mkdt_02_jobs/runs');
  write(path.join(work, 'CIFAR10/shard_000.json'), '{}');
  write(path.join(work, 'CIFAR10/distill_01000.json'), '{}');
  const partial = run({FAKE_QUEUE: 'mk02_CIFAR10_exp_5|12345\n'});
  assert.strictEqual(partial.length, 80);
  assert(partial.find(r => r[0] === 'mk02_CIFAR10_init')[6].split(':').includes('12345'));
  for (const dataset of ['CIFAR10', 'CIFAR100', 'Tiny']) {
    const source = dataset === 'Tiny' ? 'TinyImageNet' : dataset;
    const output = path.join(synthetic, 'generated/ConvNet', source, `${dataset}_0.2pct`);
    write(path.join(output, 'complete_0.2.json'), '{}');
    write(path.join(output, `images_${dataset === 'Tiny' ? 10000 : 5000}.pt`), 'fixture');
  }
  assert.strictEqual(run().length, 0);
  console.log('PASS: 83 jobs, exact 100-expert coverage, dependencies, queued-job reuse, and completed-stage skips.');
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
