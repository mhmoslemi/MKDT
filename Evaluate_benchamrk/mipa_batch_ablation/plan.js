const fs = require('fs'), path = require('path'), crypto = require('crypto'), assert = require('assert/strict');
const here = __dirname, bench = path.dirname(here), root = path.dirname(bench);
const batches = [128, 256, 512, 2048, 4096, 8192];
const accounts = ['aip-boyuwang', 'aip-yiweilu'];
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const p95 = values => {
    assert.ok(values.length && values.every(n => Number.isFinite(n) && n > 0));
    return values.sort((a, b) => a - b)[Math.ceil(.95 * values.length) - 1];
};
function generationRate(file) {
    let previous; const rates = [];
    for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
        const match = line.match(/^\[([\d-]+) ([\d:]+)\] iter = (\d+)/);
        if (!match) continue;
        const current = {time: Date.parse(match[1] + 'T' + match[2] + 'Z') / 1000, iteration: +match[3]};
        if (previous && current.iteration > previous.iteration && current.time > previous.time && current.time - previous.time < 300)
            rates.push((current.time - previous.time) / (current.iteration - previous.iteration));
        previous = current;
    }
    return p95(rates);
}
function main(batch) {
    batch = path.resolve(batch);
    assert.ok(batch.startsWith(path.join(here, 'batches') + path.sep));
    assert.equal(batches.length, 6); assert.ok(!batches.includes(1024));
    const teacher = path.join(bench, 'full_ssl_checkpoints_1000ep_3seeds/CIFAR10/ResNet18/full/pt100/simclr/seed_000/encoder.pt');
    const teacherMarker = path.join(path.dirname(teacher), 'ssl_complete.json');
    assert.ok(read(teacherMarker).complete && fs.statSync(teacher).size > 0);
    const baseline = path.join(root, 'OT-SSL-DD/ours_syntehtic/CIFAR10/ResNet18BN/pt1/res_IIC_CIFAR10_ResNet18BN_1percent_final.pt');
    assert.ok(fs.statSync(baseline).size > 0);
    const files = fs.readdirSync(bench).filter(f => f.endsWith('.py')).map(f => 'Evaluate_benchamrk/' + f).concat([
        'networks.py', 'benchmarks/downstream_data.py', 'Evaluate_benchamrk/protocol.json',
        'OT-SSL-DD/method1.py', 'OT-SSL-DD/eval.py', 'OT-SSL-DD/dm_dsa.py', 'OT-SSL-DD/dd_teachers.py',
        'OT-SSL-DD/utils.py', 'OT-SSL-DD/networks.py', 'Evaluate_benchamrk/mipa_batch_ablation/worker.py',
        'Evaluate_benchamrk/mipa_batch_ablation/plan.js', 'Evaluate_benchamrk/submit_mipa_batch_ablation_vulcan.sh']);
    const code_hashes = Object.fromEntries(files.map(f => [f, hash(path.join(root, f))]));
    const reference = path.join(bench, 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main/CIFAR10/ResNet18BN/pt1/simclr/ssl_seed_000');
    assert.ok(read(path.join(reference, 'ssl_complete.json')).complete);
    const sslRate = p95(fs.readFileSync(path.join(reference, 'ssl.jsonl'), 'utf8').trim().split('\n').slice(-200).map(l => JSON.parse(l).seconds));
    const probeRate = p95([0, 1, 2].map(seed => {
        const probe = read(path.join(reference, `evaluations/CIFAR10_lbl1_probe${seed}.json`));
        assert.ok(probe.complete && probe.encoder_unchanged && probe.expected.evaluation.epochs === 100);
        return probe.seconds;
    }));
    const oldRate = generationRate(path.join(root, 'OT-SSL-DD/ours_syntehtic/logs/CIFAR10_ResNet18BN_pt1.log'));
    const largeRate = Math.max(...['mipa6k_CIFAR10_p2_g1000-1980004.out', 'mipa6k_CIFAR10_p2_g2000-1980005.out']
        .map(f => generationRate(path.join(bench, 'mipa_b6000_i5000_lr0p0015/logs', f))));
    const queue = new Map(fs.readFileSync(path.join(batch, 'queue.tsv'), 'utf8').trim().split('\n').filter(Boolean).map(l => l.split('|').map(s => s.trim())));
    const baseline_info = {batch_real: 1024, path: path.relative(root, baseline), file_sha256: hash(baseline),
        ssl_folder: path.relative(root, reference), reused: true, jobs: 0};
    fs.writeFileSync(path.join(batch, 'baseline_1024.json'), JSON.stringify(baseline_info, null, 2) + '\n', {flag: 'wx'});
    const planned = []; let tsv = '';
    for (const [index, size] of batches.entries()) {
        const name = `mipab_CIFAR10_p1_b${size}`;
        if (queue.has(name)) { console.log(`SKIP active ${name}: ${queue.get(name)}`); continue; }
        const summary = path.join(here, `results/batch_${size}/simclr/ssl_seed_000/summary_lbl1.json`);
        if (fs.existsSync(summary)) {
            const saved = read(summary);
            assert.ok(saved.complete && saved.completed_probes === 5 && saved.ssl_epochs === 2000);
            assert.equal(saved.generation_args.batch_real, size);
            assert.equal(saved.generation_args.Iteration, 8000);
            assert.equal(saved.generation_args.lr_img, .001);
            console.log(`SKIP complete ${name}`); continue;
        }
        // Approximate from the prior 1024/500-image and 6000/1000-image runs; larger old synthetic set makes this conservative.
        const affineRate = oldRate + (size - 1024) * (largeRate - oldRate) / (6000 - 1024);
        const generation_rate = Math.max(affineRate, largeRate * (size + 3 * 500) / (6000 + 3 * 1000));
        const seconds = Math.ceil(8000 * generation_rate + 2000 * sslRate + 5 * probeRate + 40 + 60);
        assert.ok(generation_rate > 0 && seconds > 0 && seconds < 6 * 3600);
        const generation_args = {dataset: 'CIFAR10', percentage: 1, model: 'ResNet18BN', seed: 0,
            Iteration: 8000, lr_img: .001, batch_real: size, temperature: .2, disp_weight: 0, pair_weight: 0,
            num_random_networks: 1, random_models: 'ResNet18BN', dsa: true,
            distill_aug_strategy: 'color_crop_cutout_flip_scale_rotate', distill_aug_mode: 'M',
            teacher_source: 'pretrained', teacher_seed: [0], teacher_ssl: 'simclr',
            teacher_path: path.join(bench, 'full_ssl_checkpoints_1000ep_3seeds'), feature_batch: 128};
        const job = {name, account: accounts[index % 2], generation_args, code_hashes,
            teacher: {path: path.relative(root, teacher), file_sha256: hash(teacher), marker_sha256: hash(teacherMarker)},
            evaluation: {ssl: 'simclr', epochs: 2000, ssl_seeds: [0], probe_epochs: 100, probe_seeds: [0, 1, 2, 3, 4], labels: [1]},
            timing: {seconds, generation_iteration_seconds_estimated: generation_rate, ssl_epoch_seconds_p95: sslRate,
                probe_seconds_p95: probeRate, startup_and_initialization_seconds: 40, buffer_seconds: 60,
                reference_generation_1024_seconds_p95: oldRate, reference_generation_6000_seconds_p95: largeRate}};
        const file = path.join(batch, name + '.json');
        fs.writeFileSync(file, JSON.stringify(job, null, 2) + '\n', {flag: 'wx'});
        tsv += [name, job.account, seconds, file].join('\t') + '\n'; planned.push(job);
    }
    fs.writeFileSync(path.join(batch, 'jobs.tsv'), tsv, {flag: 'wx'});
    fs.writeFileSync(path.join(batch, 'coverage.json'), JSON.stringify({dataset: 'CIFAR10', subset: 1, synthetic_images: 500,
        batches, excluded_existing_batch: 1024, generation_iterations: 8000, generation_lr: .001, ssl_epochs: 2000,
        ssl_seeds: [0], probe_epochs: 100, probe_seeds: [0, 1, 2, 3, 4], labels: [1], jobs: planned.length,
        accounts: Object.fromEntries(accounts.map(a => [a, planned.filter(j => j.account === a).length])),
        walltimes_seconds: planned.map(j => ({batch: j.generation_args.batch_real, seconds: j.timing.seconds})),
        baseline: baseline_info, timings_are_approximate: true}, null, 2) + '\n', {flag: 'wx'});
    console.log(`Prepared ${planned.length} batch-ablation jobs; batch 1024 has no new jobs.`);
}
if (require.main === module) main(process.argv[2]);
