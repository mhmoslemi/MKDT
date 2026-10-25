// Login-node planning reads JSON only; never imports Python or loads tensors.
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const crypto = require('crypto');
const bench = path.resolve(__dirname, '..');
const batch = path.resolve(process.argv[2]);
assert.ok(batch.startsWith(path.join(__dirname, 'batches') + path.sep));
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const omitted = new Set(read(path.join(__dirname, 'omitted_cells.json')));
const cellKey = (unit, probe) => [unit.source, 'ResNet18', unit.kind, unit.subset,
    unit.ssl, probe.target, probe.labels].join('|');
const activeNames = new Set((process.argv[3] || '').split('\n').map(x => x.trim()).filter(Boolean));
const queued = new Set([...activeNames].filter(x => /^dp(?:3|15)_/.test(x)));
const queuedBundles = new Set();
for (const name of queued) {
    let found = false;
    for (const prior of fs.readdirSync(path.join(__dirname, 'batches')).sort().reverse()) {
        const file = path.join(__dirname, 'batches', prior, name + '.json');
        if (!fs.existsSync(file)) continue;
        const job = read(file);
        for (const bundle of job.bundles || [job]) queuedBundles.add(bundle.name);
        found = true;
        break;
    }
    assert.ok(found, 'Cannot identify active downstream job: ' + name);
}
const exists = file => fs.existsSync(file) && fs.statSync(file).size > 0;
const cached = new Map();
const load = file => { if (!cached.has(file)) cached.set(file, read(file)); return cached.get(file); };
function inside(value) {
    const file = path.resolve(bench, value);
    assert.ok(file.startsWith(bench + path.sep), 'Path must stay inside Evaluate_benchamrk');
    return file;
}
const sources = ['CIFAR10', 'CIFAR100', 'TinyImageNet'];
const targets = source => [...sources, 'Aircraft', 'CUB2011', 'Dogs', 'Flowers'].filter(x => x !== source);
const hashes = Object.fromEntries([...fs.readdirSync(bench).filter(x => x.endsWith('.py')).map(x => 'Evaluate_benchamrk/' + x),
    'networks.py', 'benchmarks/downstream_data.py'].sort().map(f => [f,
    crypto.createHash('sha256').update(fs.readFileSync(path.join(bench, '..', f))).digest('hex')]));
const same = (a, b) => { assert.deepEqual(a, b); };
const rates = new Map();
let measurements = 0;
for (const source of sources) for (const subset of [0.2, 1, 2]) for (const ssl of ['simclr', 'barlowtwins']) {
    for (let seed = 0; seed < 10; seed++) {
        const dir = path.join(bench, 'random_eval_results_interactive_10seeds', source, 'ResNet18/random',
            `pt${subset}`, ssl, `seed_${String(seed).padStart(3, '0')}`, 'evaluations');
        if (!fs.existsSync(dir)) continue;
        for (const file of fs.readdirSync(dir).filter(x => x.endsWith('.json'))) {
            const row = read(path.join(dir, file));
            if (!row.complete || !Number.isFinite(row.seconds) || row.seconds <= 0) continue;
            const key = [source === 'TinyImageNet' ? 64 : 32, row.expected.target, row.expected.label_percentage].join('|');
            if (!rates.has(key)) rates.set(key, []);
            rates.get(key).push(row.seconds);
            measurements++;
        }
    }
}
for (const [key, values] of rates) {
    values.sort((a, b) => a - b);
    rates.set(key, values[Math.ceil(values.length * 0.9) - 1]);
}
function ready(unit) {
    const folder = inside(unit.checkpoint_dir);
    if (!['encoder.pt', 'config.json', 'ssl_complete.json'].every(f => exists(path.join(folder, f)))) return false;
    const c = load(path.join(folder, 'config.json')), m = load(path.join(folder, 'ssl_complete.json')), s = c.signature;
    same([s.source, s.model, s.selection_method, s.subset, s.ssl, c.identity.seed],
        [unit.source, 'ResNet18', unit.kind, unit.subset, unit.ssl, unit.seed]);
    same(s.code_hashes, hashes);
    same(m.identity, c.identity);
    assert.ok(m.complete && m.epochs === unit.epochs && s.protocol.ssl.subset_epochs === unit.epochs);
    assert.equal(s.protocol.evaluation.epochs, 100);
    return true;
}
function resultPath(unit, probe) {
    return path.join(inside(unit.output_dir), 'evaluations',
        `${probe.target}_lbl${probe.labels}${unit.kind === 'ours' ? '_probe' + probe.seed : ''}.json`);
}
function done(unit, probe) {
    const file = resultPath(unit, probe);
    if (!exists(file)) return false;
    const row = read(file), c = load(path.join(inside(unit.checkpoint_dir), 'config.json')),
        m = load(path.join(inside(unit.checkpoint_dir), 'ssl_complete.json')), e = row.expected;
    assert.ok(row.complete && row.encoder_unchanged && row.encoder_sha256 === m.encoder_sha256, file);
    same(e.identity, c.identity);
    same([e.source, e.target, e.label_percentage, e.seed], [unit.source, probe.target, probe.labels, probe.seed]);
    same(e.evaluation, c.signature.protocol.evaluation);
    same(e.evaluation_code_hashes, hashes);
    assert.ok(Number.isFinite(row.accuracy_percent) && row.accuracy_percent >= 0 && row.accuracy_percent <= 100);
    return true;
}
const jobs = [], counts = {random: 0, ours: 0, missing_random_encoders: 0, missing_ours_encoders: 0, completed_bundles: 0, queued_bundles: 0, omitted_bundles: 0};
function bundle(unit, probes, suffix) {
    assert.equal(probes.length, 3);
    const name = `dp3_${unit.kind}_${unit.source}_p${String(unit.subset).replace('.', 'p')}_${unit.ssl}_s${unit.seed}_${suffix}`;
    const requested = probes.filter(p => !omitted.has(cellKey(unit, p)));
    if (!requested.length) { counts.omitted_bundles++; return; }
    if (queuedBundles.has(name)) { counts.queued_bundles++; return; }
    if (unit.kind === 'random' && requested.some(p => activeNames.has(
        `rnd_down_${unit.source}_p${String(unit.subset).replace('.', 'p')}_${unit.ssl}_s${unit.seed}_${p.target}_l${p.labels}`))) {
        counts.queued_bundles++; return;
    }
    const pending = requested.filter(p => !done(unit, p));
    if (!pending.length) { counts.completed_bundles++; return; }
    const seconds = Math.ceil(150 + pending.reduce((sum, p) => {
        const rate = rates.get([unit.source === 'TinyImageNet' ? 64 : 32, p.target, p.labels].join('|'));
        assert.ok(Number.isFinite(rate), 'Missing measured probe time');
        return sum + rate;
    }, 0));
    jobs.push({name, unit, probes, missing_probes: pending.length, seconds});
    counts[unit.kind]++;
}
// Finish the Random cells previously launched on Killarney; reuse the same routes/results.
for (const source of ['CIFAR100', 'TinyImageNet']) for (const subset of source === 'CIFAR100' ? [2] : [0.2, 1, 2]) {
    for (const ssl of ['simclr', 'barlowtwins']) for (let seed = 0; seed < 10; seed++) {
        const route = `random_downstream_jobs/routes/${source}_p${String(subset).replace('.', 'p')}_${ssl}_s${seed}.json`;
        if (!exists(path.join(bench, route))) { counts.missing_random_encoders++; continue; }
        const r = read(path.join(bench, route));
        const unit = {...r, kind: 'random', route};
        same([r.source, r.subset, r.ssl, r.seed], [source, subset, ssl, seed]);
        if (!ready(unit)) { counts.missing_random_encoders++; continue; }
        const probes = targets(source).flatMap(target => [1, 5].map(labels => ({target, labels, seed})));
        for (let start = 0; start < probes.length; start += 3) bundle(unit, probes.slice(start, start + 3), `b${start / 3}`);
    }
}
for (const source of sources) for (const subset of [0.2, 1, 2]) for (const ssl of ['simclr', 'barlowtwins']) {
    const oneK = subset === 2 && ssl === 'simclr';
    const root = oneK ? 'ours_eval_results_simclr1k_2ssl3probes_main' : 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_main';
    const output = oneK ? 'ours_eval_results_simclr1k_2ssl3probes_downstream' : 'ours_eval_results_simclr2k_barlow10k_2ssl3probes_downstream';
    for (const seed of [0, 1]) {
        const relative = `${source}/ResNet18BN/pt${subset}/${ssl}/ssl_seed_${String(seed).padStart(3, '0')}`;
        const unit = {kind: 'ours', source, subset, ssl, seed, epochs: ssl === 'barlowtwins' ? 10000 : oneK ? 1000 : 2000,
            checkpoint_dir: `${root}/${relative}`, output_dir: `${output}/${relative}`};
        if (!ready(unit)) { counts.missing_ours_encoders++; continue; }
        for (const target of targets(source)) for (const labels of [1, 5]) {
            bundle(unit, [0, 1, 2].map(seed => ({target, labels, seed})), `${target}_l${labels}`);
        }
    }
}
const short = jobs.filter(job => job.seconds < 1200);
const submitted = jobs.filter(job => job.seconds >= 1200);
let combined = 0;
for (let start = 0; start < short.length; start += 5) {
    const bundles = short.slice(start, start + 5);
    if (bundles.length < 5) { submitted.push(...bundles); continue; }
    const key = crypto.createHash('sha256').update(bundles.map(j => j.name).join('|')).digest('hex').slice(0, 16);
    submitted.push({name: 'dp15_' + key, bundles,
        seconds: 150 + bundles.reduce((n, j) => n + j.seconds - 150, 0),
        missing_probes: bundles.reduce((n, j) => n + j.missing_probes, 0)});
    combined++;
}
let rows = '';
for (const job of submitted) {
    const file = path.join(batch, job.name + '.json');
    fs.writeFileSync(file, JSON.stringify(job, null, 2) + '\n', {flag: 'wx'});
    rows += [job.name, job.seconds, job.missing_probes, file].join('\t') + '\n';
}
fs.writeFileSync(path.join(batch, 'jobs.tsv'), rows, {flag: 'wx'});
console.log('PLAN:', JSON.stringify({...counts, measured_probes: measurements, probe_bundles: jobs.length,
    combined_jobs: combined, jobs: submitted.length,
    missing_probes: jobs.reduce((n, j) => n + j.missing_probes, 0),
    min_minutes: submitted.length ? Math.min(...submitted.map(j => j.seconds)) / 60 : 0,
    max_minutes: submitted.length ? Math.max(...submitted.map(j => j.seconds)) / 60 : 0}));
console.log('Time = measured p90 probe seconds + 2 minutes 30 seconds per submitted job. No SSL training.');
console.log('Five bundles estimated below 20 minutes are combined into one job; leftovers stay separate.');
console.log('Ours 2% SimCLR: only 1k checkpoints; missing encoders are skipped until a later rerun.');
