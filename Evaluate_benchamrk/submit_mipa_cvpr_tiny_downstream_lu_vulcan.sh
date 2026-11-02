#!/bin/bash
set -euo pipefail
bench=/scratch/mmoslem3/MKDT/Evaluate_benchamrk
run="$bench/mipa_cvpr_followup"
subset=${MIPA_TINY_SUBSET:-0.2}
[[ $subset == 0.2 || $subset == 1 ]] || exit 2
if [[ ${1:-} == --worker ]]; then
    : "${SLURM_JOB_ID:?Submit through Slurm}"
    module --force purge
    module load StdEnv/2023 python/3.11.5
    source /home/mmoslem3/ENV/bin/activate
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export XDG_CACHE_HOME="$run/cache/xdg" TORCH_HOME="$run/cache/torch" HF_HOME="$run/cache/hf"
    export TMPDIR="${SLURM_TMPDIR:?}"
    cd "$bench"
    exec python -u - "$2" "$3" "$4" "$5" "${6:-0.2}" <<'PY'
import fcntl
import os
from pathlib import Path
import sys

bench = Path('/scratch/mmoslem3/MKDT/Evaluate_benchamrk')
sys.path[:0] = [str(bench / 'mipa_cvpr_followup'), str(bench)]
import worker as followup

group, config_sha, encoder_sha, helper_sha, subset = sys.argv[1:]
subset = float(subset)
assert subset in (0.2, 1)
assert os.environ.get('SLURM_JOB_ACCOUNT') == 'aip-yiweilu'
assert os.environ.get('SLURM_CLUSTER_NAME', os.environ.get('CC_CLUSTER')) == 'vulcan'
assert followup.torch.cuda.is_available() and followup.torch.cuda.device_count() == 1
assert followup.common.file_hash(bench / 'mipa_cvpr_followup/worker.py') == helper_sha
followup.torch.set_num_threads(1)
followup.torch.set_num_interop_threads(1)
followup.torch.backends.cuda.matmul.allow_tf32 = False
followup.torch.backends.cudnn.allow_tf32 = False
targets = {'CIFAR10': ['CIFAR10'], 'CIFAR100': ['CIFAR100'], 'Aircraft_Dogs': ['Aircraft', 'Dogs']}[group]
encoder_dir = bench / ('mipa_cvpr_color_crop/results/TinyImageNet/ResNet18BN/pt%g/simclr/ssl_seed_000' % subset)
folder = bench / ('mipa_cvpr_followup/results/downstream/TinyImageNet/ResNet18/pt%g/simclr/ssl_seed_000' % subset)
name = 'cvdn_TinyImageNet_p%s_%s' % (('%g' % subset).replace('.', 'p'), group)
with (encoder_dir / '.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    assert followup.common.file_hash(encoder_dir / 'config.json') == config_sha
    assert followup.common.file_hash(encoder_dir / 'encoder.pt') == encoder_sha
    config = followup.read(encoder_dir / 'config.json')
    marker = followup.read(encoder_dir / 'ssl_complete.json')
    signature = config['signature']
    assert followup.common.digest_json(signature) == config['identity']['signature']
    assert marker['complete'] and marker['epochs'] == 2000 and marker['identity'] == config['identity']
    assert marker['checkpoint_file_sha256'] == encoder_sha
    assert [signature['source'], signature['subset'], signature['model'], signature['ssl']] == ['TinyImageNet', subset, 'ResNet18', 'simclr']
    assert signature['protocol']['evaluation'] == followup.read(bench / 'protocol.json')['evaluation']
    assert followup.common.file_hash(signature['synthetic_path']) == signature['synthetic_sha256']
    job = dict(source='TinyImageNet', subset=subset, model='ResNet18', method='simclr', ssl_epochs=2000, seed=0,
        data_root='/scratch/mmoslem3/data', probe_seeds=list(range(5)), synthetic_file=signature['synthetic_path'],
        synthetic_sha256=signature['synthetic_sha256'], table='CVPR_tables/cifar_downstream.tex',
        cells=[dict(target=target, labels=labels) for target in targets for labels in (1, 5)])
    with followup.common.exclusive_run(folder / 'locks' / name):
        followup.probes(job, encoder_dir / 'encoder.pt', config, marker, folder)
        followup.common.atomic_json(folder / ('complete_' + name + '.json'), dict(complete=True, job=name, identity=config['identity']))
PY
fi
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || exit 2
dry=0
[[ ${1:-} != --dry-run ]] || dry=1
[[ $(scontrol show config | awk '$1=="ClusterName" {print $3}') == vulcan ]] || exit 2
[[ $(sinfo -h -o '%G' --Node | sort -u) == *gpu:l40s:* ]] || exit 2
accounts=$(sacctmgr -nP show assoc user=mmoslem3 format=Account)
awk -F '|' '$1=="aip-yiweilu" {found=1} END {exit !found}' <<< "$accounts"
module spider python/3.11.5 >/dev/null 2>&1
plan=$(node - "$bench" "$subset" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const bench=process.argv[2],read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const subset=Number(process.argv[3]);assert([0.2,1].includes(subset));
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const input=path.join(bench,`mipa_cvpr_color_crop/results/TinyImageNet/ResNet18BN/pt${subset}/simclr/ssl_seed_000`);
const output=path.join(bench,`mipa_cvpr_followup/results/downstream/TinyImageNet/ResNet18/pt${subset}/simclr/ssl_seed_000`);
const config=read(path.join(input,'config.json')),marker=read(path.join(input,'ssl_complete.json'));
assert(read(path.join(input,'complete.json')).complete);assert(marker.complete&&marker.epochs===2000);
assert.deepEqual(marker.identity,config.identity);assert.equal(config.signature.generation_args.lr_img,.0015);
assert.equal(config.signature.subset,subset);
assert.equal(config.signature.generation_args.prototype_pool_size,0);
const config_sha=hash(path.join(input,'config.json')),encoder_sha=hash(path.join(input,'encoder.pt'));
assert.equal(encoder_sha,marker.checkpoint_file_sha256);
const helper_sha=hash(path.join(bench,'mipa_cvpr_followup/worker.py'));
const reference=path.join(bench,`mipa_random128/results/TinyImageNet/ResNet18BN/pt${subset}/simclr/ssl_seed_000/evaluations`);
for(const targets of [['CIFAR10'],['CIFAR100'],['Aircraft','Dogs']]){
    const group=targets.join('_'),name='cvdn_TinyImageNet_p'+String(subset).replace('.','p')+'_'+group;
    const markerFile=path.join(output,'complete_'+name+'.json');if(fs.existsSync(markerFile)&&read(markerFile).complete)continue;
    let seconds=0;
    for(const target of targets)for(const labels of [1,5]){
        const times=fs.readdirSync(reference).filter(f=>f.startsWith(target+'_lbl'+labels+'_probe')&&f.endsWith('.json')).map(f=>read(path.join(reference,f)).seconds).sort((a,b)=>a-b);
        assert(times.length&&times.every(s=>s>0));seconds+=5*times[Math.ceil(times.length*.95)-1];
    }
    process.stdout.write([name,group,Math.ceil(seconds*1.25+60),config_sha,encoder_sha,helper_sha].join('\t')+'\n');
}
JS
)
mkdir -p "$run/logs"
exec 9>"$run/.tiny_submit.lock"
flock -n 9 || exit 2
declare -A active=()
while IFS='|' read -r name id; do [[ -z $name ]] || active["$name"]=$id; done < <(squeue -h -u mmoslem3 -o '%j|%i')
while IFS=$'\t' read -r name group seconds config_sha encoder_sha helper_sha; do
    [[ -n $name ]] || continue
    if [[ -n ${active[$name]:-} ]]; then echo "ACTIVE $name: ${active[$name]}"; continue; fi
    printf -v wall '%02d:%02d:%02d' "$((seconds/3600))" "$((seconds/60%60))" "$((seconds%60))"
    job=DRY_RUN
    if (( ! dry )); then
        job=$(sbatch --parsable --job-name="$name" --account=aip-yiweilu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=6G --gres=gpu:l40s:1 --time="$wall" --chdir="$bench" --output="$run/logs/%x-%j.out" --error="$run/logs/%x-%j.err" "$bench/submit_mipa_cvpr_tiny_downstream_lu_vulcan.sh" --worker "$group" "$config_sha" "$encoder_sha" "$helper_sha" "$subset")
        job=${job%%;*}
        [[ $job =~ ^[0-9]+$ ]] || exit 2
        printf '%s\t%s\taip-yiweilu\t%s\n' "$name" "$job" "$wall" >> "$run/submitted_tiny.tsv"
    fi
    echo "$name: $job, aip-yiweilu, $wall"
done <<< "$plan"
