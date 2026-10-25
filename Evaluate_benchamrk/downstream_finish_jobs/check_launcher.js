// Exercise shell dependency/resource handling with fake Slurm; never execute Python.
const assert=require('assert/strict'),fs=require('fs'),os=require('os'),path=require('path');
const {spawnSync}=require('child_process');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'downstream_launcher_'));
function write(file,text,executable=false){fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,text);if(executable)fs.chmodSync(file,0o700);}
try{
  const bench=path.join(temp,'repo/Evaluate_benchamrk'),script=path.join(bench,'submit_remaining_downstream_wang_killarney.sh');
  write(script,fs.readFileSync(path.join(__dirname,'../submit_remaining_downstream_wang_killarney.sh')));
  write(path.join(temp,'venv/bin/activate'),'');
  for(const marker of ['cifar-10-batches-py/data_batch_1','cifar-10-batches-py/test_batch',
    'cifar-100-python/train','cifar-100-python/test','tinyimagenet.pt','fgvc-aircraft-2013b/data/variants.txt',
    'CUB_200_2011/images.txt','StanfordDogs/train_list.mat','Oxford102Flowers/setid.mat'])write(path.join(temp,'data',marker),'fixture');
  fs.mkdirSync(path.join(temp,'synthetic'));
  write(path.join(bench,'submit_mkdt_02_wang_killarney.sh'),`#!/bin/bash\ntouch '${temp}/generation_submitted'\n`,true);
  const fakePlan=path.join(temp,'fake_plan.js');
  write(fakePlan,`const fs=require('fs'),p=require('path');
const batch=process.argv[3],queue=JSON.parse(fs.readFileSync(process.argv[4]));
const parent='mk02_CIFAR10_dist_5000';
fs.mkdirSync(batch,{recursive:true});
const jobs=[{name:'dfssl_a',stage:'ssl',seconds:230,dependencies:[parent]},
 {name:'dfssl_b',stage:'ssl',seconds:240,dependencies:[parent]},
 {name:'dfprobe_ab',stage:'probe',seconds:400,dependencies:['dfssl_a','dfssl_b']}];
let tsv='';for(const j of jobs){const f=p.join(batch,j.name+'.json');fs.writeFileSync(f,JSON.stringify(j));tsv+=[j.name,j.stage,j.seconds,j.dependencies.join(','),f].join('\\t')+'\\n';}
fs.writeFileSync(p.join(batch,'jobs.tsv'),tsv);
fs.writeFileSync(p.join(batch,'coverage.json'),JSON.stringify({summary:{blocked_generation_jobs:queue[parent]?[]:[parent]}}));
`);
  const bin=path.join(temp,'bin');
  for(const [name,body] of Object.entries({
    node:`if [[ $1 == */downstream_finish_jobs/plan.js ]]; then exec '${process.execPath}' '${fakePlan}' "$@"; fi\nexec '${process.execPath}' "$@"`,
    module:'exit 0',scontrol:"echo 'ClusterName = killarney'",sinfo:"echo 'gpu:l40s:4'",sacctmgr:"echo 'aip-boyuwang|'",
    squeue:`[[ ! -f '${temp}/generation_submitted' ]] || echo 'mk02_CIFAR10_dist_5000|70001'`,
    sbatch:`printf '%s\\n' "$*" >> '${temp}/calls'\nn=0; [[ ! -f '${temp}/counter' ]] || read -r n < '${temp}/counter'\nn=$((n+1)); echo "$n" > '${temp}/counter'; echo "$((80000+n))"`,
  }))write(path.join(bin,name),`#!/bin/bash\n${body}\n`,true);
  const env={...process.env,PATH:`${bin}:${process.env.PATH}`,SLURM_JOB_ID:'',
    BENCH_VENV:path.join(temp,'venv'),BENCH_DATA_ROOT:path.join(temp,'data'),MKDT_SYNTH_ROOT:path.join(temp,'synthetic')};
  const dry=spawnSync('bash',[script,'--dry-run'],{env,encoding:'utf8'});
  assert.equal(dry.status,0,dry.stdout+dry.stderr);
  assert(!fs.existsSync(path.join(temp,'calls')));
  assert(!fs.existsSync(path.join(temp,'generation_submitted')));
  const real=spawnSync('bash',[script],{env,encoding:'utf8'});
  assert.equal(real.status,0,real.stdout+real.stderr);
  const calls=fs.readFileSync(path.join(temp,'calls'),'utf8').trim().split('\n');
  assert.equal(calls.length,3);
  assert(calls.slice(0,2).every(c=>c.includes('--dependency=afterok:70001')));
  assert(calls[2].includes('--dependency=afterok:80001:80002'));
  assert(calls.every(c=>c.includes('--account=aip-boyuwang')&&c.includes('--mem=6G')
    &&c.includes('--cpus-per-task=1')&&c.includes('--gres=gpu:l40s:1')
    &&!c.includes('--array')&&!c.includes('--partition')));
  assert(calls.every(c=>c.includes('--worker')&&c.includes('df')));
  console.log('PASS: dry-run safety, generation -> SSL -> probe dependencies, Wang/L40S/1 CPU/6G, and worker arguments.');
}finally{fs.rmSync(temp,{recursive:true,force:true});}
