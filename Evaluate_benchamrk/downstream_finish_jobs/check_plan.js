const assert=require('assert/strict');
const fs=require('fs');
const path=require('path');
const {spawnSync}=require('child_process');
const {parseTable,classify,key,unitFor}=require('./plan');
const here=__dirname;
const sample={source:'CIFAR10',kind:'mkdt',subset:1,ssl:'simclr',target:'Aircraft',labels:1,status:'blank'};
assert.equal(classify(sample,{completed:0,expected:6},new Set(),false),'selected');
assert.equal(classify(sample,{completed:3,expected:6},new Set(),false),'partial_excluded');
assert.equal(classify(sample,{completed:6,expected:6},new Set(),false),'already_complete');
assert.equal(classify(sample,{completed:3,expected:6},new Set([key(sample)]),false),'selected');
assert.equal(classify({...sample,subset:2,ssl:'barlowtwins'},{completed:0,expected:6},new Set(),false),'table_excluded');
assert.equal(classify({...sample,subset:0.2},{completed:0,expected:6},new Set(),true),'mkdt02_excluded');
assert.equal(unitFor({...sample,kind:'ours',subset:2},0).epochs,1000);
assert.equal(unitFor({...sample,subset:2},0).epochs,2000);
fs.mkdirSync(path.join(here,'batches'),{recursive:true});
const temp=fs.mkdtempSync(path.join(here,'batches/check_'));
try{
  fs.writeFileSync(path.join(temp,'queue.json'),'{}');
  const run=(name,flags=[])=>{
    const batch=path.join(temp,name);fs.mkdirSync(batch);
    const result=spawnSync('node',[path.join(here,'plan.js'),batch,path.join(temp,'queue.json'),'--dry-run',...flags],{encoding:'utf8'});
    assert.equal(result.status,0,result.stdout+result.stderr);
    const coverage=JSON.parse(fs.readFileSync(path.join(batch,'coverage.json')));
    const jobs=fs.readFileSync(path.join(batch,'jobs.tsv'),'utf8').trim().split('\n').filter(Boolean).map(line=>{
      const columns=line.split('\t');assert.equal(columns.length,5);
      assert(columns[3]);const job=JSON.parse(fs.readFileSync(columns[4]));
      assert(job.seconds>=150);return job;
    });
    const excluded=new Set(coverage.cells.filter(c=>c.status==='partial_excluded').map(key));
    const scheduled=[];
    for(const job of jobs)if(job.stage==='probe')for(const bundle of job.bundles){
      assert(!excluded.has(key(bundle.cell)));
      assert(!(bundle.unit.subset===2&&bundle.unit.ssl==='barlowtwins'));
      assert.equal(bundle.unit.model,'ResNet18');
      for(const probe of bundle.probes)scheduled.push([bundle.unit.id,key(bundle.cell),probe].join('|'));
    }
    assert.equal(new Set(scheduled).size,scheduled.length);
    const names=new Set(jobs.map(j=>j.name));
    for(const job of jobs)for(const dependency of job.dependencies){
      assert(names.has(dependency)||/^mk02_.*_dist_\d+$/.test(dependency));
    }
    console.log(name,JSON.stringify(coverage.summary));
    return {coverage,jobs,scheduled};
  };
  const included=run('with02');
  const excluded=run('without02',['--skip-mkdt02']);
  assert(excluded.coverage.cells.filter(c=>c.status==='selected').every(c=>!(c.kind==='mkdt'&&c.subset===0.2)));
  const probeJob=included.jobs.find(j=>j.stage==='probe');
  if(probeJob){
    const prior=path.join(here,'batches',path.basename(temp),probeJob.name+'.json');
    fs.writeFileSync(prior,JSON.stringify(probeJob));
    fs.writeFileSync(path.join(temp,'queue.json'),JSON.stringify({[probeJob.name]:'12345'}));
    const queued=run('with_queued');
    assert.equal(queued.scheduled.length,included.scheduled.length-probeJob.bundles.reduce((n,b)=>n+b.probes.length,0));
  }
  console.log('PASS: real table parsing, partial exclusion, disabled cells, schedules, unique probes, dependencies and queue deduplication.');
}finally{
  fs.rmSync(temp,{recursive:true,force:true});
}
