// Only the agreed 20 cells in the current dataset-by-row SimCLR main table.
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'../..');
const sources={'CIFAR-10':'CIFAR10','CIFAR-100':'CIFAR100','Tiny ImageNet':'TinyImageNet'};
function eligible(source,subset){return source==='TinyImageNet'?[0.2,1,2].includes(subset):['CIFAR10','CIFAR100'].includes(source)&&subset===2;}
function rows(text){
    let source;const result=[];
    for(const [lineIndex,line]of text.split('\n').entries()){
        const label=line.match(/\\multirow\{3\}\{\*\}\{(CIFAR-10|CIFAR-100|Tiny ImageNet)\}/);
        if(label)source=sources[label[1]];
        const tokens=line.split('&');if(tokens.length!==12||!source)continue;
        const subset=Number(tokens[1].trim());if(![0.2,1,2].includes(subset))continue;
        result.push({source,subset,lineIndex,tokens});
    }
    assert.equal(result.length,9,'Main table layout changed');
    assert.equal(new Set(result.map(r=>r.source+'|'+r.subset)).size,9);
    return result;
}
function replacement(text,s){
    assert.ok(s.complete&&s.ssl_epochs===2000&&s.linear_epochs===100&&s.ssl_seed===0&&s.std_ddof===1);
    assert.ok(['random','kmeans'].includes(s.kind)&&eligible(s.source,s.subset)&&[1,5].includes(s.labels));
    assert.deepEqual(s.samples.map(x=>x.probe_seed),[0,1,2,3,4]);
    assert.ok(s.samples.every(x=>Number.isFinite(x.accuracy_percent)&&x.accuracy_percent>=0&&x.accuracy_percent<=100));
    assert.match(s.tex_value,/^\\dsacc\{[\d.]+\}\{[\d.]+\}$/);
    const matching=rows(text).filter(r=>r.source===s.source&&r.subset===s.subset);assert.equal(matching.length,1);
    const row=matching[0],lines=text.split('\n'),start=s.labels===1?2:7,index=start+(s.kind==='random'?0:1);
    const score=/\\(?:dsacc|bestacc)\{([\d.]+)\}\{([\d.]+)\}/;
    assert.ok(score.test(row.tokens[index]),'Unexpected existing score');
    row.tokens[index]=row.tokens[index].replace(score,s.tex_value);
    const values=row.tokens.slice(start,start+4).map(t=>{const m=t.match(score);assert.ok(m);return +m[1];});
    const best=Math.max(...values);
    for(let i=0;i<4;i++)row.tokens[start+i]=row.tokens[start+i].replace(/\\(?:dsacc|bestacc)(?=\{)/,values[i]===best?'\\bestacc':'\\dsacc');
    lines[row.lineIndex]=row.tokens.join('&');
    const tag=`% SimCLR 2k replacement: ${s.source}|ResNet18|${s.kind}|${s.subset}|simclr|${s.source}|${s.labels};`;
    const note=tag+' encoder seed=0; probe seeds=0,1,2,3,4; 100-epoch frozen probes; table.tex only; results=Evaluate_benchamrk/subset_simclr2k_1ssl5probes.';
    const previous=lines.findIndex(l=>l.startsWith(tag));if(previous>=0)lines[previous]=note;else{if(lines.at(-1)==='')lines.pop();lines.push(note,'');}
    return lines.join('\n');
}
function main(){
    const summary=path.resolve(process.argv[2]);assert.ok(summary.startsWith(path.join(__dirname,'results')+path.sep));
    const file=path.join(root,'CVPR_tables/table.tex'),old=fs.readFileSync(file,'utf8'),next=replacement(old,JSON.parse(fs.readFileSync(summary)));
    if(old!==next){const temporary=file+'.'+process.pid+'.tmp';fs.writeFileSync(temporary,next);fs.chmodSync(temporary,fs.statSync(file).mode&0o777);fs.renameSync(temporary,file);}
    console.log('TABLE UPDATED:',JSON.parse(fs.readFileSync(summary)).cell_key);
}
module.exports={rows,replacement,eligible};
if(require.main===module)main();
