const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const repo=path.resolve(__dirname,'../..');
const sources=['CIFAR10','CIFAR100','TinyImageNet'];
const slugs={cifar10:'CIFAR10',cifar100:'CIFAR100',tinyimagenet:'TinyImageNet'};
const targets={'CIFAR-10':'CIFAR10','CIFAR-100':'CIFAR100','Tiny ImageNet':'TinyImageNet',Aircraft:'Aircraft','CUB-2011':'CUB2011',Dogs:'Dogs',Flowers:'Flowers'};
const kinds={MKDT:'mkdt',Ours:'ours','Random Subset':'random','K-means Subset':'kmeans'};
const key=c=>[c.source,'ResNet18',c.kind,c.subset,c.ssl,c.target,c.labels].join('|');

function rows(relative,text){
    const name=path.basename(relative,'.tex'),parts=name.split('_'),source=slugs[parts[0]];
    const all=name==='table',main=all||parts.at(-1)==='main',split=['simclr','barlowtwins'].includes(parts.at(-1));
    const visible=text.replace(/(?<!\\)%[^\n]*/g,''),lines=text.split('\n');
    let columns;
    if(all)columns=sources.flatMap(s=>['simclr','barlowtwins'].map(ssl=>({source:s,model:'ResNet18',target:s,ssl})));
    else if(main)columns=['ConvNet','VGG11','ResNet18'].flatMap(model=>['simclr','barlowtwins'].map(ssl=>({source,model,target:source,ssl})));
    else if(split){
        const header=visible.split('\n').find(l=>/^&\s*(CIFAR-10|CIFAR-100|Tiny ImageNet|Aircraft)/.test(l));
        assert.ok(header,relative);
        columns=header.replace(/\\\\\s*$/,'').split('&').slice(1).map(t=>({source,model:'ResNet18',target:targets[t.trim()],ssl:parts.at(-1)}));
    }else columns=[...visible.split('\\midrule')[0].matchAll(/\\multicolumn\{2\}\{c\}\{([^{}]+)\}/g)]
        .flatMap(m=>['simclr','barlowtwins'].map(ssl=>({source,model:'ResNet18',target:targets[m[1]],ssl})));
    assert.ok([6,12].includes(columns.length)&&columns.every(c=>c.source&&c.target),relative);
    const result=[];let labels,subset;
    for(let i=0;i<lines.length;i++){
        if(/^\s*%/.test(lines[i]))continue;
        let m=lines[i].match(/^\s*\\multirow\{\d+\}\{\*\}\{([15])\\%\}/);if(m)labels=Number(m[1]);
        m=lines[i].match(/&\s*\\multirow\{\d+\}\{\*\}\{([.\d]+)\\%\}/);if(m)subset=Number(m[1]);
        if(!/&\s*(MKDT|Ours|Random Subset|K-means Subset)(?=\s*(?:&|$))/.test(lines[i]))continue;
        let end=i;while(!/\\\\\s*$/.test(lines[end])){end++;assert.ok(end<lines.length);}
        const text=lines.slice(i,end+1).join('\n'),hit=text.match(/&\s*(MKDT|Ours|Random Subset|K-means Subset)(?=\s*(?:&|$))/);
        const prefix=text.slice(0,hit.index+hit[0].length),tail=text.slice(prefix.length).replace(/\\\\\s*$/,'');
        let tokens=tail.split('&'),leading=tokens.shift();assert.equal(leading.trim(),'');
        const spanning=tokens.length===1&&/^\s*\\multicolumn\{\d+\}\{c\}\{\}\s*$/.test(tokens[0]);
        if(spanning)tokens=Array(columns.length).fill(' ');
        assert.equal(tokens.length,columns.length,relative);
        const cells=columns.map((c,index)=>({...c,kind:kinds[hit[1]],subset,labels,value:tokens[index].trim()}));
        result.push({start:i,end,prefix,leading,tokens,spanning,cells});i=end;
    }
    return result;
}

function applySummary(summary){
    assert.ok(summary.complete&&summary.std_ddof===1&&(summary.samples.length===6||
        summary.kind==='random'&&summary.scheme==='ten_ssl'&&summary.samples.length===10&&
        new Set(summary.samples.map(s=>s.ssl_seed)).size===10));
    const files=[...fs.readdirSync(path.join(repo,'CVPR_tables')).filter(f=>f.endsWith('.tex')).map(f=>'CVPR_tables/'+f),
        ...fs.readdirSync(path.join(repo,'CVPR_tables/full')).filter(f=>f.endsWith('.tex')).map(f=>'CVPR_tables/full/'+f)];
    for(const relative of files){
        const file=path.join(repo,relative),original=fs.readFileSync(file,'utf8'),lines=original.split('\n');
        let changed=false;
        for(const row of rows(relative,original).reverse()){
            const index=row.cells.findIndex(c=>c.model==='ResNet18'&&key(c)===summary.cell_key&&c.value==='');
            if(index<0)continue;
            const value=summary.tex_value;
            row.tokens[index]=row.tokens[index].replace(/^([ \t]*)/,'$1'+value+' ');
            const replacement=row.prefix+(row.spanning?'\n& '+row.tokens.map(t=>t.trim()).join(' & ')+' ':row.leading+'&'+row.tokens.join('&'))+'\\\\';
            lines.splice(row.start,row.end-row.start+1,replacement);changed=true;
        }
        if(!changed)continue;
        let text=lines.join('\n');
        text=text.replace(/\n?$/,'\n')+`% Probe completion: ${summary.cell_key}; SSL epochs=${summary.ssl_epochs}; encoder seeds=${summary.ssl_seeds.join(',')}; ${summary.samples.length} (SSL seed:probe seed) pairs=${summary.samples.map(s=>s.ssl_seed+':'+s.probe_seed).join(',')}.\n`;
        const temporary=file+`.${process.pid}.tmp`;fs.writeFileSync(temporary,text);fs.chmodSync(temporary,fs.statSync(file).mode&0o777);fs.renameSync(temporary,file);
        console.log('TABLE FILLED:',relative,summary.cell_key,valueForLog(summary));
    }
}
function valueForLog(s){return s.tex_value+(s.repeated_encoders?' [repeated probes]':'');}
module.exports={rows,key,applySummary};
if(require.main===module)applySummary(JSON.parse(fs.readFileSync(process.argv[2],'utf8')));
