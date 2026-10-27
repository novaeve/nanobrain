const primary={annotation:'f1_macro',integration:'integration_composite',development:'population_mse',spatial_niche:'ari'};
const taskLabel={annotation:'Annotation',integration:'Integration',development:'Development',spatial_niche:'Spatial niche'};
const metricLabel={f1_macro:'Macro F1',integration_composite:'Integration composite',population_mse:'Population MSE',ari:'ARI'};
const familyColor={classical:'#9aa3b2',foundation:'#9c82df',deep:'#5ab7ad',ours:'#557fe6',submitted:'#d89a62'};
const taskOrder=['annotation','integration','development','spatial_niche'];
let catalog=(window.__NB_CATALOG__||[]).map((r,i)=>Object.assign({_id:String(i)},r));
let leaderboardTask='all',frontierTask='annotation',query='';

const fmt=v=>Number.isFinite(Number(v))?Number(v).toFixed(4):'—';
const num=v=>Number.isFinite(Number(v))?Number(v):null;
const isMax=task=>task!=='development';
const paramOf=r=>{const n=num(r.parameter_count);return n&&n>0?n:null;};
const esc=v=>String(v==null?'':v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const displayTask=t=>taskLabel[t]||t;
const scoreText=r=>fmt(r.mean);
const metadata=r=>r.display_name+' · '+displayTask(r.task)+' · '+(metricLabel[r.metric]||r.metric)+' · score '+scoreText(r)+' · '+(paramOf(r)?paramOf(r).toLocaleString()+' parameters':'parameter count unavailable')+' · '+(r.training_regime||'benchmark');

function flow(r){
  const pipeline=r.architecture_mermaid||r.architecture_summary||'benchmark architecture';
  return '<button class="popover-close" type="button" aria-label="Close details">×</button>'+
    '<div class="popover-title">'+esc(r.display_name)+'</div>'+
    '<div class="popover-meta">'+esc(metadata(r))+'</div>'+
    '<div class="flow"><span>Input<br><b>expression matrix</b></span><i>→</i>'+
    '<span>Architecture<br><b>'+esc(r.architecture_summary||'benchmark architecture')+'</b><small>'+esc(pipeline)+'</small></span><i>→</i>'+
    '<span>Output<br><b>'+displayTask(r.task)+' · '+esc(metricLabel[r.metric]||r.metric)+'</b></span></div>'+
    '<div class="popover-source">'+esc(r.receipt_source||r.metrics_source||r.notes||'benchmark record')+'</div>';
}
function svgEl(tag,attrs={},text=''){
  const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
  Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
  if(text)e.textContent=text;
  return e;
}
function svgBase(){
  const el=document.querySelector('#full-comparison-chart');el.innerHTML='';
  const svg=svgEl('svg',{'viewBox':'0 0 1180 520','role':'img','aria-label':'Model parameter count versus benchmark performance'});
  el.appendChild(svg);return svg;
}
function placePopover(p,x,y){
  const w=Math.min(390,window.innerWidth-20);
  p.style.width=w+'px';
  p.style.left=Math.min(Math.max(x,10),window.innerWidth-w-10)+'px';
  p.style.top=Math.max(Math.min(y,window.innerHeight-260),10)+'px';
}
function showHover(r,x,y){
  const p=document.querySelector('#point-popover');
  p.classList.remove('is-pinned');p.innerHTML=flow(r);p.hidden=false;placePopover(p,x,y);
  p.querySelector('.popover-close').hidden=true;
}
function showPinned(r,x,y){
  const p=document.querySelector('#point-popover');
  p.classList.add('is-pinned');p.innerHTML=flow(r);p.hidden=false;placePopover(p,x,y);
  p.querySelector('.popover-close').hidden=false;
  p.querySelector('.popover-close').addEventListener('click',()=>{p.hidden=true;p.classList.remove('is-pinned');});
}
function hideHover(){
  const p=document.querySelector('#point-popover');
  if(!p.classList.contains('is-pinned'))p.hidden=true;
}
function compareScore(a,b,task){
  const d=isMax(task)?-1:1;
  return (num(a.mean)===null?1:num(b.mean)===null?-1:d*(num(a.mean)-num(b.mean)))||
    String(a.display_name).localeCompare(String(b.display_name));
}
function rowSearchText(r){
  return [r.display_name,r.method,r.method_family,r.training_regime,r.notes,r.task,r.metric].filter(Boolean).join(' ').toLowerCase();
}
function rowHtml(r,rank){
  const sd=num(r.std), params=paramOf(r);
  return '<tr class="leaderboard-row" data-row-id="'+esc(r._id)+'" tabindex="0">'+
    '<td>'+rank+'</td><td><span class="model-name">'+esc(r.display_name)+'</span><span class="model-sub">'+esc((r.method_family||'benchmark')+' · '+(r.training_regime||'benchmark'))+'</span></td>'+
    '<td><span class="task-chip">'+esc(displayTask(r.task))+'</span></td><td>'+esc(metricLabel[r.metric]||r.metric)+'</td>'+
    '<td class="score">'+scoreText(r)+(params?'':'<span class="model-sub">params N/A</span>')+'</td><td>'+fmt(sd)+'</td><td>'+esc(r.n_runs==null?'—':r.n_runs)+'</td>'+
    '<td><span class="status-pill">'+(r.status==='complete'?'verified':esc(r.status||'recorded'))+'</span></td></tr>';
}
function renderLeaderboard(){
  const body=document.querySelector('#leaderboard-body');
  const q=query;
  const groups=leaderboardTask==='all'?taskOrder:[leaderboardTask];
  let html='';
  groups.forEach(task=>{
    const rows=catalog.filter(r=>r.task===task&&(!q||rowSearchText(r).includes(q))).sort((a,b)=>compareScore(a,b,task));
    rows.forEach((r,i)=>{html+=rowHtml(r,i+1);});
  });
  body.innerHTML=html||'<tr><td colspan="8" class="empty-table">No benchmark rows match this task or search.</td></tr>';
  body.querySelectorAll('.leaderboard-row').forEach(row=>{
    const r=catalog.find(x=>x._id===row.dataset.rowId);
    const open=()=>{const b=row.getBoundingClientRect();showPinned(r,Math.min(b.right+12,window.innerWidth-390),b.top+8);};
    row.addEventListener('click',open);row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});
  });
  document.querySelectorAll('.segmented button').forEach(b=>{
    const active=b.dataset.task===leaderboardTask;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));
  });
}
function frontierPoints(known){
  const sorted=[...known].sort((a,b)=>a.__xRaw-b.__xRaw||(isMax(frontierTask)?num(b.mean)-num(a.mean):num(a.mean)-num(b.mean)));
  const out=[];let best=null;
  sorted.forEach(r=>{const improves=best===null||(isMax(frontierTask)?num(r.mean)>best:num(r.mean)<best);if(improves){out.push(r);best=num(r.mean);}});
  return out;
}
function renderFrontier(){
  const data=catalog.filter(r=>r.task===frontierTask&&r.metric===primary[frontierTask]&&num(r.mean)!==null);
  const svg=svgBase();
  if(!data.length){svg.appendChild(svgEl('text',{x:590,y:250,'text-anchor':'middle',fill:'#8d96a3','font-size':'13'},'No benchmark data for this task.'));return;}
  const W=1180,H=520,pad={l:92,r:28,t:35,b:82};
  const known=data.filter(r=>paramOf(r)),unknown=data.filter(r=>!paramOf(r));
  if(!known.length){svg.appendChild(svgEl('text',{x:590,y:250,'text-anchor':'middle',fill:'#8d96a3','font-size':'13'},'No verified parameter counts for this task.'));return;}
  const minP=Math.min(...known.map(r=>paramOf(r))),maxP=Math.max(...known.map(r=>paramOf(r)));
  const scores=data.map(r=>num(r.mean)),minRaw=Math.min(...scores),maxRaw=Math.max(...scores);
  let minS=minRaw,maxS=maxRaw;if(minS===maxS){minS-=1;maxS+=1;}
  const range=Math.max(1e-12,maxS-minS),yPad=range*.08;minS-=yPad;maxS+=yPad;
  const y=v=>pad.t+(isMax(frontierTask)?(maxS-v):(v-minS))/(maxS-minS)*(H-pad.t-pad.b);
  const knownStart=pad.l+86,knownEnd=W-pad.r;
  const xKnown=p=>knownStart+(Math.log10(p)-Math.log10(minP))/Math.max(1e-9,Math.log10(maxP)-Math.log10(minP))*(knownEnd-knownStart);
  const unknownX=pad.l+42;
  const groups=new Map();known.forEach(r=>{const p=paramOf(r);if(!groups.has(p))groups.set(p,[]);groups.get(p).push(r);});
  groups.forEach((rs,p)=>rs.forEach((r,i)=>{const raw=xKnown(p),offset=(i-(rs.length-1)/2)*9;r.__xRaw=raw;r.__x=raw+offset;r.__y=y(num(r.mean));}));
  const unknownJitter=new Map();unknown.forEach((r,i)=>{unknownJitter.set(r,unknownX+((i%7)-3)*9);r.__x=unknownJitter.get(r);r.__y=y(num(r.mean));r.__xRaw=unknownX;});
  for(let i=0;i<5;i++){
    const gy=pad.t+i*(H-pad.t-pad.b)/4;
    svg.appendChild(svgEl('line',{x1:pad.l,x2:W-pad.r,y1:gy,y2:gy,stroke:'#e8ebf0'}));
    const val=isMax(frontierTask)?maxS-(maxS-minS)*i/4:minS+(maxS-minS)*i/4;
    svg.appendChild(svgEl('text',{x:pad.l-12,y:gy+4,'text-anchor':'end',fill:'#8d96a3','font-size':'11'},fmt(val)));
  }
  svg.appendChild(svgEl('line',{x1:pad.l,x2:pad.l,y1:pad.t,y2:H-pad.b,stroke:'#cfd5df'}));
  svg.appendChild(svgEl('line',{x1:pad.l,x2:W-pad.r,y1:H-pad.b,y2:H-pad.b,stroke:'#cfd5df'}));
  svg.appendChild(svgEl('rect',{x:pad.l+8,y:pad.t+8,width:68,height:H-pad.t-pad.b-16,rx:10,fill:'#f7f8fa'}));
  svg.appendChild(svgEl('text',{x:unknownX,y:H-pad.b+24,'text-anchor':'middle',fill:'#8d96a3','font-size':'10'},'N/A params'));
  svg.appendChild(svgEl('text',{x:(knownStart+knownEnd)/2,y:H-22,'text-anchor':'middle',fill:'#7c8593','font-size':'11'},'Model parameters (log scale)'));
  svg.appendChild(svgEl('text',{x:pad.l-58,y:pad.t-12,fill:'#7c8593','font-size':'11'},(metricLabel[primary[frontierTask]]||primary[frontierTask])+' · '+(isMax(frontierTask)?'higher is better':'lower is better')));
  [minP,maxP].forEach(p=>svg.appendChild(svgEl('text',{x:xKnown(p),y:H-pad.b+24,'text-anchor':'middle',fill:'#8d96a3','font-size':'10'},p.toLocaleString())));
  const front=frontierPoints(known);
  if(front.length>1){
    const d=front.map((r,i)=>(i?'L':'M')+' '+r.__x.toFixed(1)+' '+r.__y.toFixed(1)).join(' ');
    svg.appendChild(svgEl('path',{d:d,fill:'none',stroke:'#557fe6','stroke-width':'3','stroke-linecap':'round','stroke-linejoin':'round',opacity:'.9'}));
    front.forEach(r=>svg.appendChild(svgEl('circle',{cx:r.__x,cy:r.__y,r:12,fill:'none',stroke:'#557fe6','stroke-width':'1.5','stroke-dasharray':'3 3',opacity:'.8'})));
  }
  data.forEach(r=>{
    const c=svgEl('circle',{cx:r.__x,cy:r.__y,r:paramOf(r)?8:6,fill:familyColor[r.method_family]||'#9aa3b2',stroke:'#fff','stroke-width':'2',tabindex:'0','aria-label':metadata(r)});
    c.style.cursor='pointer';c.addEventListener('mouseenter',e=>showHover(r,e.clientX,e.clientY));c.addEventListener('mouseleave',hideHover);c.addEventListener('focus',()=>showHover(r,Math.min(window.innerWidth-400,650),120));c.addEventListener('blur',hideHover);
    c.addEventListener('click',e=>{e.stopPropagation();showPinned(r,e.clientX,e.clientY);});c.appendChild(svgEl('title',{},metadata(r)));svg.appendChild(c);
  });
  svg.appendChild(svgEl('line',{x1:knownStart-1,x2:knownStart-1,y1:pad.t,y2:H-pad.b,stroke:'#d9dee7','stroke-dasharray':'4 5'}));
  svg.appendChild(svgEl('text',{x:knownStart+8,y:pad.t+16,fill:'#557fe6','font-size':'10'},'frontier · '+front.length+' points'));
  svg.appendChild(svgEl('text',{x:pad.l+10,y:pad.t+25,fill:'#8d96a3','font-size':'10'},unknown.length+' results without verified parameter count'));
  svg.appendChild(svgEl('text',{x:pad.l,y:H-5,fill:'#8d96a3','font-size':'10'},data.length+' total results · every benchmark method shown'));
}
function setLeaderboardTask(task){
  leaderboardTask=task;renderLeaderboard();
  const target=document.querySelector('#leaderboard');if(target)target.scrollIntoView({block:'start',behavior:'smooth'});
}
document.querySelectorAll('.segmented button').forEach(b=>{b.addEventListener('click',()=>setLeaderboardTask(b.dataset.task));});
const search=document.querySelector('#search');if(search)search.addEventListener('input',e=>{query=e.target.value.trim().toLowerCase();renderLeaderboard();});
const selector=document.querySelector('#frontier-task');if(selector)selector.addEventListener('change',e=>{frontierTask=e.target.value;renderFrontier();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){const p=document.querySelector('#point-popover');p.hidden=true;p.classList.remove('is-pinned');}});
document.querySelector('#run-count').textContent=catalog.length;
renderLeaderboard();renderFrontier();
