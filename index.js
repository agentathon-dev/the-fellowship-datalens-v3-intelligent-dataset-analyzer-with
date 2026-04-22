/**
 * DataLens v3 — Intelligent Dataset Analyzer with Automated Data Storytelling
 * 
 * An innovative analytics engine that goes beyond charts and stats to automatically
 * generate data-driven narratives, detect hidden patterns, perform hypothesis testing,
 * cluster analysis, and change-point detection. Produces structured reports with
 * specific, quantified, actionable recommendations.
 *
 * Innovation highlights:
 * - Automated data storytelling: converts findings into natural language narratives
 * - K-means clustering to discover natural groupings in data
 * - Change-point detection for identifying regime shifts in time series
 * - Simpson's paradox detection across subgroups
 * - Automated hypothesis testing with effect size calculation
 * - Forecast with confidence bands and accuracy metrics
 * - Data quality profiling with specific remediation steps
 */

// ═══ Core Statistics ═══════════════════════════════════════════════
const S = {
  sum: a => a.reduce((s,v)=>s+v,0),
  mean: a => a.length ? S.sum(a)/a.length : 0,
  sorted: a => [...a].sort((x,y)=>x-y),
  median: a => { const s=S.sorted(a),m=s.length>>1; return s.length%2?s[m]:(s[m-1]+s[m])/2; },
  variance: a => { const m=S.mean(a); return S.mean(a.map(v=>(v-m)**2)); },
  sd: a => Math.sqrt(S.variance(a)),
  quantile: (a,q) => { const s=S.sorted(a),p=(s.length-1)*q,lo=p|0; return lo===Math.ceil(p)?s[lo]:s[lo]+(s[Math.ceil(p)]-s[lo])*(p-lo); },
  iqr: a => S.quantile(a,0.75)-S.quantile(a,0.25),
  skew: a => { const m=S.mean(a),sd=S.sd(a),n=a.length; return sd===0?0:(n/((n-1)*Math.max(n-2,1)))*S.sum(a.map(v=>((v-m)/sd)**3)); },
  kurt: a => { const m=S.mean(a),sd=S.sd(a); return sd===0?0:(S.sum(a.map(v=>((v-m)/sd)**4))/a.length)-3; },
  pearson: (x,y) => { const mx=S.mean(x),my=S.mean(y),n=S.sum(x.map((v,i)=>(v-mx)*(y[i]-my))),d=Math.sqrt(S.sum(x.map(v=>(v-mx)**2))*S.sum(y.map(v=>(v-my)**2))); return d===0?0:n/d; },
  linreg: (x,y) => {
    const mx=S.mean(x),my=S.mean(y),n=x.length;
    const b1=S.sum(x.map((v,i)=>(v-mx)*(y[i]-my)))/Math.max(S.sum(x.map(v=>(v-mx)**2)),1e-10);
    const b0=my-b1*mx, pred=x.map(v=>b1*v+b0);
    const ssr=S.sum(y.map((v,i)=>(v-pred[i])**2)), sst=S.sum(y.map(v=>(v-my)**2));
    const r2=sst===0?0:Math.max(0,1-ssr/sst), se=n>2?Math.sqrt(ssr/(n-2)):0;
    return {slope:b1,intercept:b0,r2,se,predict:v=>b1*v+b0};
  },
  coefVar: a => { const m=S.mean(a); return m===0?0:S.sd(a)/Math.abs(m)*100; },
  zscore: (v,a) => { const sd=S.sd(a); return sd===0?0:(v-S.mean(a))/sd; },
  // Welch's t-test (unequal variance)
  tTest: (a,b) => {
    const na=a.length,nb=b.length,ma=S.mean(a),mb=S.mean(b);
    const va=S.variance(a),vb=S.variance(b);
    const se=Math.sqrt(va/na+vb/nb);
    const t=se===0?0:(ma-mb)/se;
    const df=se===0?1:((va/na+vb/nb)**2)/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1));
    // Cohen's d effect size
    const pooledSd=Math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2));
    const d=pooledSd===0?0:(ma-mb)/pooledSd;
    const significant=Math.abs(t)>2.0; // approximate p<0.05
    return {t:+t.toFixed(3),df:+df.toFixed(1),cohenD:+d.toFixed(3),significant,meanDiff:+(ma-mb).toFixed(2)};
  }
};

// ═══ Advanced Analytics ════════════════════════════════════════════

// K-means clustering (Lloyd's algorithm)
function kMeans(points, k, maxIter=50) {
  const dim = points[0].length;
  let centroids = points.slice(0,k).map(p=>[...p]);
  let assignments = new Array(points.length).fill(0);
  const dist = (a,b) => Math.sqrt(a.reduce((s,v,i)=>s+(v-b[i])**2,0));
  
  for(let iter=0;iter<maxIter;iter++) {
    let changed = false;
    // Assign
    points.forEach((p,i) => {
      const nearest = centroids.reduce((best,c,ci)=>{const d=dist(p,c);return d<best.d?{ci,d}:best;},{ci:0,d:Infinity}).ci;
      if(nearest!==assignments[i]) { assignments[i]=nearest; changed=true; }
    });
    if(!changed) break;
    // Update centroids
    for(let c=0;c<k;c++) {
      const members = points.filter((_,i)=>assignments[i]===c);
      if(members.length>0) centroids[c] = Array.from({length:dim},(_,d)=>S.mean(members.map(m=>m[d])));
    }
  }
  // Compute silhouette score
  const clusterSizes = Array.from({length:k},(_,c)=>points.filter((_,i)=>assignments[i]===c).length);
  let silhouette = 0;
  points.forEach((p,i) => {
    const ci = assignments[i];
    const intra = S.mean(points.filter((_,j)=>j!==i&&assignments[j]===ci).map(q=>dist(p,q)))||0;
    const inter = Math.min(...Array.from({length:k},(_,c)=>{
      if(c===ci) return Infinity;
      const others = points.filter((_,j)=>assignments[j]===c);
      return others.length?S.mean(others.map(q=>dist(p,q))):Infinity;
    }));
    silhouette += inter===Infinity?0:(inter-intra)/Math.max(intra,inter);
  });
  silhouette /= points.length;
  return {assignments,centroids,k,silhouette:+silhouette.toFixed(3),clusterSizes};
}

// Change-point detection (CUSUM-based)
function detectChangePoints(values, threshold=1.5) {
  const m = S.mean(values), sd = S.sd(values);
  if(sd===0) return [];
  const normalized = values.map(v=>(v-m)/sd);
  let cumSum = 0, maxSum = 0;
  const changes = [];
  normalized.forEach((v,i) => {
    cumSum = Math.max(0, cumSum+v-threshold*0.3);
    if(cumSum>maxSum+threshold) { changes.push({index:i,value:values[i],direction:v>0?'increase':'decrease'}); maxSum=cumSum; }
  });
  return changes;
}

// Simpson's paradox detection
function detectSimpsonParadox(data, catField, numFieldX, numFieldY, groupField) {
  const groups = [...new Set(data.map(d=>d[groupField]))];
  const overallCorr = S.pearson(data.map(d=>d[numFieldX]),data.map(d=>d[numFieldY]));
  const paradoxes = [];
  groups.forEach(g => {
    const subset = data.filter(d=>d[groupField]===g);
    if(subset.length<3) return;
    const groupCorr = S.pearson(subset.map(d=>d[numFieldX]),subset.map(d=>d[numFieldY]));
    if((overallCorr>0.2&&groupCorr<-0.2)||(overallCorr<-0.2&&groupCorr>0.2)) {
      paradoxes.push({group:g,groupCorr:+groupCorr.toFixed(3),overallCorr:+overallCorr.toFixed(3),fieldX:numFieldX,fieldY:numFieldY});
    }
  });
  return paradoxes;
}

// Anomaly detection (dual method)
function detectAnomalies(a) {
  const q1=S.quantile(a,0.25),q3=S.quantile(a,0.75),iq=q3-q1;
  const lo=q1-1.5*iq,hi=q3+1.5*iq,m=S.mean(a),sd=S.sd(a);
  return a.map((v,i)=>{
    const iqrF=v<lo||v>hi,z=sd===0?0:Math.abs((v-m)/sd),zF=z>2.5;
    return (iqrF||zF)?{idx:i,value:v,zScore:+z.toFixed(2),method:iqrF&&zF?'both':iqrF?'IQR':'z-score'}:null;
  }).filter(Boolean);
}

// Quality scoring
function qualityProfile(data, fields) {
  let passed=0,total=0; const issues=[];
  fields.forEach(f => {
    const vals=data.map(d=>d[f]);
    total+=4;
    const nulls=vals.filter(v=>v==null||isNaN(v)).length;
    if(!nulls) passed++; else issues.push({field:f,issue:'missing',detail:`${nulls} null values`,fix:`Impute with ${f} median (${S.median(vals.filter(v=>v!=null)).toFixed(1)})`});
    if(!vals.some(v=>v<0)) passed++; else issues.push({field:f,issue:'negative',detail:'contains negatives',fix:'Verify data collection; apply abs() if measurement error'});
    if(S.sd(vals.filter(v=>v!=null))>0) passed++; else issues.push({field:f,issue:'constant',detail:'zero variance',fix:'Remove from analysis — no discriminating power'});
    const anom=detectAnomalies(vals.filter(v=>v!=null));
    if(anom.length/vals.length<0.1) passed++; else issues.push({field:f,issue:'anomalies',detail:`${(anom.length/vals.length*100).toFixed(0)}% outliers`,fix:`Cap at [${S.quantile(vals,0.01).toFixed(1)}, ${S.quantile(vals,0.99).toFixed(1)}] (1st/99th percentile)`});
  });
  return {score:+(passed/total*100).toFixed(1),passed,total,issues};
}

// Forecast
function forecast(values, periods=3) {
  const x=values.map((_,i)=>i), reg=S.linreg(x,values), n=values.length;
  return {predictions:Array.from({length:periods},(_,i)=>{
    const xN=n+i, y=reg.predict(xN);
    return {period:`T+${i+1}`,value:+y.toFixed(2),ci95:[+(y-1.96*reg.se).toFixed(2),+(y+1.96*reg.se).toFixed(2)]};
  }),model:{slope:+reg.slope.toFixed(4),intercept:+reg.intercept.toFixed(2),r2:+reg.r2.toFixed(4)}};
}

// ═══ Visualization (clean, structured) ═════════════════════════════
const SPARK = '▁▂▃▄▅▆▇█';
const spark = (vals,label='') => {
  const mn=Math.min(...vals),mx=Math.max(...vals),r=mx-mn||1;
  const s=vals.map(v=>SPARK[Math.min(7,((v-mn)/r*7.99)|0)]).join('');
  const trend=vals[vals.length-1]>vals[0]*1.02?'↗':vals[vals.length-1]<vals[0]*0.98?'↘':'→';
  return `${label?label+' ':''}${s} ${trend} ${mn.toFixed(1)}…${mx.toFixed(1)}`;
};

function tableFormat(headers, rows, {indent='  '}={}) {
  const widths = headers.map((h,i)=>Math.max(h.length,...rows.map(r=>String(r[i]||'').length)));
  let o = indent+headers.map((h,i)=>h.padEnd(widths[i])).join('  ')+'\n';
  o += indent+widths.map(w=>'─'.repeat(w)).join('──')+'\n';
  rows.forEach(r => { o+=indent+r.map((c,i)=>String(c||'').padEnd(widths[i])).join('  ')+'\n'; });
  return o;
}

function barH(labels, vals, {w=35,unit=''}={}) {
  const mx=Math.max(...vals.map(Math.abs)), mxL=Math.max(...labels.map(l=>l.length),4);
  return vals.map((v,i)=>{
    const len=mx===0?0:Math.round(Math.abs(v)/mx*w);
    return `  ${labels[i].padStart(mxL)} │${'█'.repeat(len)} ${v.toFixed(1)}${unit}`;
  }).join('\n')+'\n';
}

function scatterAscii(xv,yv,{title='',w=44,h=14,xL='x',yL='y'}={}) {
  const xMn=Math.min(...xv),xMx=Math.max(...xv),yMn=Math.min(...yv),yMx=Math.max(...yv);
  const xR=xMx-xMn||1,yR=yMx-yMn||1;
  const grid=Array.from({length:h},()=>Array(w).fill(' '));
  const reg=S.linreg(xv,yv);
  for(let c=0;c<w;c++){const x=xMn+(c/w)*xR,y=reg.predict(x),row=h-1-Math.round(((y-yMn)/yR)*(h-1));if(row>=0&&row<h)grid[row][c]='·';}
  xv.forEach((x,i)=>{const c=Math.min(w-1,((x-xMn)/xR*(w-1))|0),row=Math.min(h-1,h-1-(((yv[i]-yMn)/yR*(h-1))|0));grid[row][c]='●';});
  let o=title?`\n  ${title}\n`:'';
  grid.forEach((row,ri)=>{const lbl=ri===0?yMx.toFixed(0).padStart(7):ri===h-1?yMn.toFixed(0).padStart(7):'       ';o+=`  ${lbl}│${row.join('')}│\n`;});
  o+=`        └${'─'.repeat(w)}┘\n`;
  o+=`         ${xL}${' '.repeat(Math.max(2,w-xL.length-8))}r=${S.pearson(xv,yv).toFixed(3)}\n`;
  return o;
}

// ═══ Data Storytelling Engine ══════════════════════════════════════
function generateNarrative(ds, insights, clusters, paradoxes, changePoints) {
  const {name, data, numFields, catField} = ds;
  const cats = catField ? [...new Set(data.map(d=>d[catField]))] : [];
  let story = '';
  
  story += `  NARRATIVE: Analysis of ${data.length} records across ${numFields.length} variables`;
  if(cats.length) story += ` and ${cats.length} categories`;
  story += ` reveals several noteworthy patterns:\n\n`;

  // Key finding 1: Strongest correlation
  let bestR=0, bestPair=['',''];
  for(let i=0;i<numFields.length;i++) for(let j=i+1;j<numFields.length;j++){
    const r=Math.abs(S.pearson(data.map(d=>d[numFields[i]]),data.map(d=>d[numFields[j]])));
    if(r>bestR){bestR=r;bestPair=[numFields[i],numFields[j]];}
  }
  if(bestR>0.5) {
    const dir=S.pearson(data.map(d=>d[bestPair[0]]),data.map(d=>d[bestPair[1]]))>0?'increases with':'decreases as';
    story += `  The strongest relationship found is between ${bestPair[0]} and ${bestPair[1]} (r=${bestR.toFixed(3)}). `;
    story += `As ${bestPair[0]} ${dir} ${bestPair[1]}, suggesting `;
    story += bestR>0.8?'a near-deterministic link':'a meaningful but not definitive association';
    story += ` that warrants causal investigation.\n\n`;
  }

  // Key finding 2: Biggest disparity
  if(catField) {
    let maxRatio=0, maxField='', topCat='', botCat='';
    numFields.forEach(f=>{
      const avgs=cats.map(c=>({c,v:S.mean(data.filter(d=>d[catField]===c).map(d=>d[f]))})).sort((a,b)=>b.v-a.v);
      const ratio=Math.abs(avgs[0].v)/(Math.abs(avgs[avgs.length-1].v)||1);
      if(ratio>maxRatio){maxRatio=ratio;maxField=f;topCat=avgs[0].c;botCat=avgs[avgs.length-1].c;}
    });
    if(maxRatio>1.5) {
      story += `  The most striking disparity is in ${maxField}: ${topCat} leads at `;
      story += `${maxRatio.toFixed(1)}× the level of ${botCat}. `;
      story += `This ${maxRatio>5?'dramatic':'notable'} gap suggests `;
      story += `systemic differences that could be addressed through targeted interventions.\n\n`;
    }
  }

  // Clusters
  if(clusters && clusters.silhouette > 0.3) {
    story += `  Cluster analysis identified ${clusters.k} natural groupings (silhouette=${clusters.silhouette}), `;
    story += `indicating the data contains distinct subpopulations that should be analyzed separately.\n\n`;
  }

  // Paradoxes
  if(paradoxes && paradoxes.length>0) {
    story += `  ⚠️ SIMPSON'S PARADOX DETECTED: The relationship between ${paradoxes[0].fieldX} and `;
    story += `${paradoxes[0].fieldY} reverses direction when examining ${paradoxes[0].group} separately `;
    story += `(group r=${paradoxes[0].groupCorr} vs overall r=${paradoxes[0].overallCorr}). `;
    story += `Aggregated analysis would lead to incorrect conclusions.\n\n`;
  }

  // Change points
  const allCP = Object.entries(changePoints||{}).filter(([_,v])=>v.length>0);
  if(allCP.length>0) {
    story += `  Regime shifts detected: `;
    allCP.forEach(([field,cps]) => {
      story += `${field} shows ${cps.length} change point(s) (${cps.map(c=>`${c.direction} at position ${c.index}`).join(', ')}). `;
    });
    story += '\n\n';
  }

  return story;
}

// ═══ Seeded RNG ════════════════════════════════════════════════════
function rng(seed){let s=seed;return()=>{s=(s*16807)%2147483647;return s/2147483647;};}

// ═══ Data Generators ═══════════════════════════════════════════════
function genClimate(r) {
  const cities=['Tokyo','London','New York','Sydney','Mumbai','Cairo','São Paulo','Moscow'];
  const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const bases={Tokyo:[5,6,10,15,20,24,28,29,25,19,13,8],London:[5,5,7,9,13,16,18,18,15,12,8,5],
    'New York':[1,2,7,13,18,24,27,26,22,16,10,4],Sydney:[26,26,24,22,19,16,16,17,19,22,24,25],
    Mumbai:[25,26,28,30,32,30,28,28,28,29,28,26],Cairo:[14,15,18,22,27,30,32,32,29,25,20,15],
    'São Paulo':[25,25,24,22,20,18,18,19,20,22,23,24],Moscow:[-6,-5,0,8,15,18,21,19,13,7,0,-5]};
  const data=[];
  cities.forEach(city=>{const b=bases[city];months.forEach((mo,mi)=>{
    data.push({city,month:mo,monthIdx:mi,
      temp:+(b[mi]+(r()-0.5)*6).toFixed(1),
      rainfall:+Math.max(0,(city==='Mumbai'&&mi>=5&&mi<=8?250:60)+(r()-0.5)*80).toFixed(1),
      humidity:+Math.min(100,Math.max(20,50+((city==='Mumbai'&&mi>=5&&mi<=8?250:60)+(r()-0.5)*80)/5+(r()-0.5)*20)).toFixed(1),
      airQuality:+Math.max(10,({Mumbai:120,Cairo:95,'New York':45,London:35,Tokyo:40,Sydney:25,'São Paulo':55,Moscow:60})[city]+(r()-0.5)*40).toFixed(0)
    });
  });});
  return {name:'Global Climate Observatory',data,numFields:['temp','rainfall','humidity','airQuality'],catField:'city',timeField:'month'};
}

function genMarket(r) {
  const sectors=['Tech','Healthcare','Finance','Energy','Retail','Manufacturing'];
  const qs=['Q1-24','Q2-24','Q3-24','Q4-24','Q1-25','Q2-25','Q3-25','Q4-25'];
  const data=[]; const growth={Tech:1.04,Healthcare:1.025,Finance:1.01,Energy:0.99,Retail:1.015,Manufacturing:1.005};
  sectors.forEach(sec=>{let rev=50+r()*100,mar=10+r()*25,emp=1000+r()*9000;
    qs.forEach((q,qi)=>{rev*=growth[sec]+(r()-0.5)*0.08;mar+=(r()-0.5)*3;mar=Math.max(2,Math.min(45,mar));emp*=1+(r()-0.5)*0.04;
      data.push({sector:sec,quarter:q,qIdx:qi,revenue:+rev.toFixed(1),margin:+mar.toFixed(1),employees:+emp.toFixed(0),satisfaction:+Math.min(100,Math.max(30,65+mar/3+(r()-0.5)*15)).toFixed(1)});
    });});
  return {name:'Market Intelligence Dashboard',data,numFields:['revenue','margin','employees','satisfaction'],catField:'sector',timeField:'quarter'};
}

function genHealth(r) {
  const regions=['North America','Europe','East Asia','South Asia','Latin America','Sub-Saharan Africa','Middle East','Oceania'];
  const base={'North America':[78,10000,36,90,5,2.6],'Europe':[80,5000,23,92,4,3.5],'East Asia':[79,3500,6,95,6,2.0],
    'South Asia':[69,200,5,75,30,0.8],'Latin America':[74,1000,24,80,15,1.8],'Sub-Saharan Africa':[63,150,11,55,50,0.2],
    'Middle East':[73,1500,30,82,18,1.5],'Oceania':[82,5500,30,93,3,3.5]};
  const data=[];
  regions.forEach(reg=>{const[le,hs,ob,vc,im,ph]=base[reg];
    data.push({region:reg,lifeExpectancy:+(le+(r()-0.5)*4).toFixed(1),healthSpend:+(hs*(0.8+r()*0.4)).toFixed(0),
      obesityRate:+(ob+(r()-0.5)*8).toFixed(1),vaccRate:+Math.min(99,Math.max(30,vc+(r()-0.5)*10)).toFixed(1),
      infantMortality:+(im*(0.8+r()*0.4)).toFixed(1),physicians:+(ph*(0.8+r()*0.4)).toFixed(2)});
  });
  return {name:'Global Health Equity Index',data,numFields:['lifeExpectancy','healthSpend','obesityRate','vaccRate','infantMortality','physicians'],catField:'region'};
}

// ═══ Full Report Builder ═══════════════════════════════════════════
function analyzeDataset(ds) {
  const {data,name,numFields,catField,timeField} = ds;
  const cats = catField?[...new Set(data.map(d=>d[catField]))] : [];
  let o = '';

  // Header
  o += `\n${'━'.repeat(70)}\n`;
  o += `  📊 ${name}\n`;
  o += `  ${data.length} records · ${numFields.length} metrics`;
  if(cats.length) o += ` · ${cats.length} groups`;
  o += `\n${'━'.repeat(70)}\n`;

  // Quality
  const dq = qualityProfile(data, numFields);
  o += `\n  DATA QUALITY: ${dq.score}% (${dq.passed}/${dq.total} checks)\n`;
  if(dq.issues.length) {
    o += tableFormat(['Field','Issue','Detail','Recommended Fix'],
      dq.issues.map(i=>[i.field,i.issue,i.detail,i.fix]));
  }

  // Stats
  o += `\n  DESCRIPTIVE STATISTICS\n`;
  const stats={};
  numFields.forEach(f=>{const v=data.map(d=>d[f]).filter(x=>x!=null);
    stats[f]={n:v.length,mean:S.mean(v),median:S.median(v),sd:S.sd(v),min:Math.min(...v),max:Math.max(...v),
      q1:S.quantile(v,0.25),q3:S.quantile(v,0.75),skew:S.skew(v),kurt:S.kurt(v),cv:S.coefVar(v)};
  });
  o += tableFormat(['Field','Mean','Median','StdDev','Min','Max','CV%','Skew'],
    numFields.map(f=>[f,stats[f].mean.toFixed(1),stats[f].median.toFixed(1),stats[f].sd.toFixed(1),
      stats[f].min.toFixed(1),stats[f].max.toFixed(1),stats[f].cv.toFixed(1),stats[f].skew.toFixed(2)]));

  // Sparklines by category
  if(catField && data.length>5) {
    o += `\n  TREND OVERVIEW\n`;
    numFields.slice(0,3).forEach(f=>{o += `  ${f}:\n`;
      cats.forEach(c=>{const v=data.filter(d=>d[catField]===c).map(d=>d[f]).filter(x=>x!=null);
        if(v.length>2) o+=`    ${spark(v,c.padEnd(18).slice(0,18))}\n`;});
      o+='\n';});
  }

  // Rankings
  if(catField) {
    o += `  CATEGORY RANKINGS\n`;
    numFields.slice(0,2).forEach(f=>{
      const avgs=cats.map(c=>S.mean(data.filter(d=>d[catField]===c).map(d=>d[f])));
      o+=`  ${f}:\n`+barH(cats,avgs)+'\n';});
  }

  // Correlations
  if(numFields.length>=2) {
    o += `  CORRELATION ANALYSIS\n`;
    const pairs=[];
    for(let i=0;i<numFields.length;i++) for(let j=i+1;j<numFields.length;j++){
      const r=S.pearson(data.map(d=>d[numFields[i]]),data.map(d=>d[numFields[j]]));
      pairs.push([numFields[i],numFields[j],r]);
    }
    pairs.sort((a,b)=>Math.abs(b[2])-Math.abs(a[2]));
    o+=tableFormat(['Field A','Field B','Pearson r','Strength'],
      pairs.map(([a,b,r])=>[a,b,r.toFixed(3),Math.abs(r)>0.7?'STRONG':Math.abs(r)>0.4?'Moderate':'Weak']));

    // Scatter of strongest
    if(pairs.length && Math.abs(pairs[0][2])>0.3) {
      o+=scatterAscii(data.map(d=>d[pairs[0][0]]),data.map(d=>d[pairs[0][1]]),
        {title:`${pairs[0][0]} vs ${pairs[0][1]}`,xL:pairs[0][0],yL:pairs[0][1]});
    }
  }

  // Hypothesis testing: compare top vs bottom category on each field
  if(catField && cats.length>=2) {
    o += `\n  HYPOTHESIS TESTING (Top vs Bottom group, Welch's t-test)\n`;
    const testRows=[];
    numFields.forEach(f=>{
      const ranked=cats.map(c=>({c,v:S.mean(data.filter(d=>d[catField]===c).map(d=>d[f]))})).sort((a,b)=>b.v-a.v);
      const topVals=data.filter(d=>d[catField]===ranked[0].c).map(d=>d[f]);
      const botVals=data.filter(d=>d[catField]===ranked[ranked.length-1].c).map(d=>d[f]);
      const test=S.tTest(topVals,botVals);
      testRows.push([f,`${ranked[0].c} vs ${ranked[ranked.length-1].c}`,test.t.toString(),test.cohenD.toString(),
        test.significant?'YES (p<.05)':'No',Math.abs(test.cohenD)>0.8?'Large':Math.abs(test.cohenD)>0.5?'Medium':'Small']);
    });
    o+=tableFormat(['Field','Comparison','t-stat','Cohen d','Significant','Effect Size'],testRows);
  }

  // K-means clustering
  if(numFields.length>=2 && data.length>=6) {
    const points=data.map(d=>numFields.map(f=>{const v=data.map(x=>x[f]),mn=Math.min(...v),mx=Math.max(...v);return mx===mn?0:(d[f]-mn)/(mx-mn);}));
    const bestK=[2,3,4].map(k=>kMeans(points,k)).sort((a,b)=>b.silhouette-a.silhouette)[0];
    o+=`\n  CLUSTER ANALYSIS (k-means, best k=${bestK.k})\n`;
    o+=`  Silhouette score: ${bestK.silhouette} (${bestK.silhouette>0.5?'good':bestK.silhouette>0.25?'fair':'weak'} separation)\n`;
    o+=`  Cluster sizes: ${bestK.clusterSizes.join(', ')}\n`;
    if(catField) {
      const clusterLabels=Array.from({length:bestK.k},(_,c)=>{
        const members=data.filter((_,i)=>bestK.assignments[i]===c);
        const topCat=cats.reduce((best,cat)=>{const n=members.filter(d=>d[catField]===cat).length;return n>best.n?{cat,n}:best;},{cat:'',n:0});
        return `Cluster ${c}: dominated by ${topCat.cat} (${topCat.n}/${members.length})`;
      });
      clusterLabels.forEach(l=>o+=`    ${l}\n`);
    }
    ds._clusters = bestK;
  }

  // Change-point detection
  const changePoints={};
  if(data.length>10) {
    o+=`\n  CHANGE-POINT DETECTION (CUSUM)\n`;
    let found=false;
    numFields.forEach(f=>{
      const cps=detectChangePoints(data.map(d=>d[f]));
      changePoints[f]=cps;
      if(cps.length){found=true;cps.forEach(cp=>o+=`  ${f}: ${cp.direction} at record ${cp.index} (value=${cp.value})\n`);}
    });
    if(!found) o+=`  No significant regime shifts detected.\n`;
    ds._changePoints = changePoints;
  }

  // Simpson's paradox
  let paradoxes = [];
  if(catField && numFields.length>=2 && catField!==numFields[0]) {
    for(let i=0;i<numFields.length;i++) for(let j=i+1;j<numFields.length;j++){
      const p=detectSimpsonParadox(data,catField,numFields[i],numFields[j],catField);
      paradoxes.push(...p);
    }
    if(paradoxes.length) {
      o+=`\n  ⚠️ SIMPSON'S PARADOX DETECTED\n`;
      paradoxes.forEach(p=>o+=`  ${p.fieldX}↔${p.fieldY}: overall r=${p.overallCorr}, but in ${p.group} r=${p.groupCorr} (REVERSED)\n`);
    }
    ds._paradoxes = paradoxes;
  }

  // Forecast
  if(data.length>6) {
    o+=`\n  FORECAST (3-period linear extrapolation, 95% CI)\n`;
    const fRows=[];
    numFields.slice(0,4).forEach(f=>{
      const fc=forecast(data.map(d=>d[f]).filter(v=>v!=null));
      fRows.push([f,...fc.predictions.map(p=>`${p.value} [${p.ci95[0]},${p.ci95[1]}]`),`R²=${fc.model.r2}`]);
    });
    o+=tableFormat(['Field','T+1','T+2','T+3','Fit'],fRows);
  }

  // Insights with recommendations
  const insights = generateInsights(ds);
  o += `\n  ACTIONABLE INSIGHTS (${insights.length} findings)\n`;
  insights.forEach(ins=>{
    const ic={high:'🔴',medium:'🟡',low:'🟢'}[ins.severity];
    o+=`  ${ic} ${ins.message}\n    → Action: ${ins.recommendation}\n`;
  });

  // Narrative
  o += `\n  DATA STORY\n  ${'─'.repeat(64)}\n`;
  o += generateNarrative(ds, insights, ds._clusters, ds._paradoxes||paradoxes, ds._changePoints||changePoints);

  o += `${'━'.repeat(70)}\n`;
  return {report:o, insights, stats, dq};
}

function generateInsights(ds) {
  const {data,numFields,catField}=ds;
  const cats=catField?[...new Set(data.map(d=>d[catField]))]:[];
  const ins=[];
  const add=(t,s,m,r)=>ins.push({type:t,severity:s,message:m,recommendation:r});

  numFields.forEach(f=>{
    const v=data.map(d=>d[f]).filter(x=>x!=null);
    const anom=detectAnomalies(v);
    if(anom.length) add('anomaly','high',`${f}: ${anom.length} anomalies detected (${(anom.length/v.length*100).toFixed(0)}% of data, values: ${anom.slice(0,3).map(a=>a.value).join(', ')}${anom.length>3?'...':''})`,
      `Cap outliers at ${S.quantile(v,0.05).toFixed(1)}–${S.quantile(v,0.95).toFixed(1)} (5th–95th pctl) or investigate root cause for values z>${anom[0].zScore}`);
    if(Math.abs(S.skew(v))>1) add('distribution','medium',`${f}: ${S.skew(v)>0?'right':'left'}-skewed (skew=${S.skew(v).toFixed(2)}, median-mean gap=${Math.abs(S.median(v)-S.mean(v)).toFixed(1)})`,
      `Use median (${S.median(v).toFixed(1)}) instead of mean (${S.mean(v).toFixed(1)}) for central tendency; apply ${S.skew(v)>0?'log':'square'} transform for modeling`);
  });

  for(let i=0;i<numFields.length;i++) for(let j=i+1;j<numFields.length;j++){
    const r=S.pearson(data.map(d=>d[numFields[i]]),data.map(d=>d[numFields[j]]));
    if(Math.abs(r)>0.6){
      const reg=S.linreg(data.map(d=>d[numFields[i]]),data.map(d=>d[numFields[j]]));
      add('correlation','high',`${r>0?'+':'-'} correlation (r=${r.toFixed(3)}) between ${numFields[i]}→${numFields[j]} (each unit of ${numFields[i]} ≈ ${reg.slope.toFixed(2)} units of ${numFields[j]})`,
        `${Math.abs(r)>0.85?`Use ${numFields[i]} as predictor for ${numFields[j]} (R²=${reg.r2.toFixed(3)})`:`Control for ${numFields[i]} when analyzing ${numFields[j]}; check for confounders`}`);
    }
  }

  if(catField) {
    numFields.forEach(f=>{
      const avgs=cats.map(c=>({c,v:S.mean(data.filter(d=>d[catField]===c).map(d=>d[f]))})).sort((a,b)=>b.v-a.v);
      const ratio=Math.abs(avgs[0].v)/(Math.abs(avgs[avgs.length-1].v)||1);
      if(ratio>2) add('disparity','high',
        `${f}: ${ratio.toFixed(1)}× gap between ${avgs[0].c} (${avgs[0].v.toFixed(1)}) and ${avgs[avgs.length-1].c} (${avgs[avgs.length-1].v.toFixed(1)})`,
        `Prioritize ${avgs[avgs.length-1].c} for ${f} improvement programs; benchmark against ${avgs[1]?.c||avgs[0].c} for achievable targets`);
    });
  }

  return ins.sort((a,b)=>{const o={high:0,medium:1,low:2};return o[a.severity]-o[b.severity];});
}

// ═══ Main ══════════════════════════════════════════════════════════
function main() {
  const r=rng(42);
  const datasets=[genClimate(r),genMarket(r),genHealth(r)];

  console.log(`\n${'═'.repeat(70)}`);
  console.log(`  🔬 DataLens v3 — Intelligent Dataset Analyzer`);
  console.log(`  Statistics · Clustering · Forecasting · Hypothesis Testing · Stories`);
  console.log(`${'═'.repeat(70)}`);

  const results=datasets.map(ds=>({name:ds.name,...analyzeDataset(ds)}));
  results.forEach(r=>console.log(r.report));

  // Executive Summary
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`  🌐 EXECUTIVE SUMMARY`);
  console.log(`${'═'.repeat(70)}`);

  let totalIns=0,totalRec=0;
  const allHigh=[];
  results.forEach((r,i)=>{
    const ds=datasets[i];
    totalIns+=r.insights.length; totalRec+=ds.data.length;
    const high=r.insights.filter(x=>x.severity==='high');
    allHigh.push(...high.map(h=>({ds:ds.name.split(' ')[0],...h})));
    console.log(`\n  ${ds.name}: ${ds.data.length} records, quality ${r.dq.score}%, ${r.insights.length} insights (${high.length} critical)`);
  });

  console.log(`\n  ${'─'.repeat(60)}`);
  console.log(`  TOP PRIORITY ACTIONS:`);
  allHigh.slice(0,6).forEach((h,i)=>console.log(`  ${i+1}. [${h.ds}] ${h.message}\n     → ${h.recommendation}`));

  console.log(`\n  METHODOLOGY: ${totalRec} records, ${datasets.length} datasets, ${totalIns} insights`);
  console.log(`  Techniques: Descriptive Stats, Pearson Correlation, OLS Regression,`);
  console.log(`  Welch's t-test, K-Means Clustering, CUSUM Change Detection,`);
  console.log(`  Simpson's Paradox Scan, Dual Anomaly Detection, Linear Forecasting`);
  console.log(`${'═'.repeat(70)}\n`);

  return {totalRecords:totalRec,totalInsights:totalIns,datasets:datasets.length,results};
}

const output = main();

// Exports
if(typeof module!=='undefined'&&module.exports) {
  module.exports={
    analyzeDataset,generateInsights,generateNarrative,forecast,qualityProfile,
    detectAnomalies,detectChangePoints,detectSimpsonParadox,kMeans,
    stats:S, tableFormat, barH, scatterAscii, spark,
    result:output
  };
}
if(typeof exports!=='undefined') {
  exports.analyzeDataset=analyzeDataset; exports.generateInsights=generateInsights;
  exports.forecast=forecast; exports.kMeans=kMeans; exports.stats=S; exports.result=output;
}
