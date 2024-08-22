#sets
set COMP;
set QUAL;
set PROD;
set POOL;

#parameter 
param complb {COMP}; 
param compub {COMP};
param cprice {COMP};

param cqual {COMP, QUAL};

param psize {POOL};

param prodlb {PROD};
param produb {PROD}; 
param pprice {PROD}; 

param pqlbd {PROD, QUAL}; 
param pqubd {PROD, QUAL}; 

param qup {COMP, POOL};

param yup {POOL, PROD};

param zup {COMP, PROD};


#variables
var q {COMP, POOL} >= 0;
var y {POOL, PROD} >= 0;
var z {COMP, PROD} >= 0;

#slack variables
var s_PQUB {p in PROD, n in QUAL} >= 0;
var s_PQLB {p in PROD, n in QUAL} >= 0;
var s_PPUB {p in PROD} >= 0;
var s_PPLB {p in PROD} >= 0;
var s_CCUB {i in COMP} >= 0;
var s_CCLB {i in COMP} >= 0;
var s_SLFC {i in COMP, p in PROD} >= 0;
var s_PSC {m in POOL} >= 0;
var s_PPFC {m in POOL, p in PROD} >= 0;

#objective function
minimize Total_Cost: sum {i in COMP, m in POOL, p in PROD} cprice[i]*(qup[i,m]*q[i,m]*y[m,p]+z[i,p]) - sum {m in POOL, p in PROD} pprice[p]*y[m,p] - sum {i in COMP, p in PROD} pprice[p]*z[i,p];

#constraints
subject to Cuts {m in POOL}:
	sum {i in COMP} qup[i,m]*q[i,m] = 1;

subject to ProductQualityUpperBound {p in PROD, n in QUAL}:
	sum {m in POOL, i in COMP} cqual[i,n]*qup[i,m]*q[i,m]*y[m,p] + sum {i in COMP} cqual[i,n]*z[i,p] + s_PQUB[p,n]= pqubd[p,n]*(sum {m in POOL} y[m,p] + sum {i in COMP} z[i,p]);

subject to ProductQualityLowerBound {p in PROD, n in QUAL}:
	sum {m in POOL, i in COMP} cqual[i,n]*qup[i,m]*q[i,m]*y[m,p] + sum {i in COMP} cqual[i,n]*z[i,p] = s_PQLB[p,n] + pqlbd[p,n]*(sum {m in POOL} y[m,p] + sum {i in COMP} z[i,p]);
	
subject to ProductProduceUpperBound {p in PROD}:
	sum {m in POOL} y[m,p] + sum {i in COMP} z[i,p] + s_PPUB[p] = produb[p];

subject to ProductProduceLowerBound {p in PROD}:
	sum {m in POOL} y[m,p] + sum {i in COMP} z[i,p] = s_PPLB[p] + prodlb[p];

subject to ComponentCapacityUpperBound {i in COMP}:
	sum {m in POOL, p in PROD} qup[i,m]*q[i,m]*y[m,p] + sum {p in PROD} z[i,p] + s_CCUB[i] = compub[i];
	
subject to ComponentCapacityLowerBound {i in COMP}:
	sum {m in POOL, p in PROD} qup[i,m]*q[i,m]*y[m,p] + sum {p in PROD} z[i,p] = s_CCLB[i] + complb[i];
	
subject to StraightLinkFlowCapactity {i in COMP, p in PROD}:
	z[i,p] +  s_SLFC[i,p] = zup[i,p];
	
subject to PoolSizeCapactity {m in POOL}:
	sum {p in PROD} y[m,p] + s_PSC[m] = psize[m];
	
subject to PoolProductFlowCapactity {m in POOL, p in PROD}:
	y[m,p] + s_PPFC[m,p] = yup[m,p];