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
var lambda {POOL, COMP} >= 0;
var mu_pi {PROD, COMP} >= 0;
var mu_pm {PROD, POOL} >= 0; 
var v {POOL, QUAL} >= 0;
var d {p in PROD, n in QUAL} >= pqlbd[p,n], <= pqubd[p,n];
var c_M {POOL} >= 0;
var c_D {PROD} >= 0;
var f_IM {COMP, POOL} >= 0;
var f_MP {m in POOL, p in PROD} >= 0, <= yup[m,p];
var f_IP {i in COMP, p in PROD} >= 0, <= zup[i,p];
var t_I {i in COMP} >= complb[i], <= compub[i];
var t_M {m in POOL} >= 0, <= psize[m];

var slack {COMP, POOL} >= 0;


#objective function
minimize Cost: sum{p in PROD} c_D[p]*produb[p];

subject to Link {i in COMP, m in POOL}:
	lambda[m,i] + slack[i,m] = qup[i,m];

subject to PoolQuality {m in POOL, n in QUAL}:
	v[m,n] = sum{i in COMP} lambda[m,i]*cqual[i,n];

subject to ProductQuality {p in PROD, n in QUAL}:
	d[p,n] = sum{m in POOL} mu_pm[p,m]*v[m,n] + sum{i in COMP} mu_pi[p,i]*cqual[i,n];
	
subject to convexity_m {m in POOL}:
	sum{i in COMP} lambda[m,i] = 1;

subject to convexity_d {p in PROD}:
	sum{i in COMP} mu_pi[p,i] + sum{m in POOL} mu_pm[p,m] = 1;
	
subject to PricePools {m in POOL}:
	c_M[m] = sum{i in COMP} lambda[m,i]*cprice[i];

subject to PriceProducts {p in PROD}:
	c_D[p] = sum{i in COMP}mu_pi[p,i]*cprice[i] + sum{m in POOL} mu_pm[p,m]*c_M[m];

subject to FlowPoolProducts {m in POOL, p in PROD}:
	f_MP[m,p] = mu_pm[p,m]*produb[p];

subject to PoolCapacity {m in POOL}:
	t_M[m] = sum{p in PROD} f_MP[m,p];
	
subject to FlowComponentPool {i in COMP, m in POOL}:
	f_IM[i,m] = lambda[m,i]*t_M[m];

subject to FlowComponentProduct {i in COMP, p in PROD}:
	f_IP[i,p] = mu_pi[p,i]*produb[p];

subject to ComponentCapacity {i in COMP}:
	t_I[i] = sum{m in POOL} f_IM[i,m] + sum{p in PROD} f_IP[i,p];
