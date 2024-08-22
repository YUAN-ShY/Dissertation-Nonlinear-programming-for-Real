# set
set NU;
set RM;
set MI;
set DE;

# parameter
param nuinrm{RM, NU};    # Nutrient content of the raw materials
param specs_lb{DE, NU};  # lower bound on nutrient in demand
param specs_ub{DE, NU};  # upper bound on nutrient in demand
param RMcost{RM};        # Cost of raw-materials (per unit)
param RMcostStr{RM};     # Cost if use as straight
param tons{DE};          # Tonnages of demand

param rmmi_lb{MI, RM} default 0.0; # lower bnd on raw material content in mixes
param rmmi_ub{MI, RM} default 1.0; # upper bnd on raw material content in mixes

param rmsde_lb{DE, RM} default 0.0; # lower bnd on straight content in demands
param rmsde_ub{DE, RM} default 0.0; # upper bnd on straight content in demands

param mide_lb{DE, MI} default 0.0; # lower bnd on mixer content in demands
param mide_ub{DE, MI} default 1.0; # upper bnd on mixer content in demands

param numi_lb{MI, NU} default 0.0;   # lower bound on nutrient content in mixes
param numi_ub{MI, NU} default Infinity; # upp bnd on nutrient content in mixes


# variable
var c_D {DE} >= 0;
var c_M {MI} >= 0;

var v {m in MI, n in NU} >= numi_lb[m,n], <= numi_ub[m,n];
var d {p in DE, n in NU} >= specs_lb[p,n], <= specs_ub[p,n];

var lambda {m in MI, i in RM} >= rmmi_lb[m,i], <= rmmi_ub[m,i];
var mu_pi {p in DE, i in RM} >= rmsde_lb[p,i], <= rmsde_ub[p,i];
var mu_pm {p in DE, m in MI} >= mide_lb[p,m], <= mide_ub[p,m]; 



# objective function
minimize Cost: sum{p in DE} c_D[p]*tons[p];

subject to PoolQuality {m in MI, n in NU}:
	v[m,n] = sum{i in RM} lambda[m,i]*nuinrm[i,n];

subject to ProductQuality {p in DE, n in NU}:
	d[p,n] = sum{m in MI} mu_pm[p,m]*v[m,n] + sum{i in RM} mu_pi[p,i]*nuinrm[i,n];
	
subject to convexity_m {m in MI}:
	sum{i in RM} lambda[m,i] = 1;

subject to convexity_d {p in DE}:
	sum{i in RM} mu_pi[p,i] + sum{m in MI} mu_pm[p,m] = 1;
	
subject to PricePools {m in MI}:
	c_M[m] = sum{i in RM} lambda[m,i]*RMcost[i];

subject to PriceProducts {p in DE}:
	c_D[p] = sum{i in RM} mu_pi[p,i]*RMcostStr[i] + sum{m in MI} mu_pm[p,m]*c_M[m];
