import SQP_Filter as Filter
from scipy.optimize import linprog
import numpy as np

out = 0

# check primal feasibility
def is_PrimalFeasibility(nlp, x):
    tol = 1e-6

    if Filter.eval_cviol(nlp, x) < tol:
        return True
    else:
        return False

# check dual feasibility and complementarity
def is_DualComplementarity(nlp, x):
    tol = 1e-6

    nvar = nlp.nvar
    ncon = nlp.ncon

    # bl <= x <= bu     
    # lambda <= 0
    lam_1_bound = [(None, 0)] * nvar
    lam_2_bound = [(None, 0)] * nvar

    # for not active g(x) set the lambda = 0
    for ixc in range(nvar): 
        if abs(x[ixc]-nlp.bu[ixc]) > tol:
            lam_1_bound[ixc] = (0, 0)
        
        if abs(nlp.bl[ixc]-x[ixc]) > tol:
            lam_2_bound[ixc] = (0, 0)

    # cl = h(x) = cu     
    # mu is free
    mu_bound = [(None, None)] * ncon

    bound = lam_1_bound + lam_2_bound + mu_bound

    A = np.vstack((np.identity(nvar), -np.identity(nvar)))

    for ixc in range(ncon):
        grad_h = nlp.eval_consgrad(ixc, x)
        A = np.vstack((A, grad_h))

    b = nlp.eval_objgrad(x)
    c = np.zeros(len(bound)) 

    res = linprog(c, A_eq=A.transpose(), b_eq=b, bounds=bound, method='highs-ipm')

    if res.success:
        return True
    else:
        return False


# check KKT condition
def is_KKT(nlp, x):
    if is_DualComplementarity(nlp, x) and is_PrimalFeasibility(nlp, x):
        return True
    return False