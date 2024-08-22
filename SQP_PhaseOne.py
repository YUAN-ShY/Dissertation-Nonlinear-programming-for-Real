import numpy as np
import QP
import IPM
import sys
import numpy.linalg as LA
import math

# calculate the value of merit function at x
def merit_function(nlp, x, gama):
    # func = nlp.obj(x)
    func = 0
    c = nlp.cons(x)

    # violation of equality constraints
    func += gama * np.sum(np.maximum(0, c-nlp.cu) + np.maximum(0, nlp.cl-c))

    # violation of inequality constraints on variables
    func += gama * np.sum(np.maximum(0, x-nlp.bu) + np.maximum(0, nlp.bl-x))

    return func


# check imporvement
def is_merit_improvement(nlp, x, d, gama):  
    return merit_function(nlp, x+d, gama) < merit_function(nlp, x, gama)


# solve the phase one subproblem
def solve_Ph1QP(nlp, x, rho):
    # new variables vector v = [sp^T sm^T d^t]
    ncon = nlp.ncon
    nvar = nlp.nvar
    size_v = 2*ncon+nvar

    c_QP = np.ones(size_v)
    c_QP[2*ncon:] = 0

    Q_QP = np.zeros((size_v, size_v))

    bu = np.minimum(nlp.bu-x, rho)
    bl = np.maximum(nlp.bl-x, -rho)
    bu_QP = np.block([np.full(2*ncon, np.inf), bu])
    bl_QP = np.block([np.zeros(2*ncon), bl])

    b_QP = nlp.cl-nlp.cons(x)

    A_QP = np.block([np.identity(ncon), -np.identity(ncon), nlp.jac(x).transpose()])

    qp = QP.QP(c_QP, Q_QP, A_QP, b_QP, bl_QP, bu_QP)
    ipm = IPM.IPM(qp)
    status = ipm.solve()

    if status == ipm.INFEASIBLE:
        print('Infeasible Phase one QP subproblem')
        sys.exit
    if status == ipm.UNBOUNDED:
        print('Unbounded Phase one QP subproblem')
        sys.exit

    v = ipm.getSol()

    return v


# Phase One main program
def Ph1(nlp, x, gama, rhoPh1, tol, max_iter, out):

    ncon = nlp.ncon
    nvar = nlp.nvar

    # indication find the solution or not
    flag = False

    if out==0:
        print(f"{'Current Iteration':<20}{'Merit Function value':<25}{'Accept by Filter':<20}{'Radians':<25}{'|d|inf:':<10}")

    for iter in range(max_iter):
        v = solve_Ph1QP(nlp, x, rhoPh1)

        sp = v[:ncon]
        sm = v[ncon:2*ncon]
        d = v[(2*ncon):]
        #merit_function_improve = is_merit_improvement(nlp, x, d, gama)
        old_merit =  merit_function(nlp, x, gama)
        new_merit =  merit_function(nlp, x+d, gama)
        pred_dec = old_merit
        actual_dec = old_merit-new_merit
        merit_function_improve = new_merit<old_merit

        if out==0:
            print(f"{iter:<20.2f}{merit_function(nlp, x, gama):<25.2f}{'Yes' if merit_function_improve else 'No':<20}{rhoPh1:<25}{LA.norm(d,np.inf):<10.8f}")
            
        if np.all(sp<tol) and np.all(sm<tol):
            flag = True
            return x, flag, math.ceil(LA.norm(d)), iter

        if new_merit<old_merit:
            x += d
            if (actual_dec/pred_dec>0.75):
                rhoPh1 *= 2

            
        else:
            rhoPh1 = min(rhoPh1/4, math.ceil(LA.norm(d)))
            # rhoPh1 /= 4
    
    return x, flag, math.ceil(LA.norm(d)), iter
