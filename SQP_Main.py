import numpy as np
import numpy.linalg as LA
import sys
import math

import QP
import IPM
import SQP_Filter as Filter
import SQP_QuasiNewton as QN

# SQP main phase process
def do_SQP(xk, nlp, approxi_L, update_QN, max_iter, do_QuasiNewton_logic, rho, tol, out):
    # 0: basic d and Filter
    # 1: Lagrangian matrix and multiplier

    flag = False
    filter_point = []

    if do_QuasiNewton_logic:
        f_Hessian = np.identity(nlp.nvar)
        c_Hessian = np.array([np.identity(nlp.nvar) for _ in range(nlp.ncon)])
        Q = np.identity(nlp.nvar)

    miu = np.zeros(nlp.ncon)

    # object value at current point 
    objk = nlp.obj(xk)
    # total constraint violation at current point
    cviolk = Filter.eval_cviol(nlp, xk)


    if out==0:
        print(f"{'Objective Value':<20}{'Constraints Violation':<25}{'Accept by Filter':<20}{'Radians':<25}{'|d|inf:':<10}")
        print(f"{objk:<20.5f}{cviolk:<25.5f}{'NUll':<20}{rho:<25}{'NUll':<10}")


    for iter in range(max_iter):
        c = nlp.grad(xk) # gradient of objective
        bl = nlp.bl - xk
        bu = nlp.bu - xk
        A = nlp.jac(xk).transpose() # Jacobian of constraints
        b = nlp.cl - nlp.cons(xk) 

        # get Q => Hessian matrix of Lagrangian 
        if do_QuasiNewton_logic:
            try:
                d

                if approxi_L == 'Separate':
                    Q, c_Hessian, f_Hessian = QN.Lagrangian_Hessian_S(nlp, xk, d, miu, c_Hessian, f_Hessian, update_QN)

                elif approxi_L == 'Integrate':
                    Q = QN.Lagrangian_Hessian_I(nlp, xk, d, miu, Q, update_QN)

                else:
                    print('did not recognise approximation')
                    sys.exit(1)

            except NameError:
                pass

        else:
            Q = nlp.hess(xk, -miu) 


        # impose Trust Region bounds:  -rho <= d_i <= rho
        bu = np.minimum(bu, rho) 
        bl = np.maximum(bl, -rho) 


        # solve QP
        qp = QP.QP(c, Q, A, b, bl, bu)
        ipm = IPM.IPM(qp)
        status = ipm.solve()

        if status == ipm.INFEASIBLE:
            if out==0:
                print('Infeasible SQP subproblem, start Phase One process')
            return xk, flag, iter+1
    
        if status == ipm.UNBOUNDED:
            print('Unbounded QP problem')
            return xk, flag, iter+1
            # exit(1)

        d = ipm.getSol()


        xkp = xk+d
        objkp = nlp.obj(xkp)
        cviolkp = Filter.eval_cviol(nlp, xkp)
        Filter_Improve = Filter.is_improvement(objkp, cviolkp, filter_point)

        
        if out==0:
                print(f"{objkp:<20.5f}{cviolkp:<25.5f}{'Yes' if Filter_Improve else 'No':<20}{rho:<25}{LA.norm(d,np.inf):<10.10f}")


        # update Lagrangian multiplier
        miu = ipm.getDualsCons()

        # check if xk+d is a "better point" => update x
        if Filter_Improve:
            # test if step is small enough => then stop iteration
            if LA.norm(d)<tol:
                flag = True
                if out==0:
                    print('converge with', iter+1, 'iteration')
                return xk, flag, iter+1
            
            xk += d
            rho *= 2
            # rho = min(rho*4, math.ceil(LA.norm(d, np.inf)))   
            filter_point = Filter.filter_update(filter_point, [objkp, cviolkp])
                
        # not "better point" 
        # resolve subproblem with smaller region
        else:

            rho = min(rho/4, math.ceil(LA.norm(d)))
            # rho = min(rho/2, math.ceil(LA.norm(d, np.inf)))
            del d


    if out==0:
        print('Max interation of SQP')


    flag = True
    return xk, flag, max_iter