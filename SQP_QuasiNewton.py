import numpy as np
import sys

# update single approximation of Hessian matrix
def update_QN(Bk, delta, gkp, gk, update):
    y = gkp - gk
    dy = delta.dot(y)

    if np.all(y == 0):
        row, col = Bk.shape
        Bkp = np.zeros((row, col))
        return Bkp
    
    else:
        # if dy < 0:
        #     print('Quasi Newton update negative vector product')

        if update =="BFGS":
            # BFGS update
            Bd = Bk.dot(delta)
            Bkp = Bk + np.outer(y,y)/dy - np.outer(Bd,Bd)/(Bd.dot(delta))

        elif update == "DFP":
            # DFP update
            Bd = Bk.dot(delta) 
            Bkp = Bk + (1+y.dot(Bd)/dy)*(np.outer(y, y))/dy  \
                - (np.outer(delta, Bd)+np.outer(Bd, y))/dy
        
        else:
            print("did not recognise update")
            sys.exit(1)
    
    return Bkp


# calculate the Lagrangian Hessian
# update the Hessian approximation of constraints and objective function 
def Lagrangian_Hessian_S(nlp, xkp, d, miu, c_Hessian, f_Hessian, update):
    nvar = nlp.nvar
    ncon = nlp.ncon

    xk = xkp - d
    L_Hessian = np.zeros((nvar, nvar))

    for ixc in range(ncon):
        gk = nlp.eval_consgrad(ixc, xk)
        gkp = nlp.eval_consgrad(ixc, xkp)
        c_Hessian[ixc] = update_QN(c_Hessian[ixc], d, gkp, gk, update)
        L_Hessian += miu[ixc]*c_Hessian[ixc]

    gk = nlp.grad(xk)
    gkp = nlp.grad(xkp) 
    f_Hessian = update_QN(f_Hessian, d, gkp, gk, update)
    L_Hessian += f_Hessian

    return L_Hessian, c_Hessian, f_Hessian


# calculate the approximation of Lagrangian Hessian
def Lagrangian_Hessian_I(nlp, xkp, d, miu, L_Hessian, update):
    xk = xkp - d
    gradLk = nlp.grad(xk)
    gradLkp = nlp.grad(xkp)

    for ixc in range(nlp.ncon):
        gradLk += miu[ixc]*nlp.eval_consgrad(ixc, xk)
        gradLkp += miu[ixc]*nlp.eval_consgrad(ixc, xkp)

    L_Hessian = update_QN(L_Hessian, d, gradLk, gradLkp, update)

    return L_Hessian