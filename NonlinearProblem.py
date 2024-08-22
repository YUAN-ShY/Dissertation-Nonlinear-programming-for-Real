import numpy as np
import sys

import NLTreeNode as nlt

out = -2

class NonlinearProblem:
    """
    this declares a class that represents a nonlinear problem usually given
    by an *.nl-file.
    
    It parses the nl-file and sets up expression trees for the objective
    and constraints (as well as storing the linear information)

    It further provides the following methods and fields

    - nvar
    - ncon
    - bl/bu: bounds on variables
    - cl/cu: bounds on constraints   cl<=c(x)<=cu
    - eval_objective(x)
    - eval_objgrad(x)
    - eval_objhess(x)
    - eval_constraint(nc, x)
    - eval_consgrad(nc, x)
    - eval_conshess(nc, x)
    - eval_hess(lam, x)

    - obj(x)
    - grad(x)
    - cons(x)
    - jac(x)
    - hess(x, lam)

    Derivatives are obtained by forward-AD on the trees.
    Does not (yet) support all possible expressions.
    In particular not Hessian of x^y where y is not constant

    if mode='EqBnd' then the problem is presented with all constraints
    being equality constraints and simple bounds on variables
    (by creating a g_i(x)-si=0 constraint for any inequality g_i(x)
    """
    
    def __init__(self, nlfile, mode=None):
        f = open(nlfile, "r")

        # - - - - - - - - -  reading header - - - - - - - 
        # read first line: just problem name
        l1 = f.readline()
        tok = l1.split()
        self.name = tok[6]

        # read second line: vars, constraints, objectives, ranges, eqns
        l1 = f.readline()
        tok = l1.split()
        self.nvar = int(tok[0])
        self.ncon = int(tok[1])
        self.nobj = int(tok[2])
        self.nrange = int(tok[3])
        self.neqns = int(tok[4])
        
        # read third line: nonlinear c/s, objectives
        l1 = f.readline()
        tok = l1.split()
        self.nlcons = int(tok[0])
        self.nlobj = int(tok[1])

        # read fourth line: network constraints: nonlinear, linear
        l1 = f.readline()

        # read fifth line:  nonlinear vars in constraints, objectives, both
        l1 = f.readline()
        tok = l1.split()
        self.nlvar_cs = int(tok[0])
        self.nlvar_obj = int(tok[1])
        self.nlvar_both = int(tok[2])

        # read sixth line: linear network variables; 
        l1 = f.readline()

        # read seventh line: discrete variables: bin, int, nonlinear (b,c,o)
        l1 = f.readline()

        # read eighth line: nonzeros in Jacobian, gradients
        l1 = f.readline()
        tok = l1.split()
        self.nzjac = int(tok[0])
        self.nzgrad = int(tok[1])

        # read ninth line: max name length
        l1 = f.readline()

        # read tenth line: common exprs
        l1 = f.readline()

        if out>=1:
            print("Defining nonlinear problem with ",end="")
            print("nvar = %d, ncon = %d"%(self.nvar, self.ncon))
        
        # - - - - - - - - -  end of reading header - - - - - - - 

        # declare problem arrays
        self.bl = np.zeros(self.nvar)
        self.bu = np.inf*np.ones(self.nvar)
        self.cl = np.zeros(self.ncon)
        self.cu = np.inf*np.ones(self.ncon)

        self.constraints = []
        self.objective = []

        self.glin = np.zeros(self.nvar)  # linear part of objective gradient
        self.Jlin = np.zeros((self.ncon,self.nvar)) # lin part of Jacobian
        # Master loop to read the remaining segments
        while True:
            line = f.readline()
            if not line:
                break

            if line[0]=='b':
                # read the b-section: table 17: bounds on variables

                for i in range(self.nvar):
                    l1 = f.readline()
                    self.bl[i], self.bu[i] = self.setBoundsFromLine(l1)
                    if out>=2:
                        print("Set bounds var %d: %g %g"%(i, self.bl[i], self.bu[i]))
            elif line[0]=='r':
                # read the r-section: table 17: bounds on constraints

                for i in range(self.ncon):
                    l1 = f.readline()
                    self.cl[i], self.cu[i] = self.setBoundsFromLine(l1)
                    if out>=2:
                        print("Set bounds con %d: %g %g"%(i, self.cl[i], self.cu[i]))
            elif line[0]=='k':
                # read the k-section: table 19/20: columns pointers for J
                self.colpts = []
                npts = int(line[1:])
                if (npts!=self.nvar-1):
                    print("Wrong number in k section")
                    sys.exit()
                for i in range(npts):
                    l1 = f.readline()
                    self.colpts.append(int(l1))
                if out>=2:
                    print("Column pointers are ",end="")
                    print(np.array(self.colpts))
            elif line[0]=='C':
                tok = line.split()
                nc = int(tok[0][1:])
                if out>=2:
                    print("Defining constraint %d"%(nc))
                cnd = self.readExpressionTree(self, f)
                self.constraints.append(cnd)
                
            elif line[0]=='O':
                tok = line.split()
                nc = int(tok[0][1:])
                if out>=2:
                    print("Defining objective %d"%(nc))
                self.sense = 1  # by default minimize
                if int(tok[1])==1:
                    self.sense = -1
                ond = self.readExpressionTree(self, f)
                self.objective.append(ond)
            elif line[0]=='J':
                tok = line.split()
                nj = int(tok[0][1:])
                if out>=2:
                    print("Reading J (Jacobian) section: constraint %d"%(nj))
                cnt = int(tok[1])
                for i in range(cnt):
                    l1 = f.readline()
                    tok1 = l1.split()
                    ix = int(tok1[0]) # is the variable index
                    nty = float(tok1[1]) # is the Jacobian entry
                    self.Jlin[nj][ix] = nty
            elif line[0]=='G':
                tok = line.split()
                ng = int(tok[0][1:])
                cnt = int(tok[1])
                if out>=2:
                    print("Reading G (objective gradient) section: objective %d"%(ng))
                for i in range(cnt):
                    l1 = f.readline()
                    tok1 = l1.split()
                    ix = int(tok1[0]) # is the variable index
                    nty = float(tok1[1]) # is the gradient entry
                    self.glin[ix] = nty

        if out>=2:
            print("J = ",end="")
            print(self.Jlin)
            print("G = ",end="")
            print(self.glin)

        f.close()

        self.objflat = self.objective[0].get_flat_tree(self.nvar)
        if out>=2:
            print("nodes in flat obj tree: %d"%(len(self.objflat)))

        self.consflat = self.constraints[0].get_flat_tree(self.nvar)
        if out>=2:
            print("nodes in flat cons[0] tree: %d"%(len(self.consflat)))

        self.mode = mode
        self.orignvar = self.nvar
        if mode=="EqBnd":
            # set up the transformation into an equality constrained
            # problem with simple bounds
            # 1) id inequality constraints
            
            iseq = np.abs(self.cu-self.cl)<1e-8
            self.isineq = ~iseq
            self.nineq = np.count_nonzero(self.isineq)
            print("isineq = ",end="")
            print(self.isineq)
            print("nineq = %d"%(self.nineq))
            self.cntineq = -1*np.ones(self.ncon)
            self.pineq = -1*np.ones(self.nineq)
            cnt = 0
            for i in range(self.ncon):
                if self.isineq[i]:
                   self.cntineq[i] = cnt
                   self.pineq[cnt] = i
                   cnt = cnt+1
            # now
            # self.cntineq[i] gives for inequality i the position of its slack
            # self.pineqp[i] gives for slack i the position of it inequality

            # now relabel presented size of problem
            self.orignvar = self.nvar
            self.nvar = self.nvar + self.nineq
            self.origcl = self.cl.copy()
            self.origcu = self.cu.copy()
            self.origbl = self.bl
            self.origbu = self.bu
            self.bl = np.zeros(self.nvar)
            self.bu = np.zeros(self.nvar)
            for i in range(self.orignvar):
                self.bl[i] = self.origbl[i]
                self.bu[i] = self.origbu[i]

            for i in range(self.nineq):
                ixineq = int(self.pineq[i])
                #print(ixineq)
                self.bl[self.orignvar+i] = self.cl[ixineq]
                self.bu[self.orignvar+i] = self.cu[ixineq]
                self.cl[ixineq] = 0.0
                self.cu[ixineq] = 0.0

            #- - - - -  print the changed problem - - -
            for i in range(self.orignvar):
                print("orig bl/bu[%d]: %f   %f"%(i, self.origbl[i], self.origbu[i]))
            for i in range(self.ncon):
                print("orig cl/cu[%d]: %f   %f"%(i, self.origcl[i], self.origcu[i]))

            for i in range(self.nvar):
                print("bl/bu[%d]: %f   %f"%(i, self.bl[i], self.bu[i]))
            for i in range(self.ncon):
                print("cl/cu[%d]: %f   %f"%(i, self.cl[i], self.cu[i]))
            
            #sys.exit(1)
            
    # - - - - - - - - - - - -  setBoundsFromLine - - - - - - - - -
    @staticmethod
    def setBoundsFromLine(line):
        tok = line.split()
        if tok[0]=="0":
            bl = float(tok[1])
            bu = float(tok[2])
        elif tok[0]=="1":
            bl = -np.inf
            bu = float(tok[1])
        elif tok[0]=="2":
            bl = float(tok[1])
            bu = np.inf
        elif tok[0]=="3":
            bl = -np.inf
            bu = np.inf
        elif tok[0]=="4":
            bl = float(tok[1])
            bu = float(tok[1])
        elif tok[0]=="5":
            # "body complements variable"
            print("bounds type 5 not implemented")
            sys.exit(1);
        else:
            print("unknown bound type:")
            print(tok[0])
            sys.exit(1);

        return bl, bu


    def set_scale(self, scale):
        self.scale = scale
        
# - - - - - - - - - - - readExpressionTree - - - - -
    # read in the expression tree and build a python representation
    # make sure that each variable node is only on the tree once

    @staticmethod
    def readExpressionTree(self, f):
        self.varnodes = [None]*self.nvar
        return self._readExpressionTree(self, f, 0)

    @staticmethod
    def _readExpressionTree(self, f, level):
        line = f.readline()

        c = line[0]
        nd = nlt.NLTreeNode(-1, 0)

        if c=='o':
            if out>=2:
                print("Level %d: operand: "%(level),end="")
            tok = line.split()
            op = int(tok[0][1:])
            if op==0:
                if out>=2:
                    print("+")
                c1 = self._readExpressionTree(self, f, level+1)
                c2 = self._readExpressionTree(self, f, level+1)
                nd.set_type("+")
                nd.add_chd(c1)
                nd.add_chd(c2)
            elif op==1:
                if out >= 2:
                    print("-")
                c1 = self._readExpressionTree(self, f, level+1)
                c2 = self._readExpressionTree(self, f, level+1)
                nd.set_type("-")
                nd.add_chd(c1)
                nd.add_chd(c2)
            elif op==2:
                if out>=2:
                    print("*")
                c1 = self._readExpressionTree(self, f, level+1)
                c2 = self._readExpressionTree(self, f, level+1)
                nd.set_type("*")
                nd.add_chd(c1)
                nd.add_chd(c2)
            elif op==3:
                if out>=2:
                    print("/")
                c1 = self._readExpressionTree(self, f, level+1)
                c2 = self._readExpressionTree(self, f, level+1)
                nd.set_type("/")
                nd.add_chd(c1)
                nd.add_chd(c2)
            elif op==4:
                if out>=2:
                    print("%")
                c1 = self._readExpressionTree(self, f, level+1)
                c2 = self._readExpressionTree(self, f, level+1)
                nd.set_type("%")
                nd.add_chd(c1)
                nd.add_chd(c2)
            elif op==5:
                if out>=2:
                    print("^")
                c1 = self._readExpressionTree(self, f, level+1)
                c2 = self._readExpressionTree(self, f, level+1)
                nd.set_type("^")
                nd.add_chd(c1)
                nd.add_chd(c2)
            elif op==15:
                if out>=2:
                    print("abs()")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("abs")
                nd.add_chd(c1)
            elif op==16:
                if out>=2:
                    print("un-")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("un-")
                nd.add_chd(c1)
            elif op==39:
                if out>=2:
                    print("sqrt()")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("sqrt")
                nd.add_chd(c1)
            elif op==41:
                if out>=2:
                    print("sin()")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("sin")
                nd.add_chd(c1)
            elif op==43:
                if out>=2:
                    print("ln()")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("ln")
                nd.add_chd(c1)
            elif op==44:
                if out>=2:
                    print("exp()")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("exp")
                nd.add_chd(c1)
            elif op==46:
                if out>=2:
                    print("cos()")
                c1 = self._readExpressionTree(self, f, level+1)
                nd.set_type("cos")
                nd.add_chd(c1)
            elif op==54:
                if out>=2:
                    print("sum()")
                line = f.readline()
                tok = line.split()
                nop = int(tok[0])
                nd.set_type("sum")
                for i in range(nop):
                    ci = self._readExpressionTree(self, f, level+1)
                    nd.add_chd(ci)
            else:
                print("Operator not implemented yet: %d"%(op))
                sys.exit(1)

        elif c=='n':
            tok = line.split()
            const = float(tok[0][1:])
            if out>=2:
                print("Level %d: number: %f"%(level,const))
            nd = nlt.NLTreeNode("n", const)
        elif c=='v':
            tok = line.split()
            ix = int(tok[0][1:])
            if out>=2:
                print("Level %d: variable: %d"%(level, ix))
            if self.varnodes[ix]==None:
                if out>=2:
                    print("Not registered yet")
                nd = nlt.NLTreeNode("v", ix)
                self.varnodes[ix] = nd
            else:
                nd = self.varnodes[ix]
                # FIXME: unclear how the variables nodes gets assigned
                # more than one parent
                
        return nd

    # - - - - - - - -  evaluation to call from outside  methods - - - - - - - -

    def eval_objective(self, x):
        #print(self.objective[0].to_string())
        #print("self.glin[0] = ",end="")
        #print(self.glin)
        return self.objective[0].eval_val(x)+np.dot(self.glin, x[0:self.orignvar])

    def eval_constraint(self, nc, x):
        if out>=2:
            print("eval cons[%d]: %s"%(nc,self.constraints[nc].to_string()))

        csbdy = self.constraints[nc].eval_val(x)+np.dot(self.Jlin[nc,:], x[0:self.orignvar])
        #print(self.mode)
        if self.mode=="EqBnd":
            #print("Called eval_cons in EqBnd mode")
            if self.isineq[nc]:
                ixsi = self.orignvar + int(self.cntineq[nc])
                csbdy = csbdy - x[ixsi]
        return csbdy

    def eval_objgrad(self, x):
        if out>=2:
            print("eval objgrad: %s"%(self.objective[0].to_string()))
        val, gd = self.objective[0].eval_grad_forwardADTree(x)
        #print("self.glin = ",end="")
        #print(self.glin)
        gd[0:self.orignvar] = gd[0:self.orignvar]+self.glin
        return gd
        
    def eval_consgrad(self, nc, x):
        if out>=2:
            print("eval_consgrad[%d]: %s"%(nc,self.constraints[nc].to_string()))
        val, gd = self.constraints[nc].eval_grad_forwardADTree(x)
        if out>=2:
            print("self.Jlin[nc,:] = ",end="")
            print(self.Jlin[nc,:])

        gd[0:self.orignvar] = gd[0:self.orignvar]+self.Jlin[nc,:]

        if self.mode=="EqBnd":
            #gdd.append(np.zeros(self.nineq))
            if self.isineq[nc]:
                ixsi = self.orignvar + int(self.cntineq[nc])
                gd[ixsi] = -1.0
        return gd

    def eval_objhess(self, x):
        if out>=1:
            print("eval_objhess: %s"%(self.objective[0].to_string()))
        val, gd, H = self.objective[0].eval_hess_forwardADTree(x)

        #if self.mode=="EqBnd":
        #    H2 = np.zeros((self.nvar,self.nvar))
        #    for i in range(self.orignvar):
        #        for j in range(self.orignvar):
        #            H2[i,j] = H[i,j]
        #    return H2
        return H

    def eval_conshess(self, nc, x):
        if out>=1:
            print("eval_conshess[%d]: %s"%(nc, self.constraints[nc].to_string()))
        val, gd, H = self.constraints[nc].eval_hess_forwardADTree(x)
        #if self.mode=="EqBnd":
        #    H2 = np.zeros((self.nvar,self.nvar))
        #    for i in range(self.orignvar):
        #        for j in range(self.orignvar):
        #            H2[i,j] = H[i,j]
        #    return H2
        return H

    def eval_hess(self, lam, x):
        val, gd, H = self.objective[0].eval_hess_forwardADTree(x)
        for nc in range(self.ncons):
            val, gd, Hc = self.constraints[nc].eval_hess_forwardADTree(x)
            H = H + lam[nc]*Hc

        #if self.mode=="EqBnd":
        #    H2 = np.zeros((self.nvar,self.nvar))
        #    for i in range(self.orignvar):
        #        for j in range(self.orignvar):
        #            H2[i,j] = H[i,j]
        #    return H2

        return H

    # ---- and here are the usual interfaces to the outside

    def obj(self, x):
        return self.eval_objective(x)

    def cons(self, x):
        c = np.zeros(self.ncon)
        for ixc in range(self.ncon):
            c[ixc] = self.eval_constraint(ixc, x)
        return c

    def grad(self, x):
        return self.eval_objgrad(x)

    def jac(self, x):
        J = np.zeros((self.nvar, self.ncon))
        for ixc in range(self.ncon):
            J[:,ixc] = self.eval_consgrad(ixc, x)
        return J

    def hess(self, x, lam):
        H = self.eval_objhess(x)
        for ixc in range(self.ncon):
            H += lam[ixc]*self.eval_conshess(ixc, x)
        return H
    
