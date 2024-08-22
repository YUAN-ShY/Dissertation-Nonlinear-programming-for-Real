import numpy as np
import numpy.linalg as LA
import QP

# this implements ~/php/Portfolio/javaipm/IPMBounds.java in python

class IPM:
    """
    This is a native python implementation of the primal-dual IPM
    (as in HOPDM or OOPS)

    Can solve LPs and QPs with bound constraints (and equality constraints)
 
    min c'*x + 0.5 x'*Q*x,   subject to, Ax = b,  bl < =x < =bu

    Inequalities have to be converted to equalities by introducing slacks

    This is called by

    ipm = IPM.IPM(qp)
    ipm.solve()
    x = ipm.getSol()
    y = ipm.getDualsCons()
    z = ipm.getDualsBnd()

    The qp to be solved has to be created using the QP.py class
    """

    regTheta = 1.0e-12   # this could become a vector, but not yet
    out = -1

    # IPM return codes
    SOLVED = 0
    INFEASIBLE = 1
    UNBOUNDED = 2
    MAXITER = 3
    
    def __init__(self, qp):

        self.qp = qp
        self.opt_tol = 1e-6
        self.iter_limit = 50

    # ----------------------------- solve() ------------------------
    def solve(self):
        """
        This is the main method to solve the QP problem
        """

        qp = self.qp

        if qp.Q is None:
            qp.Q = np(zeros(qp.n, qp.n))


        # it seems that some routines don't like np.infty
        qp.bl[~qp.where_l] = -1e20
        qp.bu[~qp.where_u] = 1e20

        x, y, z, s, t, w = self.find_start_point(qp)

        # print("Starting point is:")
        # print(x)
        # print(y)
        # print(z)
        

        StepMax = 1.0
        MaxCorr = 5
        always_hoc = False
        self.rtcode = IPM.SOLVED

        
        pObj0 = 0.0
        dObj0 = 0.0
        alphaP = 0.0
        alphaD = 0.0

        dxc = None
        dzc = None
        dtc = None
        dsc = None
        dwc = None
        
        
        # main IPM loop
        contPDM = True
        iter = 0
        while contPDM:
            iter += 1

            pObj = np.dot(qp.c, x) + 0.5*np.dot(x, np.dot(qp.Q, x))
            dObj = np.dot(qp.b, y) - 0.5*np.dot(x, np.dot(qp.Q, x))
            for i in range(qp.n):
                if qp.where_l[i]:
                    dObj += qp.bl[i]*z[i]
                if qp.where_u[i]:
                    dObj -= qp.bu[i]*w[i]

            #dObj += -np.dot(qp.where_u*qp.bu, w) +np.dot(qp.where_l*qp.bl, z)

            dlGap = pObj - dObj
            # save best estimate of primal and dual objectives
            if iter==1:
                oldGap = np.abs(dlGap) + 1e+8
                pObj0 = np.abs(pObj)+1e+8
                dObj0 = np.abs(dObj)+1e+8
            else:
                if np.abs(dlGap)<oldGap:
                    oldGap = np.abs(dlGap)
                    pObj0 = pObj+1.0e-6
                    dObj0 = dObj-1.0e-6
                    
                
            # Check if the problem is Unbounded or Infeasible. */
            if np.abs(dlGap) / oldGap > 1.e+2 and iter > 3:
                #print("dlGap, oldGap = %f, %f"%(np.abs(dlGap), oldGap))
                if np.abs (pObj) / (np.abs (pObj0) + 1.0) > 1.e+4:
                    if IPM.out>=2:
                        print("IPM stopped after %d iters: UNBOUNDED primal or INFEASIBLE dual"%(iter - 1))
                    self.rtcode = IPM.UNBOUNDED;
                    return self.rtcode;
	
                if np.abs (dObj)/ (np.abs (dObj0) + 1.0) > 1.e+4:
                    if IPM.out>=2:
                        print("IPM stopped after %d iters: UNBOUNDED dual or INFEASIBLE primal"%(iter - 1))
                    self.rtcode = IPM.INFEASIBLE;
                    return self.rtcode;
	
      

            # ----- IPM convergence test ----
            dp = np.abs(dObj)+1.0;
            relGap = np.abs(dlGap)/dp;

            if iter<3:
                relGap += 0.5 # make sure at least 3 iterations
            if relGap < self.opt_tol:
                if IPM.out>=2:
                    print("Converged!")
                if IPM.out>=0:
                    print("IPM Converged in %d iters. Obj = %f"%(iter-1, pObj))
                contPDM = False
      

            if iter>self.iter_limit:
                if IPM.out>=0:
                    print("IPM Exceeded iteration limit (%d): gap=%g"%(iter-1, relGap))
                self.rtcode = IPM.MAXITER
                contPDM = False

            # if we did not stop, then calculate the new step
            if contPDM:
                # pushVar
                add_t, add_z = self.pushVarAmount(relGap)
                t += add_t
                z += qp.where_l*add_z
                s += qp.where_u*add_t
                w += qp.where_u*add_z

                # - - - - - - - - - - computeResiduals - - - - - - - 
                #  xib = b - Ax             
                #  xic = c - A'y + Qx -z + w 
                #  xiu = u - x - s           
                #  xil = l - (x - t)         
                xib = qp.b - np.dot(qp.A, x)
                xic = qp.c - np.dot(qp.A.transpose(), y) + np.dot(qp.Q, x) - z + w

                xiu = qp.where_u*(qp.bu - x - s)
                xil = qp.where_l*(qp.bl - (x - t))

                err_b = LA.norm(xib, np.inf)
                err_c = LA.norm(xic, np.inf)
                err_l = LA.norm(xil, np.inf)
                err_u = LA.norm(xiu, np.inf)

                if IPM.out>=2:
                    print("CompRes: xib= %4.2e, xic= %4.2e"%(err_b, err_c))
                    print("CompRes: xil= %4.2e, xiu= %4.2e"%(err_l, err_u))


                # - - - - - - - defineTheta - - - - - -
                theta = self.defTheta(qp, t, z, s, w)

                theta2 = theta/(1+theta*IPM.regTheta)                

                # set up AugSys
                Qmod = qp.Q + np.diag(1./(theta2+1.0e-24))
                augSys = np.block([[-Qmod, qp.A.transpose()],[qp.A, np.zeros((qp.m, qp.m))]])
                
                # Compute Average Complementarity Product & stats
                # compute average = (dp = t'z + w's)/no_of_cp
                #for i in range(qp.n):
                #    print("%d: t=%f, z=%f"%(i, t[i], z[i]))

                acp = 0.0
                ncp = 0
                for i in range(qp.n):
                    if qp.where_l[i]:
                        ncp += 1
                        acp += t[i]*z[i]
                        
                    if qp.where_u[i]:
                        ncp += 1
                        acp += w[i]*s[i]
                        
                #print(acp)
                #print(ncp)
                acp = acp/ncp

                # ksmall = #cp that are < 0.01 of average
                pr_small = acp/1.0e+2
                ksmall = 0
                for i in range(qp.n):
                    if qp.where_l[i]:
                        if t[i]*z[i]<pr_small:
                            ksmall += 1
	  
                    if qp.where_u[i]:
                        if s[i]*w[i]<pr_small:
                            ksmall += 1
	  
	
                if IPM.out>=3:
                    print("avCP= %f, ksmall=%d"%(acp, ksmall))

                averageDG = np.abs(dlGap)/qp.n

                barr = self.getInitialBarr(acp, alphaP, alphaD, ksmall, iter, relGap, err_b+err_c, averageDG)

                oldbarr = 0.0
                iDir = 1
                
                # need to pass in x, y, z, t, s, w
                # also need to pass in xib, xic, xiu, xil
                dx, dy, dz, dt, ds, dw = self.getIPMDir(x, y, z, t, s, w,
                               xib, xic, xil, xiu, augSys, iDir, oldbarr,
                               barr, alphaP, alphaD, dxc, dzc, dtc, dsc, dwc,
                               0.1*(err_b+err_c))


                alphaT = self.getMaxStep(t, dt)
                alphaZ = self.getMaxStep(z, dz)
                alphaS = self.getMaxStep(s, ds)
                alphaW = self.getMaxStep(w, dw)

                oldbarr = barr

                alphaP = StepMax
                if alphaT < alphaP:
                    alphaP = alphaT
                if alphaS < alphaP:
                    alphaP = alphaS

                alphaD = StepMax
                if alphaZ < alphaD:
                    alphaD = alphaZ
                if alphaW < alphaD:
                    alphaD = alphaW


                #if (prtLvl>=2)
                #print("pred_dir: aT= %.3e, aS= %.3e, aZ= %.3e, aW= %.3e\nPrimal space: alphaP= %.3e, Dual space:   alphaD= %.3e\n",
                #alphaT, alphaS, alphaZ, alphaW, alphaP, alphaD);
                
                saveP = alphaP
                saveD = alphaD


                barr = self.computeMehrotraBarr(alphaP, alphaD, t, z, s, w, 
                                       dt, dz, ds, dw, oldbarr, acp, iter)

                # ------- now do the higher order correctors --------
                iOrder = 1
                contCorr = True

                while contCorr and iOrder < MaxCorr:
                    iOrder += 1
                    iDir = 2
                    if iOrder>2:
                        iDir = 3

                    # MinIter = iOrder;  // not sure what this does -> Matlab outname
                    if IPM.out >=3:
                        print("Call getIPMDir: idir= %d, barr= %g"%(iDir, barr))

                    # Calculate higher order correction step */
                    # affine scaling direction is delta?p,
                    # last corrected step is delta?c */
                    # GondzioCorrector (iDir=3) needs delta?c set! */
                    # Calculate higher order directions:
                    # iDir=2: Mehrotra
                    #   IN:  (vx,vy,vz,vs,vw) is current point
                    #        vdelta?p:   predictor direction
		    #                    (i.e. affine scaling)
		    #        oldbarr: barrier used for affine scaling
		    #        barr   : new barrier to use
                    #   OUT: vdelta?c:   corrector direction
                    # iDir=3: Gondzio
                    #   IN:  (vx,vy,vz,vs,vw) is current point
		    #        vdelta?p:   predictor direction
		    #                    (i.e. aggregated affine scaling
                    #                    plus all successful correctors)
		    #        AlphaP/D: largest step feas. in pred direction
		    #        barr    : last barrier used
                    #   OUT: vdelta?c:   corrector direction
                    #
                    # OOPS: -> new step into pdCorrDir
                    # OOPS: pdAgrDir always passed as first argument (pdPredDir: dXp, dZp)
                    # -> only used for iDir =2,3
                    # at this point pdAgrDir is the dir from iDir=1
                    # -> HERE: solve for corrector direction dxc, dyc, dzc
                    # current accepted dir (dx, dz) is passed in as Xp, Zp
                    dxc,dyc,dzc,dtc,dsc,dwc = self.getIPMDir(x, y, z, t, s, w,
                                    xib, xic, xil, xiu, augSys, iDir, oldbarr,
                                    barr, alphaP, alphaD, dx, dz, dt, ds, dw,
                                    0.1*(err_b+err_c))


                    oldbarr = barr
                    # CorDir += AggrDir
                    # IPM.pdCorrDir->addTo(*(IPM.pdAggrDir));
                    for i in range(qp.n):
                        dxc[i] += dx[i]
                        if qp. where_l[i]:
                            dtc[i] += dt[i]
                            dzc[i] += dz[i]
	    
                        if qp.where_u[i]:
                            dsc[i] += ds[i]
                            dwc[i] += dw[i]

                    for i in range(qp.m):
                        dyc[i] += dy[i]



                    #OOPS: max steps base on pdCorrDir
                    alphaT = self.getMaxStep(t, dtc)
                    alphaZ = self.getMaxStep(z, dzc)
                    alphaS = self.getMaxStep(s, dsc)
                    alphaW = self.getMaxStep(w, dwc)
                    
                    alphaP = StepMax
                    if alphaT < alphaP:
                        alphaP = alphaT
                    if alphaS < alphaP:
                        alphaP = alphaS

                    alphaD = StepMax
                    if alphaZ < alphaD:
                        alphaD = alphaZ
                    if alphaW < alphaD:
                        alphaD = alphaW

                    if IPM.out>=2:
                        print("corr_dir: aT= %.3e, aS= %.3e, aZ= %.3e, aW= %.3e\nPrimal space: alphaP= %.3e, saveP= %.3e\nDual space:   alphaD= %.3e, saveD= %.3e"%(alphaT, alphaS, alphaZ, alphaW, alphaP, saveP, alphaD, saveD))


                    #Here to control computing higher order correcting terms. 
                    nimproved = 0

                    if alphaP > 1.01 * saveP:
                        # If improvement use corrected step rather than old 
                        nimproved += 1
                        #IPM.pdCorrDir->copyToXTS(*(IPM.pdAggrDir));
                        for i in range(qp.n):
                            dx[i] = dxc[i]
                            if qp.where_l[i]:
                                dt[i] = dtc[i]
                            if qp.where_u[i]:
                                ds[i] = dsc[i]

                        saveP = alphaP
                    else:
                        alphaP = saveP

                    if alphaD > 1.01 * saveD:
                        nimproved += 1
                        # if improvement use corrected step rather than old 
                        # pdAggrDir is the final direction to take
                        # IPM.pdCorrDir->copyToYZW(*(IPM.pdAggrDir));
                        for i in range(qp.n):
                            if qp.where_l[i]:
                                dz[i] = dzc[i]
                            if qp.where_u[i]:
                                dw[i] = dwc[i]

                        for i in range(qp.m):
                            dy[i] = dyc[i]

                        saveD = alphaD
                    else:
                        alphaD = saveD

                    if IPM.out>=3:
                        print("IPM:   Improved= %d"%(nimproved))

                    contCorr = True
                    if nimproved == 0:
                        contCorr = False

                    if nimproved == 0 and iDir ==2:
                        # tried Mehrotra Corrector, but failed
                        # still try at least one higher order corrector 
                        if always_hoc:
                            contCorr = True;
                            iDir = 3
                    #if improvement then loop back and try next corrector */
                # end while (ContCorrs > 0 && iOrder < MaxCorr) */

                alpha0 = 0.9995

                alphaP *= alpha0
                alphaD *= alpha0

                # ---------------- make step ------------------------
                small_x = 1.0e-14
                for i in range(qp.n):
                    x[i] += alphaP*dx[i]
                    if qp.where_l[i]:
                        t[i] += alphaP*dt[i]
                        z[i] += alphaD*dz[i]
                        if t[i]<small_x:
                            t[i]=small_x
                        if z[i]<small_x:
                            z[i]=small_x
	  
                    if qp.where_u[i]:
                        s[i] += alphaP*ds[i]
                        w[i] += alphaD*dw[i]
                        if s[i]<small_x:
                            s[i]=small_x
                        if w[i]<small_x:
                            w[i]=small_x
                for i in range(qp.m):
                    y[i] += alphaD*dy[i]


                #print("after make setp:")
                #print(x)
                #print(t)
                #print(z)
                #print(s)
                #print(w)
                #print(y)
                


            #end if (contPDM)

            if IPM.out>=1:
                if iter == 1:
                    print(" Itn\tpObj\t    dObj       AlphaP\t AlphaD     relGap\txib\t  xic")

                print("%3d %11.4e %11.4e   %5.2e  %5.2e  %9.2e %9.2e %9.2e"%(iter, pObj, dObj, alphaP, alphaD, relGap, err_b, err_c))
        #end while (contPDM)

        if IPM.out>=2:
            print("Solution is x = ")
            print(x)
            print(y)
            print(z)

        self.solx = x
        self.soly = y
        self.solz = z-w
        
        return self.rtcode


    # ----------------------- find_start_point() ------------------------
    def find_start_point(self, qp):
        """
        findStartPoint (with lb and ub)
        
        min { t't+s's  s.t. A*x = b, x-t = l, x+s = u (where l/u are present)}
        min { z'z+w'w  s.t. At*y+z =c (w/z only present, where u/l is present)}}
    
        which is equivalent to solving
    
        [-TH  At] [x] = [ -u-l]  s = u-x, t=x-l 
        [ A   0 ] [v] = [  b]
    
        u = 0 if no upper bound present, TH_i = 0 if no bound
                                              = 1 if upper or lower bound
                                              = 2 if upper and lower bound
        and
  
          [-TH At] [v] = [c]   z = -v, where l is present 
          [ A  0 ] [y] = [0]   w = v,  where u is present 
  
        TH as above.
  
        For QP the minimization problems have the terms
          t'Qt + s'Qs, z'Qz + w'Qw 
        *added* to them, which is equivalent to replacing -TH with -Q-TH 
        AND changing -u to -0.5(Q+I)u (?)
  
        The so obtained x, z  are modified according to:
          - dp = max{-1.5*min{t_i}, -1.5*min{s_i}, 0}
            dd = max{-1.5*min{z_i}, -1.5*min{w_i}, 0}
  
         - dp = dp + 0.5*tzsw/(st+sw)
           dd = dd + 0.5*tzsw/(sz+ss)
  
           where xzsw = (t+dp*1)'(z+dd*1) + (s+dp*1)'(w+dd*1)
  	        st   = sum (t_i+dp)
  	        sz   = sum (z_i+dd)
  	        ss   = sum (s_i+dp)
  	        sw   = sum (w_i+dd)
  
  	 - t = t+dp*1, z = z+dd*1      x, y unaltered
           s = s+dp*1, w = w+dd*1      
        """

        qp = self.qp

        theta = np.zeros(qp.n) + 1.0*qp.where_l*1 + 1.0*qp.where_u*1
        #print("theta=")
        #print(theta)

        theta2 = theta/(1+theta*IPM.regTheta)
        #print(theta)
        #print(theta2)
        #print(1/(theta2+1.e-24))

        Qmod = qp.Q + np.diag(1./(theta2+1.0e-24))
        augSys = np.block([[-Qmod, qp.A.transpose()],[qp.A, np.zeros((qp.m, qp.m))]])

        # --- solve first system ---
        # build rhs: [-l-u, b]
        rhsx = 0.0 - qp.bl*qp.where_l - qp.bu*qp.where_u
        rhs = np.block([rhsx, qp.b])
        #print(rhsx)
        

        #print(augSys)
        #print(rhs)
        solx = LA.solve(augSys, rhs)
        #print("first solve = ")
        #print(solx)

        # vs = vu - vx
        # vt = vx - vl
        sols = 0 + qp.where_u*(qp.bu - solx[0:qp.n])
        solt = 0 + qp.where_l*(solx[0:qp.n] - qp.bl)
        #print("sols/t")
        #print(sols)
        #print(solt)

        # --- solve second system ---
        rhs = np.block([qp.c, np.zeros(qp.m)])
        #print("rhs, solv")
        #print(rhs)
        
        solv = LA.solve(augSys, rhs)
        #print("second solve = ")
        #print(solv)

    
        # copy out the solution vectors
        #    z = -v, where l is present 
        #    w = v,  where u is present 
        
        solz = 0.0 - solv[0:qp.n]*qp.where_l
        solw = 0.0 + solv[0:qp.n]*qp.where_u

        #print("solw = ")
        #print(solw)
        # - - - - -  do the modifications
        #    - dp = max{-1.5*min{t_i}, -1.5*min{s_i}, 0}
        #      dd = max{-1.5*min{z_i}, -1.5*min{w_i}, 0}

        dp = 0.0
        dd = 0.0
        for i in range(qp.n):
            if qp.where_l[i]:
                if solt[i]<dp:
                    dp = solt[i]
                if solz[i]<dd:
                    dd = solz[i];
            if qp.where_u[i]:
                if sols[i]<dp:
                    dp = sols[i]
                if solw[i]<dd:
                    dd = solw[i]  
        #dp = np.max([-1.5*np.min(solt), 1.5*np.min(sols), 0.0])
        #dd = np.max([-1.5*np.min(solz), 1.5*np.min(solw), 0.0])

        #print("dp, dd = %f, %f"%(dp, dd))
    
        dp = -1.5*dp
        dd = -1.5*dd
        if dd<1.0e-5:
            dd = 1.0e-5

        # calculate tzsw = (t+dp*1)'(z+dd*1) + (s+dp*1)'(w+dd*1) */
        # calculate st = sum(t_i+sp), sz, ss, sw 
        # st = sum(t+dp);
        # sz = sum(z+dd);

        tzsw = 0
        st=0
        sz=0
        for i in range(qp.n):
            if qp.where_l[i]:
                st += (solt[i]+dp)
                sz += (solz[i]+dd)
                tzsw += (solt[i]+dp)* (solz[i]+dd)
            if qp.where_u[i]:
                st += (sols[i]+dp)
                sz += (solw[i]+dd)
                tzsw += (sols[i]+dp)* (solw[i]+dd)
                
        #tzsw = np.dot(solt+dp, solz+dd) + np.dot(sols+dp, solw+dd)
        #st = np.sum(solt+dp) + np.sum(sols+dp)
        #sz = np.sum(solz+dd) + np.sum(solw+dd)

        # correct dp, dd 
        dp += 0.5*tzsw/st
        dd += 0.5*tzsw/sz

        #print("dp = %f, dd = %f"%(dp, dd))
        # correct vt, vz, vs, vw */
        # x = x+dp, z = z+dd;

        solt += qp.where_l*dp
        solz += qp.where_l*dd
        sols += qp.where_u*dp
        solw += qp.where_u*dd
    
        # The starting point is now in
        #  x = solx[0:nVar], z=solz, y=solv[nVar:nVar+nCon]
    
        x = (~qp.where_l)*solx[0:qp.n] + qp.where_l*(qp.bl+solt)

        t = qp.where_l*solt
        s = qp.where_u*sols
        w = qp.where_u*solw
        z = solz
        y = solv[qp.n:qp.n+qp.m]

        return x, y, z, s, t, w

    # - - - - - - - - - pushVarAmount - - - - - - - - - -
    def pushVarAmount(self, relGap):
        add_t = 1.0e-8
        add_z = 1.0e-6
        
        if relGap < 1.0e-1:
            add_t = 1.0e-9
            add_z = 1.0e-7

        if relGap < 1.0e-3:
            add_t = 1.0e-10
            add_z = 1.0e-8

        # FIXME: changed agr Oct 2003: to enable the code to get better final
        # tolerance than 1e-6. Apparently this is dangerous and should be
        # something like:
        # if rel_gap < 1.0e-5:
        #   add_x = 1.0e-12;
        #   add_z = 1.0e-10;

        if relGap < 1.0e-4:
            add_t = 1.0e-11
            add_z = 1.0e-9

        if relGap < 1.0e-5:
            add_t = 1.0e-14
            add_z = 1.0e-14

        return add_t, add_z



    # defTheta() ----------------------------------------------------------
    def defTheta(self, qp, t, z, s, w):
        """
        Define diagonal scaling matrix Theta.: x/z [1/(z/t + w/s)]
        Sets theta[i] = t[i]/z[i]  ( t[i]*s[i]/(z[i]*s[i]+w[i]*t[i])  )
        with the following regularizations:
            - Any denominator <small_t (1.0e-20) is set to small_t (ret = 2)
            - Any theta[i] > th_max (1.0e10) is set to sqrt(1.0e10*theta[i])
        """
    
        small_t = 1.0e-20
        theta = np.zeros(qp.n)
        for i in range(qp.n):
            #theta[i] = x[i]/z[i];
            # set the denominator of 1/(z/t+w/s)
            denom = 0.0
            if qp.where_l[i]:
                if qp.where_u[i]:   #both upper and lower bounds
                    denom = z[i]*s[i] + w[i]*t[i]
                else:               # only lower bound
                    denom = z[i]
            
            else:
                if qp.where_u[i]:      # only upper bound
                    denom = w[i]
                    
            if denom<small_t:
                denom = small_t
            # set the numerator t*s
            if qp.where_l[i]:
                if qp.where_u[i]:   # both upper and lower bounds
                    theta[i] = t[i]*s[i]
                else:            # only lower bound
                    theta[i] = t[i]
            
            else:
                if qp.where_u[i]:   #only upper bound
                    theta[i] = s[i];
            
          
          
            if ~qp.where_l[i] and ~qp.where_u[i]:
                theta[i] = 1e+40  #THETA_FOR_FREE_VARS;
            else:
                theta[i] = theta[i]/denom
          
            # and regularize
            th_max = 1.0e+10
            if theta[i]>th_max:
                theta[i] = np.sqrt(th_max*theta[i])

        return theta

    # getInitialBarr() -------------------------------------------------
    def getInitialBarr(self, acp, alphaP, alphaD, ksmall, iter, relGap, err_bc, averageDG):
        """
         Set the initial barrier parameter for the affine scaling step. This is
         affected by rather a lot of parameters depending on different
         algorithm options.

         Basically:
             - barr = max(averageCP/1000, averageDG/20);
             
         unless
         - last steps (AlphaP/AlphaD) small => barr = averageCP/(2, 5, 10)
         - more than 1% cp < avCP/100 => barr = averageCP/10
         - Iter==1 => barr = average/5, Iter<=3 => barr = average/10
         - do_center => barr = average
         - feas LAGGING opt => barr = average/2
         - increase_barr => barr = 2*average
         - ret_adv_center_mode => barr = adv_center_barr
         - use_start_point&&Iter==1&&adv_mu>0 => barr = adv_mu
         
         @param acp           Average complementarity product
         @param alphaP        last primal stepsize
         @param alphaD        last dual stepsize
         @param ksmall        number of complementarity prod < averageCP/100
         @param Iter          iteration number
         @param err_bc    residual: err_b+err_c - measure for feas for LAGGING)
         @param averageDG     average duality gap: |p_Obj-d_Obj|/#CP
         """
        qp = self.qp
        no_of_cp = qp.n
        
        force_feas = False

        prt_log = False
        if IPM.out>=3:
            prt_log = True

        barr = acp * 1.0e-3
        if prt_log:
            print("Set barr to 0.001*avrgCP = %f"%(barr))

        if alphaP < 0.5 or alphaD < 0.5:
            barr = acp * 1.0e-2
            if prt_log:
                print("AlphaP/D<0.5 => set barr to 0.01*avrgCP = %f"%(barr))

        if alphaP < 0.2 or alphaD < 0.2:
            barr = acp * 1.0e-1
            if prt_log:
                print("AlphaP/D<0.2 => set barr to 0.1*avrgCP = %f"%(barr))
    
        if alphaP < 0.1 or alphaD < 0.1:
            barr = acp * 2.0e-1
            if prt_log:
                print("AlphaP/D<0.1 => set barr to 0.2*avrgCP = %f"%(barr))
                
        if 100 * ksmall > no_of_cp and barr < acp * 1.0e-1:
            barr = acp * 1.0e-1
            if prt_log:
                print("ksmall>1%% #cp => set barr to 0.1*avrgCP = %f"%(barr))
    
        # barr = averageCP * 1.0e-1;*/ /* removed for warmstart version */
        # barr = averageCP * 0.25;*/

        if iter <= 3:
            barr = acp * 1.0e-1
            if prt_log:
                print("Iter<=3 => set barr to 0.1*avrgCP = %f"%(barr))
                
        if iter == 1:
            barr = acp * 2.0e-1
            if prt_log:
                print("Iter==1 => set barr to 0.2*avrgCP = %f"%(barr))


        if err_bc>10.0*relGap and err_bc>1.e-5:
            if force_feas:
                barr = 0.5*acp
                print("FEASIBILITY LAGGING OPTIMALITY: barr = %.4e"%(barr))

        #averageDG = fabs(p_Obj - d_Obj) / no_of_cp;
        if barr < 0.05 * averageDG and iter > 3:
            barr = 0.05 * averageDG
            if prt_log:
                print("barr<avrgDG/20 => set barr to 0.05*avrgeDG = %f"%(barr))


        if IPM.out>=2:
            print("Avr. Gaps:  duality= %.4e, complementarity= %.4e"%(averageDG, acp))
            print("Iter= %d, barr= %.3e."%(iter, barr))


        return barr

    # getIPMDir() ----------------------------------------------------------
    def getIPMDir(self, x, y, z, t, s, w, xib, xic, xil, xiu, augSys, iDir, oldbarr, barr, alphaP, alphaD, dXp, dZp, dTp, dSp, dWp, ireps):
        """        
        Compute higher-order primal-dual directions.
        
        IN: pdPoint, xib, xic, xiu, xil, barr, pdRegx/y, theta, where_u
            pdPredDir, oldbarr         (for Mehrotra)
            pdNewDir, AlphaP, AlphaD      (for Gondzio)
        OUT: pdNewDir
   
        affine scaling direction:
        [dX] =  inv[TH A'] [xic - (barr*e-t.*z)./t + (barr*e-s.*w)./s
                      - w./s.*xiu- z./t.*xil]
        [dY]       [A  0 ] [xib]
        dT  = -xil + dX
        dZ  = (r_tz - z.*dT)./t
        dS  = xiu - dX
        dW  = (r_sw - w.*dS)./s
   
        Mehrotra:      (dbarr = barr-oldbarr), xib=xic=xiu=xil=0
        [dX] = inv[TH A'] [-((dbarr)*e - dTp.*dZp)./t + ((dbarr)*e-dSp.*dWp)./s ]
        [dY]       [A  0 ] [ 0 ]
        dT  = + dX
        dZ  = (r_tz - z.*dT)./t
        dS  = - dX
        dW  = (r_sw - w.*dS)./s
        
        Gondzio: a1P = min(1.d0, 1.08*AlphaP+0.08)
                 a1D = min(1.d0, 1.08*AlphaD+0.08)
                 
        dp_tz = (t+a1P*dT).*(z+a1D*dZ)
        r_tz = dp_tz<barr/10, ,>barr*10  :  barr-dp_tz, 0, -5*barr
        dp_sw = (s+a1P*dS).*(w+a1D*dW)
        r_sw = dp_sw<barr/10, ,>barr*10  :  barr-dp_sw, 0, -5*barr
        [dX] =  inv[TH A'] [-r_tz./t + r_sw./s ]
        [dY]       [A  0 ] [ 0 ]
        dT  = + dX
        dZ  = (r_tz - z.*dT)./t
        dS  = - dT
        dW  = (r_sw - w.*dS)./s
   
        For feasibility (and modified problem) only xib, xic, xiu, xil and the
        augmented system matrix change.
   
        The optional argument vXrhs_x (can be NULL) is added to the rhs of the
        augmented system before solving (currently only for the affine scaling
        direction).
   
        @param iDir      Type of direction to calculate:
                         - 1: primal-dual affine-scaling direction;
                         - 2: Mehrotra's corrector;
                         - 3: Gondzio's corrector.
        @param oldbarr   For Mehrotra's corrector (=barr used in aff scaling)
        @param barr      Barrier (mu) to use for direction
        @param alphaP    Largest primal step in current direction
                        (only for Gondzio corrector)
        @param alphaD    Largest dual step in current direction
                         (only for Gondzio corrector)
        @param ireps     Residual tolerance to use for iterative refinement
        """

        qp = self.qp
        
        if IPM.out>=2:
            print("getIPMDir: iDir= %d, barr= %.4e"%(iDir, barr))

        rhs = np.zeros(qp.n+qp.m)
        sol = np.zeros(qp.n+qp.m)
        rtz = np.zeros(qp.n)
        rsw = np.zeros(qp.n)
 


        if iDir == 1:
                
            # - - - Compute primal-dual affine-scaling direction - - - - */
            #rhs_x = [xic - (barr*e-x.*z)./x]
            #                       (this was without upper bounds)
            #         xic - (barr*e-t.*z)./t
            #            + (barr*e-s.*w)./s - w./s.*xiu- z./t.*xil
            #rhs_y = xib

            for i in range(qp.n):
                rhs[i] = xic[i]
                if qp.where_l[i]:
                    rtz[i] = (barr - t[i]*z[i])  # needed later
                    rhs[i] += -rtz[i]/t[i] - xil[i]*z[i]/t[i]; 
	
                if qp.where_u[i]:
                    rsw[i] = (barr - s[i]*w[i])  # needed later
                    rhs[i] += rsw[i]/s[i] -xiu[i]*w[i]/s[i] 

            for i in range(qp.m):
                rhs[qp.n+i] = xib[i]

            #print("rhs = ")
            #print(rhs)
        elif iDir == 2:
            # Mehrotra:      (dbarr = barr-oldbarr), xib=xic-xiu=xil=0
            # [dX] = inv[TH A'] [-((dbarr)*e -dTp.*dZp)./t
            #                                         +((dbarr)*e-dSp.*dWp)./s]
            # [dY]      [A  0 ] [ 0 ]
            # dT  = + dX
            # dZ  = (r_tz - z.*dT)./t
            # dS  = - dX
            # dW  = (r_sw - w.*dS)./s
            #
            # old:
            # [dX] = inv[TH A'] [-((dbarr)*e - dXp.*dZp)./x]
            # [dY]      [A  0 ] [ 0 ]
            # dZ  = (r_xz - z.*dX)./x


            dbarr = barr-oldbarr;
            for i in range(qp.n):
                rhs[i] = 0.0;
                if qp.where_l[i]:
                    rtz[i] = dbarr - dTp[i]*dZp[i] # needed later
                    rhs[i] -= rtz[i]/t[i]
                    
                if qp.where_u[i]:
                    rsw[i] = dbarr - dSp[i]*dWp[i] #/ needed later
                    rhs[i] += rsw[i]/s[i]
	
                    
            for i in range(qp.m):
                rhs[qp.n+i] = 0
      

    
        elif iDir == 3:
            # - - - - - - - Compute Gondzio's corrector - - - - - - */
            # Gondzio: a1P = min(1.d0, 1.08*AlphaP+0.08)
            #          a1D = min(1.d0, 1.08*AlphaD+0.08)
            # 
            # dp_tz = (t+a1P*dT).*(z+a1D*dZ)
            # r_tz = dp_tz<barr/10, ,>barr*10  :  barr-dp_tz, 0, -5*barr
            # dp_sw = (s+a1P*dS).*(w+a1D*dW)
            # r_sw = dp_sw<barr/10, ,>barr*10  :  barr-dp_sw, 0, -5*barr
            # [dX] =  inv[TH A'] [-r_tz./t + r_sw./s ]
            # [dY]       [A  0 ] [ 0 ]
            # dT  = + dX
            # dZ  = (r_tz - z.*dT)./t
            # dS  = - dT
            # dW  = (r_sw - w.*dS)./s
            # 
            # old:
            # dp_xz = (x+a1P*dX).*(z+a1D*dZ)
            # r_xz = dp_xz<barr/10, ,>barr*10  :  barr-dp_xz, 0, -5*barr
            # 
            # [dX] =  inv[TH A'] [-r_xz./x]
            # [dY]       [A  0 ] [ 0 ]
            # dZ  = (r_xz - z.*dX)./x

            alP = alphaP * 1.08 + 0.08
            alD = alphaD * 1.08 + 0.08

            if alP > 1.0:
                alP = 1.0
            if alD > 1.0:
                alD = 1.0

            # this here is done in Vector::GondzioCorr in OOPS
            # GondzioCorr: (componentwise)
            # dp = (x+alp dX)(z+ald dZ)
            # dp < 1e-1barr => r_xz_i = barr - dp
            # dp > 1e+1barr => r_xz_i = -5*barr
            # rhs_x_i = -r_xz_i/x_i

            for i in range(qp.n):
                rhs[i] = 0.0
                if qp. where_l[i]:
                    dp = (t[i]+alP*dTp[i])*(z[i]+alD*dZp[i])

                    # first build up r_tz in rhs[0:nVar-1] -> needed later
                    if dp<1e-1*barr:
                        rtz[i] = barr - dp
                    elif dp > 1e+1 * barr:
                        rtz[i] = -5*barr
                    else:
                        rtz[i] = 0.0

                    # then set rhs[i] = r_tz[i]/t[i];
                    rhs[i] -= rtz[i]/t[i]
	
                if qp.where_u[i]:
                    dp = (s[i]+alP*dSp[i])*(w[i]+alD*dWp[i])

                    # first build up r_sw in rhs[0:nVar-1] -> needed later
                    if dp<1e-1*barr:
                        rsw[i] = barr - dp
                    elif dp > 1e+1 * barr:
                        rsw[i] = -5*barr
                    else:
                        rsw[i] = 0.0

                    #then set rhs[i] = r_sw[i]/s[i];
                    rhs[i] += rsw[i]/s[i]

            for i in range(qp.m):
                rhs[qp.n+i] = 0

    
        # - - - - - - - end setting up rhs for different iDir - - - - - - - -


        sol = self.IterRefSolve(augSys, rhs, ireps)
        #print("augSys = ")
        #print(augSys)

        # set dx = sol[0:nVar-1]
        # set dz = (r_xz - z*dX)/x
        # set dy = sol[nVar:nVar+
        # dT  = + dX
        # dZ  = (r_tz - z.*dT)./t
        # dS  = - dT
        # dW  = (r_sw - w.*dS)./s

        #print("sol = ")
        #print(sol)

        dx = sol[0:qp.n]
        dy = sol[qp.n:qp.n+qp.m]

        ds = np.zeros(qp.n)
        dw = np.zeros(qp.n)
        dt = np.zeros(qp.n)
        dz = np.zeros(qp.n)
    
        for i in range(qp.n):
            if qp.where_l[i]:
                if iDir==1:
                    dt[i] = -xil[i] + dx[i]
                else:
                    dt[i] = dx[i]
	
                dz[i] = (rtz[i] - z[i]*dt[i])/t[i]

            if qp.where_u[i]:
                if iDir==1:
                    ds[i] =  xiu[i] - dx[i]
                else:
                    ds[i] = -dx[i]

                dw[i] = (rsw[i] - w[i]*ds[i])/s[i]


        return dx, dy, dz, dt, ds, dw


    # IterRefSolve() --------------------------------------------------------
    def IterRefSolve(self, augSys, rhs, ireps):
        """
        IterRefSolve solves the augmented system
            [-Q-TH^-1  A'] [del_x] = [rhs_x]
            [  A       0 ] [del_y] = [rhs_y]
        directly (by augmented System)
        
        IterRefSolve should take regularizations into account: the system
        that is actually solved is
        
           [-Q-TH2^{-1}-pdregx    A'   ] [del_x] = [rhs_x]
           [  A                pdregy  ] [del_y] = [rhs_y]
   
        where TH2 = TH/(1+TH*regthx) and TH2^{-1} = 1/(TH2+1e-12)
        
        Iterative Refinement: res_y = rhs_y - A*del_x - pdRegy*del_y
                            res_x = rhs_x + (Q+TH2^{-1}+pdregx)*delx - A'*del_y
                            del_x += [-Q-TH^-1  A']^-1 res_x
                            del_y += [ A        0 ]^-1 res_y
   
        The method only calls SolveCholesky (Assumes factors are in place)
        
        ireps = 0.1*(err_b+err_c);
        where err_b and err_c are the scaled reiduals
            err_b = || x.*inv(AAt)*xib||_infty
            err_c = || z.*xic||_infty
        """

        qp = self.qp

        do_scale = True

        if do_scale:
            max_rhs = LA.norm(rhs, 1)

            did_scale = False
            if max_rhs>1.0e6:
                did_scale = True
                rhs = 0.001*rhs

        sol = LA.solve(augSys, rhs)

        svsol = sol

        contIR = True
        iters=0

        while contIR:
            iters += 1

            # compute residuals, taking into account regTheta and pdreg
            #
            # [resx] = rhs_x + (Q+th2^{-1}+preg)*x - A'*y
            # [resy]   rhs_y - A*x - dreg*y  (I think that must be a "-" here)
            #
            # [resx] = - augSys*[x]  + preg.*x + rhs_x
            # [resy]            [y]  - dreg.*y + rhs_y


            res = np.dot(augSys, sol)

            res = rhs - res

            # correct for primal/dual regularization: pointless here since
            # we don't do primal/dual reg.
            # for(int i=0;i<nVar;i++){
            #    res[i] += augSys.preg[i]*sol[i];
            # }
            # 
            # for(int i=0;i<nCon;i++){
            #    res[nVar+i] -= augSys.dreg[i]*sol[nVar+i];
            # }

            # get largest element in x and y part of residual
            mresx = LA.norm(res[0:qp.n], 1)
            mresy = LA.norm(res[qp.n:qp.n+qp.m], 1)
            mresy = 0.0;
            if IPM.out>=2:
                print("IterRef iter %d: mresx+y = %f"%(iters, mresx+mresy))

            # in the first iteration just set the old residuals to the
            # current ones
            if iters == 1:
                mresx0 = mresx
                mresy0 = mresy


            if mresx + mresy < 1.0e-8:
                contIR = False
            # /*if (mresx + mresy < 1.0e-7 && iters >= 2)
            # contIR = 0;*/
            if iters >= 3 and mresx + mresy<ireps:
                contIR = False
            if iters >= 8:
                contIR = False
            if mresx + mresy > 1.0e+6:
                Alarm = 2
                contIR = False
                
            if mresx + mresy > mresx0 + mresy0 + 1.e-12:
                #// take old solution

                sol = svsol
                mresx = mresx0
                mresy = mresy0

                if IPM.out>=2:
                    print("IterRefSolve: error growth in LLt factors.")
                contIR = False

            if contIR:
                # Repeat solution of the augmented system for residuals. */
                # Correct current solution. */

                dxy = LA.solve(augSys, res)

                sol += dxy


        if did_scale:
            sol = 1000.0*sol
                
        return sol


    # getMaxStep() ----------------------------------------------------------
    def getMaxStep(self, p, dp):

        # we know that p > 0 -> only interested in dp <0
        # => -p/dp >= 0

        neg = dp<0
        ratio = -p[neg]/dp[neg]

        try:
            ms = np.min(ratio)
        except ValueError:  #raised if 'ratio' is empty.
            ms = 1.0e20

        return ms

    # computeMehrotraBarr() --------------------------------------------------
    def computeMehrotraBarr(self, alphaP, alphaD, t, z, s, w, dt, dz, ds, dw, oldbarr, av, iter):

        qp = self.qp
        #Step0 = min(alphaP, alphaD)
        Step0 = alphaP;
        if alphaD < Step0:
            Step0 = alphaD

        #Compute current complementarity gap:  ttzstw = T'*Z + S'*W.
        sxz = 0.0
        dp = 0
        for i in range(qp.n):
            if qp.where_l[i]:
                sxz += t[i]*z[i]
                dp += 1
      
            if qp.where_u[i]:
                sxz += s[i]*w[i]
                dp += 1

        # Compute the minimum complementarity gap that can be achieved
        # when moving in a primal-dual affine scaling direction:
        # GapMin =(t+ALPHAP*dt)'*(z+ALPHAD*dz)+(s+ALPHAP*ds)'*(w+ALPHAD*dw). 

        GapMin = 0
        for i in range(qp.n):
            if qp.where_l[i]:
                GapMin += (t[i]+alphaP*dt[i])*(z[i]+alphaD*dz[i])
            if qp.where_u[i]:
                GapMin += (s[i]+alphaP*ds[i])*(w[i]+alphaD*dw[i])

        barr = (GapMin * GapMin * GapMin) / (sxz * sxz * dp)

        barrmx = 0.33
        if Step0 > 0.1:
            barrmx = 0.2
        if Step0 > 0.2:
            barrmx = 0.1
        if barr > barrmx * GapMin / dp:
            barr = barrmx * GapMin / dp
        if barr < oldbarr:
            barr = oldbarr
        if iter <= 3:
            barr = av * 1.0e-1
        if iter == 1:
            barr = av * 2.0e-1


        if IPM.out>=3:
            print("ttzstw= %.3e, GapMin= %.3e, Mehrotra: barr= %.3e"%(sxz, GapMin, barr))

        return barr

    # ================= access methods for the solution =====================

    # getSol() --------------------------------------------------
    def getSol(self):
        return self.solx

    # getDualBnd() --------------------------------------------------
    def getDualsBnd(self):
        return self.solz

    # getSol() --------------------------------------------------
    def getDualsCons(self):
        return self.soly


    
