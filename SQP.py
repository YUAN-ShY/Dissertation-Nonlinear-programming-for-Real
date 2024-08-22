import numpy as np
import time

import NonlinearProblem as NLP
import SQP_KKT as KKT
import SQP_PhaseOne as Ph1
import SQP_Main as SQP


nlp = NLP.NonlinearProblem('sonoco97.nl')
# nlp = NLP.NonlinearProblem('haverly1_pq.nl')
# nlp = NLP.NonlinearProblem('haverly1_qq.nl')
# nlp = NLP.NonlinearProblem('smallNLP3.nl')

# set the bound of start points
x_l = np.copy(nlp.bl)
x_u = np.copy(nlp.bu)
x_l[np.isinf(nlp.bl)] = -50
x_u[np.isinf(nlp.bu)] = 50

# generate random start points
total_num = 30
np.random.seed(42)
x_list = np.random.uniform(low=x_l, high=x_u, size=(total_num, x_l.shape[0]))

num = 0
iteration = 0
out = 2

if out==2:
    print(f"{'start point':<15}{'Ph1':<10}{'Converge':<15}{'KKT':<15}{'Object value':<25}")
    

start_time = time.time()

for i, x in enumerate(x_list):

    rhoSQP = 3
    total_iter = 0

    # do the SQP and Phase one of a start point
    while True:
        # SQP mian phase
        x, find, iter = SQP.do_SQP(x, nlp, approxi_L='Separate', update_QN="BFGS", max_iter=80, do_QuasiNewton_logic=True, rho=rhoSQP, tol=1e-6, out=out)
         
        total_iter += iter

        # if converge of SQP main phase then stop
        if find:
            break 
        
        if out==0:
            print("-----------------------------------------------------------------------------------")
        

        # feasibility restoration phase
        x, find_Ph1, rhoSQP, iter = Ph1.Ph1(nlp, x, gama=1, rhoPh1=10, tol=1e-4, max_iter=100, out=out)

        total_iter += iter

        if out==0:
            print("-----------------------------------------------------------------------------------")

        # if infeasible of feasibility restoration phase problem then stop
        if not find_Ph1:
            break

    if out==0:
        print(f"{'start point':<15}{'Ph1':<10}{'Converge':<15}{'KKT':<15}{'Object value':<25}")
        print(f"{i+1:<15}{'Fail' if not find_Ph1 else 'Success':<10}{total_iter:<15}{'True' if KKT.is_KKT(nlp, x) else 'False':<15}{nlp.obj(x):<25}")

    if out==2:
        print(f"{i+1:<15}{'Fail' if not find_Ph1 else 'Success':<10}{total_iter:<15}{'True' if KKT.is_KKT(nlp, x) else 'False':<15}{nlp.obj(x):<25}")
    

    # calculate the number of points successfully converge to stationary point
    if KKT.is_KKT(nlp, x):
        iteration += total_iter
        num += 1
    

end_time = time.time()


percentage = (num / total_num) * 100
time_per_operation = (end_time - start_time) / total_num
per_iteration = iteration / total_num

print(f'&{time_per_operation:.6f}&{percentage:.2f}&{per_iteration:.2f}')