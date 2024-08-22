import numpy as np

class QP:
    """
    This class describes a Quadratic Programming problem

    min c'*x + 0.5 x'*Q*x,   subject to, Ax = b,  bl < =x < =bu

    A QP can be created by using

    qp =  QP.QP(c, Q, A, b, bl, bu)
    
    all parameters c, Q, A, b, bl, bu have to be of type np.array
    (of appropriate dimension)

    """

    def __init__(self, c, Q=None, A=None, b=None, bl=None, bu=None):
        self.c = c
        self.Q = Q
        self.A = A
        self.b = b
        self.bl = bl
        self.bu = bu

        self.n = c.size
        self.m = b.size

        self.where_l = self.bl > -np.infty
        self.where_u = self.bu < np.infty
        
