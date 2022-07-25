import numpy as np
from typing import List, Union, Tuple

from .errors import UnboundedError, InfeasibleError, InvalidBaseError

# implements simplex algorithm in standard form
# contraint matrix A must contain slack variables
# max c.T * x
# s.t A * x <= b
def SIMPLEX(A: np.ndarray, b: np.ndarray, c: np.ndarray, NB: List=None, B: List=None) -> Tuple[Union[None,np.ndarray], np.ndarray, np.ndarray]:
    
    # init objective value
    objective = None

    # init decision variables x 
    x = np.zeros(A.shape[1])
 
    # init dual variables y
    y = np.zeros(A.shape[0])
    
    # init bases, if not given
    # use slack variables for init base
    if NB is None and B is None:
        # nonbase indices
        NB = list(range(A.shape[1] - A.shape[0]))
        # base indices
        B = list(range(A.shape[1] - A.shape[0], A.shape[1]))

    # check if base is valid 
    # xB = inv(AB) * b !<= 0
    x[B] = b.dot(np.linalg.inv(A[:,B]))

    if np.amin(x[B]) < 0:
        #raise InvalidBaseError(f"base indices: {B} are invalid")
        raise InfeasibleError

    while True:

        # solve Ab.T * y = cb
        y = np.linalg.solve(A[:,B].T, c[B])

        # pricing
        # solve cn' = cn - An.T * y 
        cn = c[NB] - np.matmul(A[:,NB].T, y)

        j = -1
        
        # bland's rule, select lowest index of item > 0
        for i in range(cn.shape[0]):
            if cn[i] > 0:
                j = NB[i]
                break

        # if all items of cn' <= 0, optimal solution found
        if j == -1:
            objective = np.sum(c * x)
            break

        # solve Ab * w = Aj
        w = np.linalg.solve(A[:,B], A[:,j])

        # ratio test
        # if w <= 0, lp is unbounded
        if np.amax(w) <= 0: 
            raise UnboundedError
            
        # disable runtime warning zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            t = x[B] / w

        # convert negative value to infity to ignore them
        t = np.where(t > 0, t, np.inf)

        # bland's rule, get lowest index of min val 
        i = np.nanargmin(t)
        
        # xj = t
        x[j] = t[i] 

        # xb = xb - t * w
        x[B] -= t[i] * w

        # update nonbase indices
        NB.remove(j)
        NB.append(B[i])
        NB.sort()

        # update base indices
        B.remove(B[i])
        B.append(j)
        B.sort()

    return objective, x, y


def DUAL_SIMPLEX(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[Union[None,np.ndarray], np.ndarray, np.ndarray]:
    
    # lp -> dlp 

    _b = b
    _A = A

    # b !>= 0
    while np.amin(_b) < 0:
        idx = np.argmin(_b)
        _b[idx] *= -1
        _A[idx] *= -1

    # min sum(y)
    # Ax + y 
    _A = np.concatenate((_A, np.eye(A.shape[0])), axis=1)

    _c = np.zeros(_A.shape[1])
    # min f -> - max -f
    _c[-A.shape[0]:] = -1
    
    # nonbase indices
    NB = list(range(_A.shape[1] - b.shape[0]))
    # base indices
    B = list(range(_A.shape[1] - b.shape[0], _A.shape[1]))

    # solve dual lp
    try:
        obj,x,duals = SIMPLEX(_A, _b, _c, NB, B)
    except UnboundedError:
        raise InfeasibleError
    except InfeasibleError:
        raise UnboundedError

    if obj is not None and obj == 0:
        # TODO
        # check for degenerated solution
        # special case if x[B] = 0 = y'

        # construct new nonbase indices based of new base indices
        NB = [i for i in range(A.shape[1]) if not i in B]
        
        # solve simplex with optimal initial base
        return SIMPLEX(A, b, c, NB, B)

    else:
        raise InfeasibleError
    
    

def validateBase(AB: np.ndarray, b: np.ndarray) -> bool:
    # xB = inv(AB) * b 
    # is valid if xB <= 0
    if np.amin(b.dot(np.linalg.inv(AB))) < 0:
        return False

    return True


