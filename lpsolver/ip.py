import numpy as np
import time
from typing import Union, Tuple

from .lp import MaxLp
from .simplex import SIMPLEX, DUAL_SIMPLEX, validateBase


# maximization integer Problem 
# using branch and bound algorithm
class MaxIp(MaxLp):

    def solve(self, verbose: int=0) -> Tuple[Union[float,None],Union[np.ndarray,None],Union[np.ndarray,None]]:
        self.verbose = verbose

        start = time.monotonic()

        # solve linear relaxation for integer problem
        # check if lp problem is feasible & bounded
        relax,_,_ = super().solve() 

        if relax is not None:
            self._objective, self._x, self._duals = self._solve(self.constrMtx, self.bounds, self.coef, self._x)
        
        if verbose > 0:
            print("Objective value: ", self._objective)
            print("X: ", self._x)

            if verbose > 1:
                print(f"Execution time {format(time.monotonic() - start, '.5f')}s")
                print("Duals: ", self._duals)

        return self._objective, self._x, self._duals


    def _solve(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray) -> Tuple[Union[float,None],Union[np.ndarray,None],Union[np.ndarray,None]]:
        if all([i % 1 < 1e-4 for i in x]):

            if c.shape != x.shape:
                c = np.append(c, np.zeros(x.shape[0] - c.shape[0]))

            self._objective = np.sum(c * x)
            self._x = x

            return self._objective, self._x, self._duals

        if self.verbose > 0:
            print("x: ", x)
 
        # convert integer to negative infinity
        # get index of largest fractional
        idx = np.argmax(np.where(x % 1 != 0, x, -np.inf))

        # construct new row for contraint matrix
        tmp = np.zeros(A.shape[1])
        tmp[idx] = 1
        A = np.append(A, [tmp], axis=0)

        # construct new column for contraint matrix
        tmp = np.zeros((A.shape[0],1))

        # x <= floor(fractional)
        tmp[-1] = 1
        A1 = np.append(A, tmp, axis=1)
        b1 = np.append(b, np.floor(x[idx]))
 
        # x >= ceil(fractional)
        tmp[-1] = -1
        A2 = np.append(A, tmp, axis=1)
        A2[-1,idx] = -1
        b2 = np.append(b, -np.ceil(x[idx]))

        c = np.append(c, 0)

        # solve linear relaxation of sub problems
        if validateBase(A1[:,[i for i in range(-A1.shape[0], 0)]], b1):
            obj1, x1, y1 = SIMPLEX(A1, b1, c)
        else:
            obj1, x1, y1 = DUAL_SIMPLEX(A1, b1, c)

        obj2, x2, y2 = DUAL_SIMPLEX(A2, b2, c)
        
        # TODO: is this correct ?
        if obj1 is not None and obj2 is not None:
            if obj1 >= obj2:
                self._duals = y1
                return self._solve(A1, b1, c, x1)
            else:
                self._duals = y2
                return self._solve(A2, b2, c, x2)

        if obj1 is not None:
            self._duals = y1
            return self._solve(A1, b1, c, x1)
            
        if obj2 is not None:
            self._duals = y2
            return self._solve(A2, b2, c, x2)

        return self._objective, self._x, self._duals


# minimization integer problem
# using branch and bound
class MinIp(MaxIp):

    def solve(self, verbose: int=0):

        # convert min to max
        # min f -> max -f 
        self.coef *= -1
        
        super().solve(verbose)

        if self._objective is not None:
            self._objective *= -1

        return self._objective, self._x, self._duals


