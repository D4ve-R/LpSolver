import numpy as np
import time
from typing import Union, List, Tuple

from .simplex import SIMPLEX, DUAL_SIMPLEX, validateBase
from .errors import ShapeError

# maximization linear problem
class MaxLp:

    def __init__(self, constrMtx: Union[np.ndarray,List[List[int]]], bounds: Union[np.ndarray,List[int]], coef: Union[np.ndarray,List[int]], equos=None):

        # objective value
        self._objective = None

        # decision variables
        self._x = None
 
        # dual variables
        self._duals = None

        if isinstance(bounds, List):
            self.bounds = np.array(bounds)
        else:
            self.bounds = bounds
        
        if isinstance(coef, List):
            self.coef = np.array(coef)
        else:
            self.coef = coef

        if isinstance(constrMtx, List):
            self.constrMtx = np.array(constrMtx)
        else:
            self.constrMtx = constrMtx

        if self.constrMtx.shape[0] != self.bounds.shape[0]:
            raise ShapeError(f"A has {self.constrMtx.shape[0]} and b has {self.bounds.shape[0]} rows")

        if self.constrMtx.shape[1] != self.coef.shape[0]:
            raise ShapeError(f"A has {self.constrMtx.shape[1]} columns and c has {self.coef.shape[0]} rows")


        # add unit matrix for slack variables
        self.constrMtx = np.concatenate((self.constrMtx, np.eye(self.bounds.shape[0])), axis=1)

        # add zeros for slack variables
        self.coef = np.concatenate((self.coef, np.zeros(self.constrMtx.shape[0])))

        # TODO
        # handle >= constraints
        # currently row *= -1


    def solve(self, verbose: int=0) -> Tuple[Union[float, None],Union[np.ndarray, None],Union[np.ndarray, None]]:
 
        start = time.monotonic()

        # check if slack variables are valid initial base
        if not validateBase(self.constrMtx[:,[i for i in range(-self.constrMtx.shape[0], 0)]], self.bounds):
            # run dual simplex
            self._objective, self._x, self._duals = DUAL_SIMPLEX(self.constrMtx, self.bounds, self.coef)

        else:
            # run simplex algorithm
            self._objective, self._x, self._duals = SIMPLEX(self.constrMtx, self.bounds, self.coef)

        if verbose > 0:
            print("objective value: ", self._objective)
            print("x: ", self._x)

            if verbose > 1:
                print(f"Execution time {format(time.monotonic() - start, '.5f')}s")
                print("y: ", self._duals)


        return self._objective, self._x, self._duals


    def __str__(self):
        ret = ""

        if isinstance(self, MaxLp):
            ret += "Max Linear Problem\n"
        elif isinstance(self, MaxIp):
            ret += "Max Integer Problem\n"
        elif isinstance(self, MinIp):
            ret += "Min Integer Problem\n"
        else:
            ret += "Min Problem\n"

        ret += f"c: {self.coef}\n"
        ret += f"A: \n{self.constrMtx}\n"
        ret += f"b: {self.bounds}\n"
        
        if self._objective is not None:
            ret += f"objective value: {self._objective}\n"
            ret += f"x: {self._x}\n"
            ret += f"y: {self._duals}\n"

        return ret


# minimization linear problem
class MinLp(MaxLp):

    def solve(self, verbose: int=0):

        # convert min to max
        # min f -> -max -f 
        self.coef *= -1

        super().solve(verbose)
        
        if self._objective is not None:
            self._objective *= -1

        return self._objective, self._x, self._duals


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
        
        # is this correct ?
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
        
        super.solve(verbose)

        if self._objective is not None:
            self._objective *= -1

        return self._objective, self._x, self._duals


