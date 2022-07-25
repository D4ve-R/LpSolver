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
            raise ShapeError(f"A.shape: {self.constrMtx.shape}; b.shape: {self.bounds.shape}")

        if self.constrMtx.shape[1] != self.coef.shape[0]:
            raise ShapeError(f"A.shape: {self.constrMtx.shape}; c.shape: {self.coef.shape}")

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

        if isinstance(self, MinIp):
            ret += "Min Integer Problem\n"
        elif isinstance(self, MaxIp):
            ret += "Max Integer Problem\n"
        elif isinstance(self, MinLp):
            ret += "Min Linear Problem\n"
        else:
            ret += "Max Linear Problem\n"

        ret += f"c: {self.coef}\n"
        ret += f"A: \n{self.constrMtx}\n"
        ret += f"b: {self.bounds}\n"
        
        if self._objective is not None:
            ret += f"objective value: {self._objective}\n"
            ret += f"x: {self._x}\n"
            ret += f"y: {self._duals}\n"

        return ret

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, val):
       raise Exception("objective value is protected and can not be set")
 
    @property
    def x(self):
       return self._x
    
    @x.setter
    def x(self, val):
       raise Exception("x is protected and can not be set")

    @property
    def duals(self):
       return self._duals

    @duals.setter
    def duals(self):
       raise Exception("duals is protected and can not be set")



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


