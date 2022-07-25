class ShapeError(Exception):
    def __init__(self, msg: str=""):
        super().__init__("ShapeError: " + msg)

class InvalidBaseError(Exception):
    def __init__(self, msg: str):
        super().__init__("InvalidBaseError: " + msg)

class UnboundedError(Exception):
    def __init__(self, msg: str=""):
        super().__init__("Lp is Unbounded:\n" + msg)

class InfeasibleError(Exception):
    def __init__(self, msg: str=""):
        super().__init__("Lp is infeasible:\n" + msg)
