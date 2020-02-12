# Exceptions defined by CellBender


class NanException(Exception):
    """Exception raised when a NaN is present.

    Attributes:
        param: Name of parameter containing the NaN
    """

    def __init__(self, param):
        self.param = param
        self.message = 'A wild NaN appeared!  In param ' + self.param
