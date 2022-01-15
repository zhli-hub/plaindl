class product(object):
    """
    product(*iterables, repeat=1) --> product object

    Cartesian product of input iterables.  Equivalent to nested for-loops.

    For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B).
    The leftmost iterators are in the outermost for-loop, so the output tuples
    cycle in a manner similar to an odometer (with the rightmost element changing
    on every iteration).

    To compute the product of an iterable with itself, specify the number
    of repetitions with the optional repeat keyword argument. For example,
    product(A, repeat=4) means the same as product(A, A, A, A).

    product('ab', range(3)) --> ('a',0) ('a',1) ('a',2) ('b',0) ('b',1) ('b',2)
    product((0,1), (0,1), (0,1)) --> (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) ...
    """

    def __getattribute__(self, *args, **kwargs):  # real signature unknown
        """ Return getattr(self, name). """
        pass

    def __init__(self, *iterables, repeat=1):  # known case of itertools.product.__init__
        """ Initialize self.  See help(type(self)) for accurate signature. """
        return []

    def __iter__(self, *args, **kwargs):  # real signature unknown
        """ Implement iter(self). """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __next__(self, *args, **kwargs):  # real signature unknown
        """ Implement next(self). """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        """ Return state information for pickling. """
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        """ Set state information for unpickling. """
        pass

    def __sizeof__(self, *args, **kwargs):  # real signature unknown
        """ Returns size in memory, in bytes. """
        pass
