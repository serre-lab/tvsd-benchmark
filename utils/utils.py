import functools

def rgetattr(obj, path, default):
    """
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    try:
        return functools.reduce(getattr, path.split(), obj)
    except AttributeError:
        return default
