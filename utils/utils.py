import functools

def rgetattr(obj, path, default):
    """
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    try:
        return functools.reduce(getattr, path.split(), obj)
    except AttributeError:
        return default

def get_region(monkey: str, array: int):
    """
    Get the brain region of an array for a given monkey.
    """
    region_dict = {
        'monkeyN' : {
            range(0, 8) : 'V1',
            range(8, 12) : 'V4',
            range(12, 16) : 'IT',
        },
        'monkeyF' : {
            range(0, 8) : 'V1',
            range(8, 13) : 'IT',
            range(13, 16) : 'V4',
        }
    }
    monkey_dict = region_dict.get(monkey, {})
    for idx_range, region in monkey_dict.items():
        if array in idx_range:
            return region
    raise ValueError(f"Invalid array index {array} for monkey {monkey}")
