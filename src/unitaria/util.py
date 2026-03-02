def logreduce(f, values):
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    half = len(values) // 2
    return f(logreduce(f, values[:half]), logreduce(f, values[half:]))
