def make_repetition_penalty(*args, **kwargs):
    # Return a simple callable or marker; runner only checks presence
    return lambda *a, **k: None


def make_sampler(*args, **kwargs):
    # Return a simple callable representing sampler
    return lambda *a, **k: None

