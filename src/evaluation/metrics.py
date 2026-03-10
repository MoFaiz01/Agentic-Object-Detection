def precision(tp, fp):
    return tp / (tp + fp + 1e-12)

def recall(tp, fn):
    return tp / (tp + fn + 1e-12)

def f1_score(p, r):
    return 2 * p * r / (p + r + 1e-12)
