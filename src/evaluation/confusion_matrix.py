def simple_confusion(pred_count: int, gt_count: int):
    """Placeholder confusion estimate."""
    tp = min(pred_count, gt_count)
    fp = max(0, pred_count - gt_count)
    fn = max(0, gt_count - pred_count)
    return tp, fp, fn
