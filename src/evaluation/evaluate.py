from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.evaluation.confusion_matrix import simple_confusion
from src.evaluation.metrics import precision, recall, f1_score

def main():
    cfg = load_config("config.yaml")
    logger = get_logger("evaluate", cfg["logging"]["level"], cfg["logging"]["log_dir"])

    pred_count = 120
    gt_count = 100

    tp, fp, fn = simple_confusion(pred_count, gt_count)
    p = precision(tp, fp)
    r = recall(tp, fn)
    f1 = f1_score(p, r)

    logger.info(f"Eval Results (placeholder): TP={tp}, FP={fp}, FN={fn}")
    logger.info(f"Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    main()
