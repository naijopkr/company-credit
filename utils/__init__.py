def print_cm(pred_cm):
    pred_false, pred_true = pred_cm
    tn, fn = pred_false
    fp, tp = pred_true
    print(
        f'TN\tFN\tFP\tTP\n{tn}\t{fn}\t{fp}\t{tp}'
    )
