def find_score_thr(results, min_p=0.80, min_r=0.85):
    """Find score threshold, at the best F-score.

    Args:
        results (list): List of `(file_name, target, predict, dt, gt)`.
            Via `simplecv.mmdet_v1.ext_utils.translate.trans_test_results()`.
        min_p (float): Accepted minimum call precision.
        min_r (float): Accepted minimum recall rate.
    Returns:
        score_thr (dict): Such as `dict(CODE1=S1,CODE2=S2,...)`.
    """
    pass
