from lightgbm import LGBMClassifier


def create_classifier(params: dict, seed: int, threads: int, positive_weight: float):
    return LGBMClassifier(**params, objective='binary', metric='average_precision',
                          scale_pos_weight=positive_weight, random_state=seed, n_jobs=threads,
                          verbosity=-1, deterministic=True, force_col_wise=True)
