from xgboost import XGBClassifier


def create_classifier(params: dict, seed: int, threads: int, positive_weight: float, patience: int):
    return XGBClassifier(**params, objective='binary:logistic', eval_metric='aucpr',
                         scale_pos_weight=positive_weight, random_state=seed, n_jobs=threads,
                         early_stopping_rounds=patience)
