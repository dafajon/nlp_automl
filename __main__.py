from sadedegel.extension.sklearn import OnlinePipeline, TfidfVectorizer, Text2Doc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

import joblib
import pandas as pd
import numpy as np
import optuna

from functools import partial

from loguru import logger
from rich.console import Console
import os
import warnings
import json

from uuid import uuid4

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

console = Console()

__MODEL_ZOO__ = [("sgd", SGDClassifier),
                 ("logreg", LogisticRegression),
                 ("pa", PassiveAggressiveClassifier)]


def validate(loader):
    def wrapper(*args, **kwargs):
        df = loader(*args, **kwargs)
        if 'ID' not in list(df.columns):
            raise ValueError("Column named \"ID\" is not included. Make sure you include or "
                             "rename column if a such data exists")
        if 'text' not in list(df.columns):
            raise ValueError("Column named \"text\" is not included. Make sure you include or "
                             "rename column if a such data exists")
        if 'class' not in list(df.columns):
            raise ValueError("Column named \"class\" is not included. Make sure you include or "
                             "rename column if a such data exists")
        return df
    return wrapper


def standardize_logs(param_dict):
    params = {
        "model_params": {},
        "preprocessing_params": {}
    }

    keys = param_dict.keys()
    for key in keys:
        if key.startswith("hyper_"):
            param_name = key.split("hyper_")[-1][:-3]
            params["model_params"][param_name] = param_dict[key]
        elif key.startswith("preprocessing_"):
            param_name = key.split("preprocessing_")[-1]
            params["preprocessing_params"][param_name] = param_dict[key]

    return params


def inference(trainer):
    def wrapper():
        run_id = trainer()
        test_df = pd.read_csv("data/test.csv")

        pipeline = joblib.load(f"models/pipeline_{run_id}.joblib")

        predictions = pipeline.predict(test_df["text"])
        test_df['pred'] = predictions
        sub_df = test_df[['ID', 'pred']]
        sub_df.to_csv(f"submissions/submission_{run_id}.csv", index=False)

    return wrapper


def train_optimal(runner):
    def wrapper():
        run_id = runner()

        try:
            with open(f"logs/run_{run_id}.json", "r") as logfile:
                best = json.load(logfile)
        except FileNotFoundError:
            logger.info("json file for best parameters of this run is not found. "
                        "Appearently the run is interrupted before any trial was concluded.")

        modelname = best[0]['params']['model']
        params = standardize_logs(best[0]['params'])

        model_dict = {"pa": PassiveAggressiveClassifier,
                      "sgd": SGDClassifier,
                      "logreg": LogisticRegression}

        text2doc = Text2Doc(tokenizer="icu")
        preprocessor = TfidfVectorizer(**params["preprocessing_params"])
        model = model_dict[modelname](**params["model_params"])

        pipeline = OnlinePipeline([("text2doc", text2doc),
                                   ("tfidf", preprocessor),
                                   ("model", model)])

        df = load_data("data/cagri_df.csv.gz").sample(100)
        pipeline.fit(df["text"], df["class"])

        joblib.dump(pipeline, f"models/pipeline_{run_id}.joblib")

        return run_id

    return wrapper


@validate
def load_data(datapath: str):
    df = pd.read_csv(datapath, usecols=['ID', 'text', 'class'])
    return df


def optimize(trial, data):
    # ============================FEATURE SPACE===============================
    tf_m = trial.suggest_categorical("preprocessing_tf_method", ["binary", "raw", "freq", "log_norm"])
    idf_m = trial.suggest_categorical("preprocessing_idf_method", ["smooth", "probabilistic"])

    tfidf = TfidfVectorizer(tf_method=tf_m, idf_method=idf_m, show_progress=True)

    # =============================MODEL SPACE================================
    model_name, model = trial.suggest_categorical("model", __MODEL_ZOO__)
    if model_name == 'logreg':
        params = {
            "C": trial.suggest_float("hyper_C_lr", 1e-3, 1e1),
            "class_weight": trial.suggest_categorical("hyper_class_weight_lr", ["balanced", None])
        }
    elif model_name == 'sgd':
        params = {
            "alpha": trial.suggest_float("hyper_alpha_sg", 1e-10, 10, log=True),
            "penalty": trial.suggest_categorical("hyper_penalty_sg", ["elasticnet", "l2"]),
            "eta0": trial.suggest_uniform('hyper_eta0_sg', 1e-5, 1),
            "learning_rate": trial.suggest_categorical("hyper_learning_rate_sg", ["adaptive", "optimal"]),
            "class_weight": trial.suggest_categorical("hyper_class_weight_sg", ["balanced", None])
        }
    elif model_name == 'pa':
        params = {
            "C": trial.suggest_float("hyper_C_sg", 1e-10, 10, log=True),
            "average": trial.suggest_categorical("hyper_average_sg", [True, False]),
            "class_weight": trial.suggest_categorical("hyper_class_weight_sg", ["balanced", None])
        }

    transformed_text = Text2Doc(tokenizer="icu").transform(data["text"])
    X, y = tfidf.transform(transformed_text), data["class"]
    classifier = model(**params)
    f1_macro = make_scorer(f1_score, average="macro")

    score = cross_val_score(classifier, X, y, n_jobs=-1,
                            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                            scoring=f1_macro)

    return score.mean()


def log_best_callback(study, trial, run_id):
    logger.info(study.best_params)

    best_params = study.best_params
    best_params['model'] = best_params['model'][0]
    with open(f"logs/run_{run_id}.json", "w") as jfile:
        json.dump([{"score": study.best_value,
                    "params": best_params}], jfile)


def log_progress_callback(study, trial, run_id, total_trials):
    print(trial.number / total_trials * 100)
    print(trial.params)
    print(trial.value)


#@inference
@train_optimal
def run():
    total_trials = 2
    run_id = str(uuid4())
    logger.add(f"logs/run_{run_id}.log")

    df = load_data("data/cagri_df.csv.gz").sample(100)
    objective = partial(optimize, data=df)
    best_callback = partial(log_best_callback, run_id=run_id)
    progress_callback = partial(log_progress_callback, run_id=run_id, total_trials=total_trials)

    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=total_trials, callbacks=[best_callback, progress_callback])

        console.log(study.best_params)

        return run_id
    except KeyboardInterrupt:
        logger.info("Interrupted by user checking logs for this run. "
                    "If a log file exists final model will be trained from latest checkpoint.")
        return run_id


if __name__ == "__main__":
    run()
