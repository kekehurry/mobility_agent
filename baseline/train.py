from .random_forest import RandomForest
from .xgboost import XGBoost
from .multilayer_perceptron import MultilayerPerceptron
import random

def train_baseline_model(num_samples,model_key,train_file,eval_file,seed=None):
    if not seed:
        seed = random.randint(0,9999)
    if model_key == "RF":
        model = RandomForest(train_file=train_file, eval_file=eval_file, sample_num=num_samples, seed=seed)
    elif model_key == "XGB":
        model = XGBoost(train_file=train_file, eval_file=eval_file, sample_num=num_samples, seed=seed)
    elif model_key == "MLP":
        model = MultilayerPerceptron(train_file=train_file, eval_file=eval_file, sample_num=num_samples, seed=seed)
    else:
        return None
    model.train()
    kl_df, overall_kl, overall_mae = model.evaluate()
    return {
        "model": model_key,
        "seed": seed,
        "num_samples": num_samples,
        "overall_kl": overall_kl,
        "overall_mae": overall_mae
    }