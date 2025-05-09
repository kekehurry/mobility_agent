import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from .base_model import BaseModel

class XGBoost(BaseModel):
    def __init__(self, train_file=None,eval_file=None, model_path=None,save_dir='models/xgboost',sample_num=1000, seed=None):
        super().__init__(train_file=train_file,eval_file=eval_file, model_path=model_path,sample_num=sample_num, seed=seed,save_dir=save_dir)
        base_model = xgb.XGBClassifier(objective='multi:softprob')
        self.model = MultiOutputClassifier(base_model)
        self.param_grid = {
            'estimator__n_estimators': [10, 50, 100],
            'estimator__max_depth': [None, 10, 20],
            'estimator__learning_rate': [1e-5, 1e-4, 1e-3],
            'estimator__subsample': [0.7, 0.8, 0.9],
            'estimator__gamma': [0, 0.1, 0.2],
        }