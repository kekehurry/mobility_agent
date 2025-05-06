from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from .base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self, train_file=None,eval_file=None, model_path=None,save_dir='models/random_forest',sample_num=1000, seed=42):
        super().__init__(train_file=train_file,eval_file=eval_file, model_path=model_path,sample_num=sample_num, seed=seed,save_dir=save_dir)
        base_model = RandomForestClassifier(random_state=seed)
        self.model = MultiOutputClassifier(base_model)
        self.param_grid = {
            'estimator__n_estimators': [10, 50, 100],
            'estimator__max_depth': [None, 10, 20],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4]
        }