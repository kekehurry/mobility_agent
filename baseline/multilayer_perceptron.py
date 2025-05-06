from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from .base_model import BaseModel

class MultilayerPerceptron(BaseModel):
    def __init__(self, train_file=None,eval_file=None, model_path=None,save_dir='models/multilayer_perceptron',sample_num=1000, seed=42):
        super().__init__(train_file=train_file,eval_file=eval_file, model_path=model_path,sample_num=sample_num, seed=seed,save_dir=save_dir)
        base_model = MLPClassifier(max_iter=20000, random_state=self.seed)
        self.model = MultiOutputClassifier(base_model)
        self.param_grid = {
            'estimator__hidden_layer_sizes': [(50, 50), (100, 100), (150, 150)],
            'estimator__activation': ['tanh', 'relu'],
            'estimator__alpha': [1e-5, 1e-4, 1e-3],
            'estimator__learning_rate_init': [1e-4, 1e-3, 1e-2],
        }