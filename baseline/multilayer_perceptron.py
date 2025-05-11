from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from .base_model import BaseModel

class MultilayerPerceptron(BaseModel):
    def __init__(self, train_file=None,eval_file=None, model_path=None,save_dir='models/multilayer_perceptron',sample_num=2000, seed=42):
        super().__init__(train_file=train_file,eval_file=eval_file, model_path=model_path,sample_num=sample_num, seed=seed,save_dir=save_dir)
        base_model = MLPClassifier(max_iter=10000,random_state=self.seed,
                                   early_stopping=True,batch_size=1)
        self.model = MultiOutputClassifier(base_model)
        self.param_grid = {
            'estimator__hidden_layer_sizes': [(64,64),(128,128),(256,256)],
            'estimator__activation': ['tanh', 'relu','logistic'],
            'estimator__n_iter_no_change' : [10, 50, 100],
            'estimator__learning_rate_init': [1e-4, 1e-3, 1e-2],
        }