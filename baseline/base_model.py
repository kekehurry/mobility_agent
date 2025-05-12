from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from .data import load_data, prepare_data
from .eval import cal_kl_divergence, cal_group_kl_divergence,cal_topk_acc
import os
import joblib
import random

class BaseModel:
    def __init__(self, train_file=None,eval_file=None, model_path=None,save_dir=None,sample_num=1000, seed=None):
        # self.model = MultiOutputClassifier(RandomForestClassifier(random_state=seed))
        # self.param_grid = {
        #     'estimator__n_estimators': [10, 50, 100],
        #     'estimator__max_depth': [None, 10, 20],
        #     'estimator__min_samples_split': [2, 5, 10],
        #     'estimator__min_samples_leaf': [1, 2, 4]
        # }
        self.sample_num = sample_num
        self.save_dir = save_dir
        self.seed = seed
        self.model_name = self.__class__.__name__

        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

        if train_file and os.path.exists(train_file):
            self.train_df = load_data(train_file)
            self.train_df = self.train_df.sample(self.sample_num,random_state=self.seed)
            # self.train_df = self.train_df[:self.sample_num]
            self.X_train, self.y_train, self.encoder = prepare_data(self.train_df)

        if eval_file and os.path.exists(eval_file):
            self.eval_df = load_data(eval_file)
            self.X_eval, self.y_eval, self.encoder = prepare_data(self.eval_df,self.encoder)
    
    def optimize(self,max_iter=100,verbose=True):
        """
        Optimize the model using GridSearch

        Returns:
            Best parameters found by GridSearch
        """
        # Fit grid search to data
        best_score = float('inf')  # For minimizing KL divergence
        best_params = None
        best_estimator = None
        param_grid_list = list(ParameterGrid(self.param_grid))
        
        # Limit iterations
        if max_iter < len(param_grid_list):
            random.seed(self.seed)
            param_grid_list = random.sample(param_grid_list, max_iter)
        
        # Iterate through parameter combinations
        param_grid_list = list(ParameterGrid(self.param_grid))
        for i, params in enumerate(param_grid_list):
            # Clone the model and set parameters
            estimator = clone(self.model)
            estimator.set_params(**params)
            
            # Fit and evaluate on training data
            estimator.fit(self.X_train, self.y_train)
            # Convert probabilities to class predictions
            y_pred = estimator.predict(self.X_train)
            score = cal_kl_divergence(self.y_train, y_pred)
            
            # Update best parameters if better score found
            if score < best_score:
                best_score = score
                best_params = params
                best_estimator = estimator
                last_improvement = i

            # Early stopping if no improvement for a while
            if (i - last_improvement) > 10 and i > 100:
                break

        # Update model with best parameters
        self.model = best_estimator
        if verbose:
            print(f"========Optimizing parameters (model={self.model_name} num_samples={self.sample_num}) ========")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_score:.4f}")
        return best_params
    
    def train(self, optimize_first=True,verbose=True):
        """
        Train the random forest model for multiple outputs.
        
        Args:
            optimize_first: Whether to optimize hyperparameters before training.
                           If False, uses default or previously set parameters.
            scoring: Scoring method for optimization ('accuracy' or 'kl_divergence')
        
        Returns:
            Trained model
        """
        if optimize_first:
            self.optimize(verbose=verbose)
        
        # Train the model
        if verbose:
            print(f"=======Training model (model={self.model_name} num_samples={self.sample_num}) =======")
        self.model.fit(self.X_train, self.y_train)
        return
    
    def evaluate(self, X_eval=None, y_eval=None,group_features=None,k=3,verbose=True):
        """
        Evaluate the model on evaluation data using both top-k-accuracy and KL divergence.
        
        Args:
            X_eval: Features for evaluation (uses self.X_eval if None)
            y_eval: Target values for evaluation (uses self.y_eval if None)
            
        Returns:
            Dictionary with evaluation results
        """
        if X_eval is None:
            X_eval = self.X_eval
        if y_eval is None:
            y_eval = self.y_eval
            
        # Calculate top-k-accuracy
        # topk_accuracies = cal_topk_acc(self,X_eval, y_eval, k=k)
        kl_df, overall_kl,overall_mae = cal_group_kl_divergence(self, X_eval, y_eval)
        if verbose:
            print(f"=======Evaluating model  (model={self.model_name} num_samples={self.sample_num}) =======")
            # print(f"Top {k} accuracy: { topk_accuracies['average']:.4f}")
            print(f"Overall average KL divergence: {overall_kl:.4f}")
            print(f"Overall mean absolute error: {overall_mae:.4f}")
        return kl_df,overall_kl,overall_mae
    
    def save_model(self):
        """
        Save the trained model and encoder to disk.
        
        Args:
            save_path: Directory to save the model.
        """
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        # Save model
        model_name = self.__class__.__name__.lower()
        model_path = os.path.join(self.save_dir, f"{model_name}_{self.sample_num}.joblib")
        joblib.dump(self.model, model_path)
        # print(f"Model saved to {model_path}")
        return model_path
    
    def predict(self, X_eval):
        """Predict both targets simultaneously."""
        return self.model.predict(X_eval)
    
    def predict_proba(self, X_eval):
        """Predict probabilities for both targets."""
        return self.model.predict_proba(X_eval)