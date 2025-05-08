from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from .data import load_data, prepare_data
from .eval import cal_kl_divergence, cal_group_kl_divergence,cal_topk_acc
import os
import joblib
import pandas as pd

class BaseModel:
    def __init__(self, train_file=None,eval_file=None, model_path=None,save_dir=None,sample_num=1000, seed=42):
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

        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

        if train_file and os.path.exists(train_file):
            self.train_df = load_data(train_file)
            self.train_df = self.train_df.sample(self.sample_num,random_state=self.seed)
            # # Ensure better representation by stratifying the sample
            # if sample_num < len(self.train_df):
            #     # Sample with stratification to maintain class distribution
            #     # Use both target variables for stratification
            #     train_df_with_targets = self.train_df.copy()
            #     # Create a combined strata column
            #     train_df_with_targets['strata'] = (
            #         train_df_with_targets['primary_mode'].astype(str) + '_' + 
            #         train_df_with_targets['duration_minutes'].astype(str)
            #     )
            #     self.train_df = train_df_with_targets.groupby('strata', group_keys=False).apply(
            #         lambda x: x.sample(min(len(x), max(1, int(sample_num * len(x) / len(train_df_with_targets)))), 
            #                         random_state=seed)
            #     )
            #     # If we didn't get enough samples, add more randomly
            #     if len(self.train_df) < sample_num:
            #         remaining = sample_num - len(self.train_df)
            #         excluded = train_df_with_targets[~train_df_with_targets.index.isin(self.train_df.index)]
            #         if len(excluded) > 0:
            #             additional = excluded.sample(min(len(excluded), remaining), random_state=seed)
            #             self.train_df = pd.concat([self.train_df, additional])
            self.X_train, self.y_train, self.encoder = prepare_data(self.train_df)

        if eval_file and os.path.exists(eval_file):
            self.eval_df = load_data(eval_file)
            self.X_eval, self.y_eval, self.encoder = prepare_data(self.eval_df)
    
    def optimize(self):
        """
        Optimize the random forest model using GridSearchCV with a specified
        scoring function for multi-output classification.

        Returns:
            Best parameters found by GridSearchCV
        """
        # Define cross-validation strategy
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        
        kl_scorer = make_scorer(cal_kl_divergence,greater_is_better=False)
        
        # Set up grid search with the defined parameter grid
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=kl_scorer,
            cv=cv,
            n_jobs=-1,
            error_score='raise'  # This will raise the actual error instead of returning nan
        )
        # Fit grid search to data
        print("========Optimizing parameters========")
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train(self, optimize_first=True):
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
            self.optimize()
        
        # Train the model
        print(f"=======Training model (num_samples={self.sample_num})=======")
        self.model.fit(self.X_train, self.y_train)
        self.save_model()
        return
    
    def evaluate(self, X_eval=None, y_eval=None,group_features=None,k=3):
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
        print(f"=======Evaluating model=======")
        # print(f"Top {k} accuracy: { topk_accuracies['average']:.4f}")
        print(f"Overall average KL divergence: {overall_kl:.4f}")
        print(f"Overall mean absolute error: {overall_mae:.4f}")
        return kl_df,overall_kl,overall_mae,
    
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
        print(f"Model saved to {model_path}")
        return model_path
    
    def predict(self, X_eval):
        """Predict both targets simultaneously."""
        return self.model.predict(X_eval)
    
    def predict_proba(self, X_eval):
        """Predict probabilities for both targets."""
        return self.model.predict_proba(X_eval)