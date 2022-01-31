"""
Test Metaflow Flow integration
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from wandb.integration.metaflow import wandb_log
import wandb
import numpy as np

from metaflow import FlowSpec, Parameter, step

os.environ["WANDB_SILENT"] = "true"
os.environ["METAFLOW_USER"] = "test_user"


@wandb_log(datasets=True, models=True, others=True)
class WandbExampleFlowDecoClass(FlowSpec):
    # Not obvious how to support metaflow.IncludeFile
    seed = Parameter("seed", default=1337)
    test_size = Parameter("test_size", default=0.2)
    raw_data = Parameter(
        "raw_data",
        default="https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv",
        help="path to the raw data",
    )

    @classmethod
    def get_tune_data(cls, search_clf):
        """Extract results of hyperparameter tuning as a dataframe."""
        tune_data = search_clf.cv_results_
        tune_viz = pd.DataFrame(tune_data['params'])
        tune_viz['neg_log_loss'] = tune_data['mean_test_score']
        return tune_viz

    @step
    def start(self):
        self.wandb_runid = wandb.run.id
        self.wandb_url = wandb.run.get_url()
        print(f'{self.wandb_runid=}')
        print(f'{self.wandb_url=}')
        self.raw_df = pd.read_csv(self.raw_data)
        self.next(self.split_data)

    @step
    def split_data(self):
        X = self.raw_df.drop("Wine", axis=1)
        y = self.raw_df[["Wine"]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )
        self.next(self.train)

    @step
    def train(self):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.extend([3,4,5])
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

        rf = RandomForestClassifier(random_state=self.seed)
        rf_search = RandomizedSearchCV(estimator=rf, 
                                       scoring = "neg_log_loss",
                                       param_distributions=random_grid, 
                                       n_iter=10, 
                                       cv=3, 
                                       random_state=42, 
                                       n_jobs=-1)

        self.cv_clf = rf_search.fit(self.X_train, self.y_train.values.ravel())
        self.clf = self.cv_clf.best_estimator_
        self.best_params_idx = self.cv_clf.best_index_
        # self.tune_data = self.get_tune_data(self.cv_clf)
        self.next(self.end)

    @step
    def end(self):
        self.preds = self.clf.predict(self.X_test)
        self.probas = self.clf.predict_proba(self.X_test)
        wandb.sklearn.plot_classifier(self.clf, 
                                      self.X_train, 
                                      self.X_test, 
                                      self.y_train, 
                                      self.y_test, 
                                      self.preds, 
                                      self.probas, 
                                      labels=[["1", "2"]],
                                      feature_names=self.X_train.columns.to_list(),
                                      log_learning_curve=True)
        
        self.accuracy = accuracy_score(self.y_test, self.preds)

    

if __name__ == "__main__":
    WandbExampleFlowDecoClass()