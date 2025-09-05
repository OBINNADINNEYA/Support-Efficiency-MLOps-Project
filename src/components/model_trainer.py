import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from src.exceptions import CustomException
from src.logger import logging
from src.utils import *
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import shap


@dataclass
class ModelTrainerConfig:

    #model pickle file path assignment
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:

    def __init__(self):
        self.Model_Trainer_config = ModelTrainerConfig()


    def initiate_model_training(self,train_array,test_array,preprocessor_path,encoder_path, enable_shap = False):

        try:
            logging.info("splitting training and test input data")

            X_train,X_test,y_train,y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1],
            )

            print(train_array[:5, -1])
            print(test_array[:5, -1])
            
            models = {
                "LogisticRegression": LogisticRegression(),
                "LinearSVC": LinearSVC(),
                "KNN": KNeighborsClassifier(),
                "RandomForest": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
                "LightGBM": LGBMClassifier(),
                "MLPClassifier": MLPClassifier(max_iter=500)
            }

            params = {
                "LogisticRegression": {
                    "penalty": ["l2"],
                    "C": [0.1, 1, 10],
                    "max_iter": [1000],
                },
                "LinearSVC": {
                    "penalty": ["l2"],
                    "C": [0.1, 1, 10],
                    "max_iter": [2000],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "metric": ["minkowski"],
                    "p": [1, 2]
                },
                "RandomForest": {
                    "n_estimators": [100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                },
                "CatBoost": {
                    "depth": [6,8],
                    "learning_rate": [None,0.1],
                    "iterations": [1000],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.5, 1.0],
                },

                "LightGBM": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [200, 400],
                    "num_leaves": [31, 63],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
                
            }



            #load encoder file 
            le = load_object(encoder_path)

            # Evaluation (optionally set use_search=True to enable Grid/Randomized search)
            results, fitted_models, best_model_name, best_model = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models,
                params=params,        # pass your params dict
                le=le,                # your LabelEncoder (for per-class metrics)
                use_search=True,     # set True if you want to tune
                search_type="grid",   # "grid" or "random"
                cv=3, n_jobs=-1, verbose=1,
                scoring="f1_macro",   # only used if use_search=True
                refit=True
            )

            logging.info("getting best model")
            # Best model score by f1_macro on the TEST set
            best_model_score = results[best_model_name]["f1_macro"]

            if best_model_score < 0.60:
                raise CustomException(f"No good model found (best f1_macro={best_model_score:.3f} < 0.60)")

            logging.info(f"Best model on test set: {best_model_name} (f1_macro={best_model_score:.3f})")

            # Persist the tuned/fitted best model
            save_object(
                file_path=self.Model_Trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("saving best model")

            te = results[best_model_name]

                    # ------------ SHAP (optional, non-blocking) ------------
            if enable_shap:
                

                TREE_LIKE = (
                    CatBoostClassifier, XGBClassifier, LGBMClassifier,
                    RandomForestClassifier, GradientBoostingClassifier,
                    ExtraTreesClassifier, HistGradientBoostingClassifier
                )

                if isinstance(best_model, TREE_LIKE):
                    # Pick feature names
                    if x_cols is None:
                        # fall back to generic names matching the transformed matrix width
                        x_cols = [f"f{i}" for i in range(X_test.shape[1])]

                    # sample to keep SHAP fast
                    idx = np.random.RandomState(42).choice(X_test.shape[0],
                                                        size=min(shap_max_rows, X_test.shape[0]),
                                                        replace=False)
                    X_small = X_test[idx]
                    y_small = y_test[idx]

                    X_small_df = pd.DataFrame(X_small, columns=x_cols)

                    explainer = shap.TreeExplainer(best_model)
                    sv = explainer(X_small_df)  # new API Explanation

                    # Global importance (handles binary or multiclass)
                    vals = sv.values
                    if vals.ndim == 2:
                        glob = np.abs(vals).mean(axis=0)
                    elif vals.ndim == 3:
                        glob = np.abs(vals).mean(axis=(0, 2))
                    else:
                        glob = np.abs(vals).reshape(vals.shape[1], -1).mean(axis=1)

                    # Save bar plot
                    plt.figure(figsize=(6, 3.5))
                    order = np.argsort(glob)[::-1]
                    plt.barh(np.array(x_cols)[order], glob[order])
                    plt.gca().invert_yaxis()
                    plt.title(f"{best_model_name} – SHAP Feature Importance")
                    plt.xlabel("Mean |SHAP|")
                    plt.tight_layout()
                    os.makedirs("artifacts", exist_ok=True)
                    out_path = os.path.join("artifacts", "shap_global.png")
                    plt.savefig(out_path, dpi=150)
                    plt.close()
                    logging.info(f"Saved SHAP global importance → {out_path}")
                else:
                    logging.info(f"SHAP skipped: {best_model_name} is not tree-based")
            # --------------------------------------------------------

            summary = (
                f"Best model: {best_model_name}\n"
                f"Test | Acc: {te['acc']:.3f}  F1-macro: {te['f1_macro']:.3f}  "
                f"Prec-macro: {te['precision_macro']:.3f}  Rec-macro: {te['recall_macro']:.3f}  "
                f"Recall(Long): {np.nan if np.isnan(te['recall_long']) else te['recall_long']:.3f}  "
                f"Precision(Short): {np.nan if np.isnan(te['precision_short']) else te['precision_short']:.3f}"
            )

            return summary

        except Exception as e:
            raise CustomException(e, sys)