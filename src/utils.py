import os
import sys
import re
import numpy as np 
import pandas as pd
import dill
import pickle
from src.exceptions import CustomException
from src.logger import logging
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support
)
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


#---------------------------- CLEANING UTILS-------------------------------------------------


# --- helpers ---
TITLE_RE   = re.compile(r"(?i)\btitle:\s*(.+?)\s*(?=username_\d+:|$)")
SPEAKER_RE = re.compile(r"\busername_\d+:\s*")

X_COLS = ["text_size","comment_count","first_response_minutes"]


def clean_content(text: str) -> str:
    try:
        if not isinstance(text, str):
            return ""
        # remove special tokens like <issue_start>, <issue_comment>, <issue_closed>
        text = re.sub(r"<[^>]+>", " ", text)
        # remove usernames like @username_123
        text = re.sub(r"@\w+", " ", text)
        # remove issue/bug IDs like #TRAC123 or #1234
        text = re.sub(r"#\w+", " ", text)
        # collapse multiple spaces
        return re.sub(r"\s+", " ", text).strip()
    
    except Exception as e:
        raise CustomException(e, sys)

def split_title_body(text: str):

    try:
        if not isinstance(text, str) or not text.strip():
            return "", ""
        m = TITLE_RE.search(text)
        title = m.group(1).strip() if m else ""
        body  = TITLE_RE.sub(" ", text) if m else text
        body  = SPEAKER_RE.sub(" ", body)
        body  = re.sub(r"\s+", " ", body).strip()
        return title, body
    
    except Exception as e:
        raise CustomException(e, sys)

def parse_events_min(events):
    """
    Parse issue events into modeling features:
      - opened_at, closed_at, is_closed
      - lifecycle_hours
      - comment_count
      - first_response_minutes
      - participants_count
    """
    opened_at = None
    closed_at = None
    first_comment_at = None
    first_responder = None
    comment_count = 0
    participants = set()

    try:
        # Accept list/tuple/np.ndarray of event dicts
        if isinstance(events, np.ndarray):
            events = events.tolist()
        if not isinstance(events, (list, tuple)):
            return {
                "opened_at": None,
                "closed_at": None,
                "is_closed": False,
                "lifecycle_hours": None,
                "comment_count": 0,
                "first_response_minutes": None,
                "participants_count": 0,
            }

        for ev in events:
            ev_type = (ev.get("type") or "").lower()       # 'issue' | 'comment' | ...
            action  = (ev.get("action") or "").lower()     # 'opened' | 'closed' | 'created' ...
            author  = ev.get("author")
            dt      = pd.to_datetime(ev.get("datetime"), utc=True, errors="coerce")

            if author:
                participants.add(author)

            if ev_type == "issue" and action == "opened" and opened_at is None:
                opened_at = dt
            elif ev_type == "issue" and action == "closed" and closed_at is None:
                closed_at = dt
            elif ev_type == "comment":
                comment_count += 1
                if first_comment_at is None:
                    first_comment_at = dt
                    first_responder = author

        is_closed = closed_at is not None

        lifecycle_hours = (
            (closed_at - opened_at).total_seconds() / 3600.0
            if opened_at is not None and closed_at is not None
            else None
        )

        first_response_minutes = (
            (first_comment_at - opened_at).total_seconds() / 60.0
            if opened_at is not None and first_comment_at is not None
            else None
        )

        return {
            "opened_at": opened_at,
            "closed_at": closed_at,
            "is_closed": is_closed,
            "lifecycle_hours": lifecycle_hours,
            "comment_count": int(comment_count),
            "first_response_minutes": first_response_minutes,
            "participants_count": int(len(participants)),
        }
    except Exception as e:
        raise CustomException(e, sys)

# --- main ------------------------------------------------------------------------------------------

def build_model_table_from_parquet(parquet_path: str, out_csv: str | None = None) -> pd.DataFrame:


    """
    Read a parquet file -> filter & engineer features -> (optionally) save CSV.
    Steps:
      1) Keep closed issues with non-null lifecycle_hours
      2) first_response_missing flag + impute -1 for first_response_minutes
      3) Clean content; extract issue_title + issue_body
      4) Bin lifecycle_hours into 3 classes: Short (<=24h), Medium (<=168h), Long (>168h)
      5) Drop rows where resolution_class is NaN
    """

    try:
        # 0) Read
        df = pd.read_parquet(parquet_path)

        # 1) Drop duplicates issue_ids and all pull requests
        df = df.dropna(subset=["issue_id"])
        df = df[df["pull_request"].isna()].reset_index(drop=True)
        df.drop(columns='pull_request',inplace=True)

        # 2) Parse the events column  
        fe = df["events"].apply(parse_events_min).apply(pd.Series)
        df = pd.concat([df.drop(columns=["opened_at","closed_at","is_closed",
                                        "lifecycle_hours","comment_count",
                                        "first_response_minutes","participants_count"], errors="ignore"),
                        fe], axis=1)

        # 3) Filter to valid, closed issues (guard against negatives just in case)
        df_model = df[(df["is_closed"] == True) & (df["lifecycle_hours"].notna())].copy()
        if "lifecycle_hours" in df_model.columns:
            df_model = df_model[df_model["lifecycle_hours"] >= 0].copy()

        # 4) First response flags/imputation
        df_model["first_response_missing"] = df_model["first_response_minutes"].isna().astype(int)
        df_model["first_response_minutes"] = df_model["first_response_minutes"].fillna(-1)

        # 5) Clean content + split title/body
        df_model["content_clean"] = df_model["content"].apply(clean_content)
        titles_bodies = df_model["content_clean"].apply(split_title_body)
        df_model["issue_title"] = titles_bodies.apply(lambda x: x[0])
        df_model["issue_body"]  = titles_bodies.apply(lambda x: x[1])

        # 6) 3-class resolution bins
        bins   = [0, 24, 168, float("inf")]
        labels = ["Short", "Medium", "Long"]
        df_model["resolution_class"] = pd.cut(
            df_model["lifecycle_hours"],
            bins=bins, labels=labels, right=True, include_lowest=True
        )

        # 7) Drop NaN classes
        df_model = df_model.dropna(subset=["resolution_class"]).copy()

        # # Optional: write to CSV
        # if out_csv:
        #     df_model.to_csv(out_csv, index=False)

        return df_model

    except Exception as e:
        raise CustomException(e, sys)
      
def build_features_for_inference(data) -> pd.DataFrame:
    try:
        # normalize input -> DataFrame
        if isinstance(data, str):
            df = pd.read_parquet(data) if data.lower().endswith(".parquet") else (
                 pd.read_csv(data) if data.lower().endswith(".csv") else
                 (_ for _ in ()).throw(ValueError("Use .csv or .parquet")))
        elif isinstance(data, pd.Series): df = data.to_frame().T
        elif isinstance(data, dict):      df = pd.DataFrame([data])
        else:                             df = data.copy()

        # fast path: already have the 5 features
        if all(c in df.columns for c in X_COLS):
            out = df[X_COLS].copy()
        else:
            # drop PRs if present
            if "pull_request" in df.columns:
                df = df[df["pull_request"].isna()].drop(columns=["pull_request"], errors="ignore")

            # parse events if available
            if "events" in df.columns:
                fe = df["events"].apply(parse_events_min).apply(pd.Series)
                df = pd.concat([df, fe], axis=1)

            # first response flags + impute
            frm = df["first_response_minutes"] if "first_response_minutes" in df.columns else pd.Series([-1]*len(df), index=df.index)
            df["first_response_missing"]  = frm.isna().astype(int)
            df["first_response_minutes"]  = frm.fillna(-1)

            # text_size from content or empty series
            if "text_size" not in df.columns:
                content = df["content"].astype(str) if "content" in df.columns else pd.Series([""]*len(df), index=df.index)
                df["text_size"] = content.apply(lambda x: len(clean_content(x)))

            # ensure numeric cols exist
            for col, default in [("comment_count", 0), ("participants_count", 0)]:
                if col not in df.columns: df[col] = default

            out = df[X_COLS].copy()

        # final dtypes
        out["text_size"]              = out["text_size"].astype(int)
        out["comment_count"]          = out["comment_count"].astype(int)
        out["participants_count"]     = out["participants_count"].astype(int)
        out["first_response_minutes"] = out["first_response_minutes"].astype(float)
        out["first_response_missing"] = out["first_response_missing"].astype(int)
        return out

    except Exception as e:
        raise CustomException(e, sys)

    
#-----------------------------OTHER UTILS-----------------------------------------------------
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def eval_metrics(y_true, y_pred, le):

    try:
        acc   = accuracy_score(y_true, y_pred)
        f1_ma = f1_score(y_true, y_pred, average="macro")
        f1_wt = f1_score(y_true, y_pred, average="weighted")
        prec_ma = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_ma  = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # per-class metrics
        classes = list(le.classes_)  # e.g., ["Short","Medium","Long"]
        p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
        )
        # Find indices for "Long" and "Short" if present
        idx_long  = classes.index("Long")  if "Long"  in classes else None
        idx_short = classes.index("Short") if "Short" in classes else None
        recall_long     = r_cls[idx_long]  if idx_long  is not None else np.nan
        precision_short = p_cls[idx_short] if idx_short is not None else np.nan

        return {
            "acc": acc, "f1_macro": f1_ma, "f1_weighted": f1_wt,
            "precision_macro": prec_ma, "recall_macro": rec_ma,
            "recall_long": recall_long, "precision_short": precision_short
        }

    except Exception as e:
        raise CustomException(e, sys)




def evaluate_models(
    X_train, y_train, X_test, y_test,
    models: Dict[str, Any],
    params: Dict[str, Dict[str, list]] = None,
    le=None,
    use_search: bool = True,            # set True to run Grid/Randomized search
    search_type: str = "grid",           # "grid" or "random"
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
    scoring: str = "f1_macro",           # used only if use_search=True
    refit: bool = True,                  # used only if use_search=True
    n_iter: int = 30                     # used for RandomizedSearchCV
) -> Tuple[Dict[str, dict], Dict[str, Any], str, Any]:
    """
    Returns:
        results: dict[name] -> test metrics (from eval_metrics)
        fitted_models: dict[name] -> fitted estimator
        best_name: name of best model by F1-macro (on test set)
        best_model: fitted estimator for best_name
    """
    try:
        results = {}
        fitted_models = {}

        for name, base_model in models.items():
            model = base_model

            # Optional hyperparameter search
            if use_search and params is not None and name in params and params[name]:
                search_space = params[name]
                if search_type.lower() == "grid":
                    search = GridSearchCV(
                        estimator=model,
                        param_grid=search_space,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        scoring=scoring,
                        refit=refit
                    )
                elif search_type.lower() == "random":
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=search_space,
                        n_iter=n_iter,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        scoring=scoring,
                        refit=refit,
                        random_state=42
                    )
                else:
                    raise ValueError("search_type must be 'grid' or 'random'.")

                search.fit(X_train, y_train)
                # Use the tuned estimator if refit=True, otherwise set best params and fit
                model = search.best_estimator_ if refit else model.set_params(**search.best_params_)

            # Fit final model
            model.fit(X_train, y_train)

            # Predictions
            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)

            # Metrics (your eval_metrics uses LabelEncoder `le`)
            tr = eval_metrics(y_train, y_tr_pred, le)
            te = eval_metrics(y_test,  y_te_pred,  le)

            # Store
            results[name] = te
            fitted_models[name] = model

            # Print like your notebook
            # print(name)
            # print(
            #     f"Train | Acc: {tr['acc']:.3f}  F1-macro: {tr['f1_macro']:.3f}  "
            #     f"Prec-macro: {tr['precision_macro']:.3f}  Rec-macro: {tr['recall_macro']:.3f}  "
            #     f"Recall(Long): {tr['recall_long'] if np.isnan(tr['recall_long'])==False else np.nan:.3f}  "
            #     f"Precision(Short): {tr['precision_short'] if np.isnan(tr['precision_short'])==False else np.nan:.3f}"
            # )
            # print(
            #     f"Test  | Acc: {te['acc']:.3f}  F1-macro: {te['f1_macro']:.3f}  "
            #     f"Prec-macro: {te['precision_macro']:.3f}  Rec-macro: {te['recall_macro']:.3f}  "
            #     f"Recall(Long): {te['recall_long'] if np.isnan(te['recall_long'])==False else np.nan:.3f}  "
            #     f"Precision(Short): {te['precision_short'] if np.isnan(te['precision_short'])==False else np.nan:.3f}"
            # )
            # print("-" * 70)

        # Pick best by F1-macro (on test set)
        best_name = max(results, key=lambda k: results[k]["f1_macro"])
        best_model = fitted_models[best_name]
        logging.info(f"\nBest model by F1-macro: {best_name} -> {results[best_name]}")

        return results, fitted_models, best_name, best_model

    except Exception as e:
        # Keep your CustomException handling if you have it defined elsewhere
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
