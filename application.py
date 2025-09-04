from flask import Flask, request, render_template
import os
import tempfile
import pandas as pd

from src.pipelines.predict_pipeline import PredictPipeline, CustomData
from src.utils import build_features_for_inference
from src.exceptions import logging, CustomException

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    # POST: either file-upload (batch) or manual single-row entry
    try:
        pipe = PredictPipeline()

        # ----- File upload branch (batch) -----
        file = request.files.get("data_file")
        if file and file.filename:
            suffix = ".parquet" if file.filename.lower().endswith(".parquet") else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            # Build features for ALL rows, then predict
            X = build_features_for_inference(tmp_path)
            preds = pipe.predict(X)  # vectorized predictions
            preds = list(preds)

            # Optional: summary + tiny preview
            counts = pd.Series(preds).value_counts().to_dict()
            preview = X.head(5).copy()
            preview["prediction"] = preds[: min(5, len(preds))]
            preview_records = preview.to_dict(orient="records")

            os.remove(tmp_path)

            return render_template(
                "home.html",
                predictions=preds,
                batch_counts=counts,
                preview=preview_records,
            )

        # ----- Manual single-row branch -----
        one = CustomData(
            text_size=int(request.form["text_size"]),
            comment_count=int(request.form["comment_count"]),
            participants_count=int(request.form["participants_count"]),
            first_response_minutes=float(request.form["first_response_minutes"]),
            first_response_missing=int(request.form["first_response_missing"]),
        ).get_data_as_data_frame()

        pred = pipe.predict(one)  # returns array-like
        return render_template("home.html", predictions=list(pred))

    except Exception as e:
        logging.exception("Prediction failed.")
        return render_template("home.html", error=str(e)), 500


if __name__ == "__main__":
    # Flask dev server (don’t use in production)
    app.run(host="0.0.0.0", port=5050, debug=True)
