import os
import re
import joblib
from flask import Flask, render_template, request
from langdetect import detect, LangDetectException

app = Flask(__name__, static_folder="static", template_folder="templates")

# === Model paths ===
MODEL_PATHS = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
}

# === Load models ===
MODELS = {}
for name, path in MODEL_PATHS.items():
    try:
        MODELS[name] = joblib.load(path)
        print(f"[OK] {name} loaded successfully.")
    except Exception as e:
        print(f"[ERROR] {name} not loaded: {e}")
        MODELS[name] = None

# === Vectorizer ===
VECTORIZER = None
if os.path.exists("tfidf_vectorizer.pkl"):
    VECTORIZER = joblib.load("tfidf_vectorizer.pkl")
    print("[OK] Vectorizer loaded.")
else:
    print("[WARN] No tfidf_vectorizer.pkl found.")

# === Reference scores ===
MODEL_SCORES = {
    "Logistic Regression": 98.5652,
    "Decision Tree": 99.5260,
    "Random Forest": 98.1040,
    "Gradient Boosting": 99.5388,
}


def clean_text(text):
    text = text.strip()
    text = re.sub(r"[^\w\s\.,\?\!\-:؛،']", " ", text)
    return text


def ml_predict(text, model_name):
    model = MODELS.get(model_name)
    if model is None:
        return 50.0, "Model Not Loaded", 0.5

    try:
        input_data = [clean_text(text)]
        # Try raw text prediction first (for pipelines)
        try:
            probs = model.predict_proba(input_data)
        except Exception:
            if VECTORIZER is not None:
                X = VECTORIZER.transform(input_data)
                probs = model.predict_proba(X)
            else:
                raise ValueError("Vectorizer missing for non-pipeline model.")

        fake_prob = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])
        legit_prob = 1 - fake_prob
        ml_conf = max(fake_prob, legit_prob) * 100
        label = "Likely Fake News" if fake_prob >= 0.5 else "Appears Legitimate"
        return ml_conf, label, fake_prob
    except Exception as e:
        print(f"[ERROR] Prediction failed for {model_name}: {e}")
        return 50.0, "Model Error", 0.5


def ai_explain(text, language):
    txt = text.strip()
    lower = txt.lower()
    url_count = len(re.findall(r"https?://", txt))
    exclamations = txt.count("!")
    uppercase_words = sum(1 for w in txt.split() if w.isupper() and len(w) > 1)
    suspicious_phrases = ["breaking", "exclusive", "viral", "shocking", "alert"]
    sp_count = sum(1 for p in suspicious_phrases if p in lower)

    score = url_count * 8 + exclamations * 3 + uppercase_words * 2 + sp_count * 6
    ai_conf = max(5, min(98, int(score)))
    explanation = "Text shows signs of possible exaggeration or misleading content." if ai_conf > 60 else "Text seems neutral."
    return ai_conf, explanation


@app.route("/")
def index():
    return render_template("index.html", model_names=list(MODEL_PATHS.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text", "").strip()
    model_choice = request.form.get("model_choice", "Gradient Boosting")
    lang_choice = request.form.get("language", "auto")

    if not text:
        return render_template("result.html", error="Please enter text to analyze.")

    detected_lang = lang_choice
    if lang_choice == "auto":
        try:
            detected_lang = detect(text)
        except LangDetectException:
            detected_lang = "unknown"

    ml_conf, ml_label, fake_prob = ml_predict(text, model_choice)
    ai_conf, explanation = ai_explain(text, detected_lang)

    final_label = ml_label
    if ml_label == "Likely Fake News" and ai_conf >= 60 and ml_conf >= 55:
        final_label = "Likely Fake News"
    elif ml_label == "Appears Legitimate" and ai_conf < 60:
        final_label = "Appears Legitimate"
    else:
        final_label = "Needs Further Review"

    return render_template(
        "result.html",
        text=text,
        model_choice=model_choice,
        detected_lang=detected_lang,
        ml_conf=round(ml_conf, 2),
        ai_conf=round(ai_conf, 2),
        fake_prob=round(fake_prob * 100, 2),
        explanation=explanation,
        final_label=final_label
    )


if __name__ == "__main__":
    app.run(debug=True)
