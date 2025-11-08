import re
import joblib
import numpy as np
from flask import Flask, render_template, request
from langdetect import detect, LangDetectException

app = Flask(__name__, static_folder="static", template_folder="templates")

# Available models and file paths
MODEL_PATHS = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
}

# Small helper: clean text
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s\p{L}\p{M}\.\,\?\!\-:؛،']", " ", text, flags=re.UNICODE)
    return text

# AI explanation helper
def ai_explain(text: str, language: str):
    txt = text.strip()
    lower = txt.lower()
    url_count = len(re.findall(r"https?://", txt))
    exclamations = txt.count("!")
    uppercase_words = sum(1 for w in txt.split() if w.isupper() and len(w) > 1)
    all_caps_ratio = sum(len(w) for w in txt.split() if w.isupper()) / max(1, len(txt.replace(" ", "")))
    numeric_ratio = sum(c.isdigit() for c in txt) / max(1, len(txt))
    suspicious_phrases = [
        "shocking", "you won't believe", "must see", "alert", "breaking", "claim", "rumor",
        "secret", "exposed", "exclusive", "viral", "conspiracy", "hoax", "fake"
    ]
    sp_count = sum(1 for p in suspicious_phrases if p in lower)

    score = 0
    score += min(url_count, 3) * 8
    score += min(exclamations, 5) * 3
    score += min(uppercase_words, 10) * 2
    score += min(int(all_caps_ratio * 100), 30) * 0.2
    score += sp_count * 6
    score += min(int(numeric_ratio * 100), 10) * 0.5

    if language and language.startswith("ur"):
        if re.search(r"\b(ki|hai|tha|hain|nahi|nahin)\b", lower):
            score -= 4
    if len(txt) < 40:
        score += 5

    ai_conf = max(8, min(98, int(score)))
    explanation_parts = []
    if url_count:
        explanation_parts.append(f"Contains {url_count} link(s).")
    if exclamations:
        explanation_parts.append(f"Exclamation marks ({exclamations}) suggest sensational tone.")
    if sp_count:
        explanation_parts.append("Contains sensational phrases often used in fake news.")
    if uppercase_words:
        explanation_parts.append("High use of ALL-CAPS words suggests aggressive style.")
    if not explanation_parts:
        explanation_parts.append("The text appears informational and neutral.")

    explanation = " ".join(explanation_parts)
    if ai_conf > 65:
        explanation = "Text shows signs of misleading content. " + explanation
    else:
        explanation = "Text appears more neutral. " + explanation

    return ai_conf, explanation


# ML Prediction helper
def ml_predict(text: str, model_path: str):
    try:
        model = joblib.load(model_path)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])
            classes = getattr(model, "classes_", None)
            if classes is not None:
                idx_fake = None
                for i, c in enumerate(classes):
                    if str(c).lower() in ("1", "fake", "true"):
                        idx_fake = i
                        break
                if idx_fake is None:
                    idx_fake = 1 if probs.shape[1] > 1 else 0
            else:
                idx_fake = 1 if probs.shape[1] > 1 else 0

            fake_prob = float(probs[0][idx_fake])
            legit_prob = 1.0 - fake_prob
            label = "Likely Fake News" if fake_prob >= 0.5 else "Appears Legitimate"
            ml_conf = max(fake_prob, legit_prob) * 100.0
            return ml_conf, label, fake_prob
        else:
            pred = model.predict([text])[0]
            if isinstance(pred, (int, np.integer, float, np.floating)):
                fake_prob = 0.9 if pred == 1 else 0.1
            else:
                fake_prob = 0.9 if str(pred).lower() in ("1", "fake", "true") else 0.1
            label = "Likely Fake News" if fake_prob >= 0.5 else "Appears Legitimate"
            ml_conf = fake_prob * 100.0
            return ml_conf, label, fake_prob
    except Exception as e:
        return 50.0, f"Model Error: {e}", 0.5


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_names=list(MODEL_PATHS.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text", "").strip()
    model_choice = request.form.get("model_choice", "Gradient Boosting")
    lang_choice = request.form.get("language", "auto")

    if not text:
        return render_template("result.html", error="Please provide news text.")

    detected_lang = None
    if lang_choice and lang_choice != "auto":
        detected_lang = lang_choice
    else:
        try:
            detected_lang = detect(text)
        except LangDetectException:
            detected_lang = "unknown"

    ml_conf, ml_label, fake_prob = ml_predict(text, MODEL_PATHS[model_choice])
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
        ml_conf=round(float(ml_conf), 2),
        ai_conf=round(int(ai_conf), 2),
        ml_label=ml_label,
        final_label=final_label,
        explanation=explanation,
        fake_prob=round(float(fake_prob), 3)
    )


if __name__ == "__main__":
    app.run(debug=True)
