import os
import csv
import re
import torch
import torchaudio
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import pandas as pd
TARGET_SR = 16000  # target sampling rate


speech_dir = "/hdd/drawspeech/log/exp/lddpm_esd_basic/mdit_librittsesd_cutdurspn_pe/random"
#speech_dir = "/hdd/drawspeech/log/exp/lddpm_esd_basic/styletts2/random"

output_csv = "/hdd/drawspeech/log/exp/lddpm_esd_basic/mdit_librittsesd_cutdurspn_pe/emotion_predictions.csv"
#output_csv = "/hdd/drawspeech/log/exp/lddpm_esd_basic/mdit_librittsesd_cutdurspn_pe/emotion_predictions_styleTTS2.csv"

# ---- CONFIG ----
EMOTION_CLASSES = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]

# ---- Mapping from abbreviated model tags to full names ----
EMOTION_MAP = {
    "ang": "Angry",
    "hap": "Happy",
    "sad": "Sad",
    "neu": "Neutral",
    "sur": "Surprise",
    "fear": "Fear",
    "dis": "Disgust",
    "fru": "Frustrated",
    "exc": "Happy",   # sometimes "excited" should map to Happy
}

# ---- Load pretrained model ----
model_name = "superb/wav2vec2-base-superb-er"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)
model.eval()

# ---- Predict emotion ----
def predict_emotion(filepath):
    # Load and resample if needed
    speech, sr = torchaudio.load(filepath)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        speech = resampler(speech)
        sr = TARGET_SR

    inputs = extractor(speech.squeeze(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        label = model.config.id2label[pred_id]
    # Normalize to full name if abbreviated
    label = label.lower()
    label = EMOTION_MAP.get(label, label.capitalize())
    return label

# ---- Process all files ----
records = []
for fname in os.listdir(speech_dir):
    if not fname.endswith(".wav"):
        continue
    fpath = os.path.join(speech_dir, fname)

    # Extract ground truth emotion (between "spkXXXX_" and "_ref")
    match = re.search(r"spk\d+_([A-Za-z]+)_ref\d+", fname)
    if not match:
        print(f"Cannot parse emotion from {fname}")
        continue
    gt_emotion = match.group(1)

    try:
        pred_emotion = predict_emotion(fpath)
    except Exception as e:
        print(f"Prediction failed for {fname}: {e}")
        pred_emotion = "Unknown"

    records.append((fname, gt_emotion, pred_emotion))

# ---- Save to CSV ----
df = pd.DataFrame(records, columns=["filename", "ground_truth_emotion", "predicted_emotion"])
df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")

# ---- Compute metrics ----
df = df[df["predicted_emotion"] != "Unknown"]

y_true = df["ground_truth_emotion"]
y_pred = df["predicted_emotion"]

report = classification_report(y_true, y_pred, labels=EMOTION_CLASSES, zero_division=0)
macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\n===== Classification Report =====")
print(report)
print(f"Macro Precision: {macro_precision:.3f}")
print(f"Macro Recall:    {macro_recall:.3f}")
print(f"Macro F1-score:  {macro_f1:.3f}")
