import os
import glob
import jiwer
import torch
import torchaudio
from transformers import pipeline
import pandas as pd



def load_transcripts(transcript_dir, file_type="*.lab"):
    transcripts = {}
    for path in glob.glob(os.path.join(transcript_dir, file_type)):
        key = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            transcripts[key] = f.read().strip()
    return transcripts


def transcribe_audio(audio_path, asr):
    result = asr(audio_path)
    return result["text"].strip()


# -------- Normalization Pipeline --------
class ExpandAbbreviations:
    """Expand short forms while preserving input type (str or list[str])."""
    def __init__(self):
        self.map = {
            "dr": "doctor",
            "mr": "mister",
            "mrs": "misses",
            "ms": "miss",
            "prof": "professor",
            "dept": "department",
            "etc": "et cetera",
        }

    def _expand_in_string(self, s: str) -> str:
        words = s.split()
        return " ".join(self.map.get(w, w) for w in words)

    def __call__(self, x):
        if isinstance(x, str):
            return self._expand_in_string(x)
        elif isinstance(x, list):  # list[str] (sentences)
            return [self._expand_in_string(s) for s in x]
        return x

class EnsureNonEmptyStrings:
    """Before tokenization: ensure each sentence string is non-empty."""
    def __call__(self, x):
        if isinstance(x, str):
            s = x.strip()
            return s if s else "<empty>"
        elif isinstance(x, list):  # list[str]
            out = []
            for s in x:
                s2 = (s or "").strip()
                out.append(s2 if s2 else "<empty>")
            # If somehow all vanished, keep one placeholder
            if not out:
                out = ["<empty>"]
            return out
        return x

class EnsureNonEmptyWordLists:
    """After tokenization: ensure each sentence is a non-empty list of tokens."""
    def __call__(self, x):
        # x should be List[List[str]]
        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], list)):
            out = []
            for sent in x:
                if not isinstance(sent, list) or len(sent) == 0:
                    out.append(["<empty>"])
                else:
                    out.append(sent)
            if not out:
                out = [["<empty>"]]
            return out
        # If it is a flat list[str] (single sentence), wrap & fix
        if isinstance(x, list) and all(isinstance(t, str) for t in x):
            return [x if len(x) else ["<empty>"]]
        # Fallback
        return [["<empty>"]]

# Build the normalization pipeline
normalizer = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.ExpandCommonEnglishContractions(),  # don't -> do not
    ExpandAbbreviations(),                    # dr -> doctor, etc.
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    EnsureNonEmptyStrings(),                  # strings are never empty
    EnsureNonEmptyWordLists(),                # each inner list non-empty
])


def evaluate_wer(speech_dir, output_csv="wer_results.csv"):
    # 1. Load ASR model (you can switch to a larger one for better accuracy)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   device=0 if torch.cuda.is_available() else -1)

    transcripts = load_transcripts(speech_dir, "*.lab")

    records = []
    refs, hyps = [], []

    for wav_path in glob.glob(os.path.join(speech_dir, "*.wav")):
        key = os.path.splitext(os.path.basename(wav_path))[0]
        if key not in transcripts:
            print(f"Warning: No transcript found for {key}")
            continue

        ref = transcripts[key]
        hyp = transcribe_audio(wav_path, asr)

        # Save for global evaluation
        refs.append(ref)
        hyps.append(hyp)

        # Per-utterance WER
        #m = jiwer.compute_measures(ref, hyp)
        # Per-utterance WER (normalized)
        m = jiwer.compute_measures(
            ref, hyp,
            truth_transform=normalizer,
            hypothesis_transform=normalizer
        )

        wer = m["wer"] * 100
        records.append({
            "utt_id": key,
            "reference": ref,
            "hypothesis": hyp,
            "WER(%)": wer,
            "Substitutions": m["substitutions"],
            "Deletions": m["deletions"],
            "Insertions": m["insertions"],
            "Hits": m["hits"]
        })

    # Save per-utterance results
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved per-utterance results to {output_csv}")

    # Global evaluation
    measures = jiwer.compute_measures(refs, hyps)
    wer = measures["wer"] * 100
    sub = measures["substitutions"] / (measures["substitutions"] + measures["deletions"] + measures["hits"]) * 100
    dele = measures["deletions"] / (measures["substitutions"] + measures["deletions"] + measures["hits"]) * 100
    ins = measures["insertions"] / (measures["substitutions"] + measures["deletions"] + measures["hits"]) * 100

    with open(output_csv.split(".")[0] + "_summary.txt", "w") as f:
        f.write("=== Global Results ===")
        f.write("=== Global Results ===")
        f.write(f"WER: {wer:.2f}%")
        f.write(f"Substitution: {sub:.2f}%")
        f.write(f"Deletion: {dele:.2f}%")
        f.write(f"Insertion: {ins:.2f}%")
    return wer, sub, dele, ins


if __name__ == '__main__':
    # Example usage
    if True:
        #dit_pe_libritts_dir = "/hdd/drawspeech/log/exp/lddpm_basic/lddpm_dit_pe_libritts/random"
        #evaluate_wer(dit_pe_libritts_dir, output_csv="/hdd/drawspeech/log/exp/lddpm_basic/lddpm_dit_pe_libritts/random_wer.csv")

        #ref_dir = "/hdd/drawspeech/log/exp/lddpm_basic/reference/random"
        #evaluate_wer(ref_dir, output_csv="/hdd/drawspeech/log/exp/lddpm_basic/reference/random_wer.csv")

        # styleTTS2
        styleTTS2_dir = "/hdd/StableTTS/exp/cfm_dstgl/styletts2/random"
        evaluate_wer(styleTTS2_dir, output_csv="/hdd/StableTTS/exp/cfm_dstgl/styletts2/random_wer.csv")