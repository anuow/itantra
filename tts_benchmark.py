"""
Phase 1 — TTS baseline benchmark for iTantra.

Loads AI4Bharat's Indic Parler-TTS and, per language:
  - synthesizes a handful of test sentences (including one alert-style
    message, since that's a distinct requirement in the problem
    statement)
  - measures RTF (synthesis_time / output_audio_duration)
  - saves the .wav files so you can do human MOS / intelligibility
    scoring by ear — that part can't be automated

Run this on a machine with GPU + internet access to huggingface.co
(Colab is fine). It will NOT run inside a sandbox with no HF access.

Usage:
    python tts_benchmark.py --out_dir ./tts_samples --out results_tts.csv

Edit TEST_SENTENCES below with real sentences in each target script —
the placeholders here are English glosses only, swap in native-script
text before running.
"""
import argparse
import csv
import time
from pathlib import Path

import soundfile as sf
from tabulate import tabulate

MODEL_ID = "ai4bharat/indic-parler-tts"

# lang_code -> (normal test sentence, alert-style test sentence)
# NOTE: replace these placeholders with real native-script sentences
# for hi/gu/mr/kn/ml/ta/te/or/bn before running for real.
TEST_SENTENCES = {
    "hi": ("<replace with Hindi test sentence>", "<replace with Hindi alert sentence>"),
    "gu": ("<replace with Gujarati test sentence>", "<replace with Gujarati alert sentence>"),
    "mr": ("<replace with Marathi test sentence>", "<replace with Marathi alert sentence>"),
    "kn": ("<replace with Kannada test sentence>", "<replace with Kannada alert sentence>"),
    "ml": ("<replace with Malayalam test sentence>", "<replace with Malayalam alert sentence>"),
    "ta": ("<replace with Tamil test sentence>", "<replace with Tamil alert sentence>"),
    "te": ("<replace with Telugu test sentence>", "<replace with Telugu alert sentence>"),
    "or": ("<replace with Odia test sentence>", "<replace with Odia alert sentence>"),
    "bn": ("<replace with Bengali test sentence>", "<replace with Bengali alert sentence>"),
    "en": ("Water levels are rising near the riverbank.", "Evacuate the area immediately."),
}

VOICE_DESCRIPTION = (
    "A clear, calm adult voice speaking at a natural pace, "
    "recorded with good audio quality and minimal background noise."
)


def load_model():
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} ...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer, device


def synthesize(model, tokenizer, device, text: str, description: str):
    import torch
    desc_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        generation = model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)
    elapsed = time.perf_counter() - start

    audio = generation.cpu().numpy().squeeze()
    sample_rate = model.config.sampling_rate
    return audio, sample_rate, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="tts_samples")
    parser.add_argument("--out", default="results_tts.csv")
    args = parser.parse_args()

    model, tokenizer, device = load_model()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    rows = []
    for lang, (normal_text, alert_text) in TEST_SENTENCES.items():
        for kind, text in [("normal", normal_text), ("alert", alert_text)]:
            audio, sr, elapsed = synthesize(model, tokenizer, device, text, VOICE_DESCRIPTION)
            duration = len(audio) / sr
            wav_path = out_dir / f"{lang}_{kind}.wav"
            sf.write(str(wav_path), audio, sr)
            rows.append({
                "language": lang,
                "type": kind,
                "audio_duration_s": round(duration, 2),
                "synthesis_time_s": round(elapsed, 2),
                "RTF": round(elapsed / max(duration, 1e-6), 3),
                "file": str(wav_path),
            })
            print(f"  {lang}/{kind}: RTF={rows[-1]['RTF']}  -> {wav_path}")

    print("\n" + tabulate(rows, headers="keys", tablefmt="github"))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {args.out}")
    print("Listen to the .wav files yourself (or run a listening panel) "
          "to score intelligibility/MOS — that part isn't automated.")


if __name__ == "__main__":
    main()
