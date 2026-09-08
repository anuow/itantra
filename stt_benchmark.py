import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel
from jiwer import wer


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

LANGUAGES = ["hi", "bn", "ta"]
DATA_DIR = Path("data")

DEVICE = "cpu"


def load_audio(path):
    wav, sr = torchaudio.load(str(path))

    # Convert stereo -> mono if necessary
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # IndicConformer expects 16 kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=16000,
        )
        wav = resampler(wav)

    return wav


def benchmark_language(model, language):
    data_dir = DATA_DIR / language
    wav_files = sorted(data_dir.glob("*.wav"))

    if not wav_files:
        print(f"\nNo WAV files found for {language} in {data_dir}")
        return None

    print("\n" + "-" * 60)
    print(f"LANGUAGE: {language.upper()}")
    print("-" * 60)
    print(f"Found {len(wav_files)} WAV files.")

    references = []
    hypotheses = []

    total_audio_seconds = 0.0
    total_inference_seconds = 0.0

    for index, wav_path in enumerate(wav_files, start=1):

        txt_path = wav_path.with_suffix(".txt")

        if not txt_path.exists():
            print(f"WARNING: Missing transcript: {txt_path}")
            continue

        reference = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        wav = load_audio(wav_path)

        audio_seconds = wav.shape[-1] / 16000
        total_audio_seconds += audio_seconds

        print(
            f"[{index}/{len(wav_files)}] "
            f"{wav_path.name} | "
            f"{audio_seconds:.2f}s",
            end=" "
        )

        start = time.perf_counter()

        with torch.no_grad():
            hypothesis = model(
                wav,
                language,
                "ctc",
            )

        elapsed = time.perf_counter() - start
        total_inference_seconds += elapsed

        print(f"| inference {elapsed:.2f}s")

        references.append(reference)
        hypotheses.append(hypothesis)

    if not references:
        print(f"No valid samples found for {language}.")
        return None

    score = wer(
        references,
        hypotheses,
    )

    avg_latency = (
        total_inference_seconds / len(references)
    )

    rtf = (
        total_inference_seconds / total_audio_seconds
        if total_audio_seconds > 0
        else 0
    )

    results = {
        "language": language,
        "samples": len(references),
        "audio": total_audio_seconds,
        "inference": total_inference_seconds,
        "latency": avg_latency,
        "rtf": rtf,
        "wer": score,
    }

    print("\nResults:")
    print(f"Samples:               {results['samples']}")
    print(f"Total audio:           {results['audio']:.2f}s")
    print(f"Total inference:       {results['inference']:.2f}s")
    print(f"Average latency:       {results['latency']:.2f}s")
    print(f"Real-time factor:      {results['rtf']:.3f}")
    print(f"WER:                   {results['wer'] * 100:.2f}%")

    return results


def main():
    print("=" * 60)
    print("IndicConformer STT Multilingual Benchmark")
    print("=" * 60)

    print(f"\nLoading model: {MODEL_ID}")
    print(f"Device: {DEVICE}")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device=DEVICE,
    )

    print("Model loaded.")

    all_results = []

    for language in LANGUAGES:
        result = benchmark_language(
            model,
            language,
        )

        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(
        f"{'Language':<12}"
        f"{'Samples':<10}"
        f"{'Latency':<12}"
        f"{'RTF':<10}"
        f"{'WER':<10}"
    )

    print("-" * 60)

    for result in all_results:
        print(
            f"{result['language']:<12}"
            f"{result['samples']:<10}"
            f"{result['latency']:.2f}s"
            f"{'':<8}"
            f"{result['rtf']:.3f}"
            f"{'':<6}"
            f"{result['wer'] * 100:.2f}%"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()