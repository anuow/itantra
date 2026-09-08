from datasets import load_dataset
from pathlib import Path
import soundfile as sf


LANGUAGES = {
    "hi": "hi",
    "bn": "bn",
    "ta": "ta",
}

SAMPLES_PER_LANGUAGE = 30

OUTPUT_DIR = Path("data")


def download_language(code, dataset_language):
    print(f"\nLoading {dataset_language}...")

    dataset = load_dataset(
        "ekacare/vistaar_small_asr_eval",
        dataset_language,
        split="test"
    )

    print(f"Available samples: {len(dataset)}")
    print("Columns:", dataset.column_names)

    output_dir = OUTPUT_DIR / code
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(SAMPLES_PER_LANGUAGE, len(dataset))

    for i in range(count):
        sample = dataset[i]

        audio = sample["audio"]
        transcript = sample["text"]

        wav_path = output_dir / f"utt{i + 1:03d}.wav"
        txt_path = output_dir / f"utt{i + 1:03d}.txt"

        sf.write(
            str(wav_path),
            audio["array"],
            audio["sampling_rate"]
        )

        txt_path.write_text(
            transcript.strip(),
            encoding="utf-8"
        )

        print(f"Saved {wav_path}")

    print(f"Finished {dataset_language}")


def main():
    for code, language in LANGUAGES.items():
        download_language(code, language)

    print("\nAll audio downloaded.")


if __name__ == "__main__":
    main()