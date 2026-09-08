from datasets import load_dataset

print("Loading Hindi dataset...")

dataset = load_dataset(
    "ekacare/vistaar_small_asr_eval",
    "hi",
    split="test"
)

print("Dataset loaded.")
print("Samples:", len(dataset))
print("Columns:", dataset.column_names)

sample = dataset[0]

print("\nTranscript:")
print(sample["text"])

print("\nAudio information:")
print(sample["audio"])

print("\nSample rate:")
print(sample["audio"]["sampling_rate"])

print("\nNumber of audio samples:")
print(len(sample["audio"]["array"]))