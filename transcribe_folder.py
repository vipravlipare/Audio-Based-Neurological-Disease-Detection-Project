import whisper
from pathlib import Path

AUDIO_DIR = Path(r"Data\DementiaData\Dementia")
NO_AUDIO_DIR = Path(r"Data\DementiaData\NoDementia")

OUT_DIR = Path(r"transcripts\dementia")
NO_OUT_DIR = Path(r"transcripts\nodementia")

OUT_DIR.mkdir(parents=True, exist_ok=True)
NO_OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")  # faster than medium/large
print("Model loaded.")

def transcribe_folder(input_dir, output_dir):
    wav_files = list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.WAV"))

    print(f"\nFound {len(wav_files)} wav files in {input_dir}")

    for i, wav_path in enumerate(wav_files, start=1):
        relative_path = wav_path.relative_to(input_dir).with_suffix(".txt")
        out_file = output_dir / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            print(f"[{i}/{len(wav_files)}] Skipping existing: {out_file}")
            continue

        print(f"\n[{i}/{len(wav_files)}] Transcribing: {wav_path}")

        try:
            result = model.transcribe(
                str(wav_path),
                language="en",
                fp16=False,
                verbose=False,
                beam_size=1,
                best_of=1
            )

            transcript = result["text"].strip()
            out_file.write_text(transcript, encoding="utf-8")

            print(f"Saved: {out_file}")

        except Exception as e:
            print(f"ERROR with {wav_path}: {e}")

transcribe_folder(AUDIO_DIR, OUT_DIR)
transcribe_folder(NO_AUDIO_DIR, NO_OUT_DIR)

print("\nDone transcribing.")