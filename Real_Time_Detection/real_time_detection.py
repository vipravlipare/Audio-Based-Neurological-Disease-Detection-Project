from __future__ import annotations

import argparse
import json
import re
import site
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

candidate_sites = []
user_site = site.getusersitepackages()
if user_site:
    candidate_sites.append(Path(user_site))
candidate_sites.append(Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages")

for candidate in candidate_sites:
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

import joblib
import librosa
import numpy as np
import soundfile as sf
from scipy.sparse import csr_matrix, hstack
from scipy import signal
from scipy.fft import dct

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

warnings.filterwarnings("ignore", message="X does not have valid feature names*")


MODELS_DIR = ROOT / "Models"
TARGET_SR = 16000
EPS = 1e-10
GOOD_CONFIDENCE_THRESHOLD = 0.90


def confidence_bucket(confidence: float) -> tuple[str, bool]:
    if confidence >= 0.99:
        return "Elite", False
    if confidence >= 0.97:
        return "Excellent", False
    if confidence >= 0.95:
        return "Great", False
    if confidence >= 0.93:
        return "Good", False
    if confidence >= 0.90:
        return "Decent", False
    if confidence >= 0.70:
        return "Weak", False
    return "Bad", True


def load_audio(path: Path) -> np.ndarray:
    try:
        audio, sr = sf.read(path, always_2d=False)
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
    except Exception:
        audio, sr = librosa.load(path, sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio


def trim_silence(audio: np.ndarray, ratio: float = 0.02) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    idx = np.flatnonzero(np.abs(audio) >= peak * ratio)
    if idx.size == 0:
        return audio
    return audio[idx[0] : idx[-1] + 1]


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2.0]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = max(bins[i], left + 1)
        right = max(bins[i + 1], center + 1)
        filters[i - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        filters[i - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return filters


def quick_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    ]


def run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=bool)
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [mask.size]))
    lengths = ends - starts
    return lengths.astype(np.int32), mask[starts].astype(bool)


def extract_features(audio: np.ndarray) -> np.ndarray:
    audio = trim_silence(audio)
    if audio.size < TARGET_SR // 2:
        audio = np.pad(audio, (0, TARGET_SR // 2 - audio.size))

    pre = np.append(audio[:1], audio[1:] - 0.97 * audio[:-1])
    frame_len = 400
    overlap = 240
    n_fft = 512

    freqs, _, stft = signal.stft(
        pre,
        fs=TARGET_SR,
        window="hann",
        nperseg=frame_len,
        noverlap=overlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(stft) + EPS
    power = mag ** 2

    padded = np.pad(pre, (0, max(0, frame_len - pre.size % frame_len)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_len)[:: max(1, frame_len - overlap)]
    if frames.shape[0] > mag.shape[1]:
        frames = frames[: mag.shape[1]]
    elif frames.shape[0] < mag.shape[1]:
        gap = mag.shape[1] - frames.shape[0]
        frames = np.vstack([frames, np.zeros((gap, frame_len), dtype=frames.dtype)])

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + EPS)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    vad = rms >= max(float(np.percentile(rms, 35)), float(np.max(rms) * 0.12))
    lengths, flags = run_lengths(vad)
    frame_seconds = (frame_len - overlap) / TARGET_SR
    voiced = lengths[flags] * frame_seconds
    pauses = lengths[~flags] * frame_seconds

    spec_sum = np.sum(mag, axis=0) + EPS
    centroid = np.sum(freqs[:, None] * mag, axis=0) / spec_sum
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * mag, axis=0) / spec_sum)
    cumulative = np.cumsum(power, axis=0)
    total = cumulative[-1, :] + EPS
    rolloff_85 = freqs[np.argmax(cumulative >= total * 0.85, axis=0)]
    flatness = np.exp(np.mean(np.log(power), axis=0)) / (np.mean(power, axis=0) + EPS)

    fb = mel_filterbank(TARGET_SR, n_fft, n_mels=40)
    mel = fb @ power
    log_mel = np.log(mel + EPS)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:13, :]
    delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
    small_mel = signal.resample(log_mel, 24, axis=1)
    small_mel = (small_mel - np.mean(small_mel)) / (np.std(small_mel) + EPS)

    values: list[float] = []
    values.extend(
        [
            float(len(audio) / TARGET_SR),
            float(np.mean(np.abs(audio))),
            float(np.std(audio)),
            float(np.max(np.abs(audio))),
            float(np.mean(vad)),
        ]
    )
    for vec in [rms, zcr, centroid, bandwidth, rolloff_85, flatness, voiced, pauses]:
        values.extend(quick_stats(vec))
    for matrix in [mfcc, delta]:
        values.extend(np.mean(matrix, axis=1).astype(float).tolist())
        values.extend(np.std(matrix, axis=1).astype(float).tolist())
    values.extend(small_mel.astype(float).reshape(-1).tolist())
    return np.asarray(values, dtype=np.float32)


def probability_for_positive(model, features: np.ndarray) -> float:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = list(model.classes_)

            if 1 in classes:
                return float(probs[classes.index(1)])
            if True in classes:
                return float(probs[classes.index(True)])
            if "1" in classes:
                return float(probs[classes.index("1")])

            return float(probs[-1])
    except Exception as e:
        pass

    prediction = model.predict(features)[0]
    return float(prediction)


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def speaker_stats(text: str, clip_count: int = 1) -> np.ndarray:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return np.zeros(12, dtype=np.float32)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    unique_ratio = len(set(words)) / max(1, len(words))
    repeats = 1.0 - unique_ratio
    long_ratio = float(np.mean([length >= 20 for length in sentence_lengths])) if sentence_lengths else 0.0
    short_ratio = float(np.mean([length <= 6 for length in sentence_lengths])) if sentence_lengths else 0.0
    filler_words = {"um", "uh", "like", "well", "so", "you", "know"}
    filler_ratio = sum(1 for w in words if w in filler_words) / max(1, len(words))
    return np.asarray(
        [
            float(len(words)),
            float(len(sentences)),
            float(np.mean(sentence_lengths)),
            float(np.std(sentence_lengths)),
            float(unique_ratio),
            float(repeats),
            float(long_ratio),
            float(short_ratio),
            float(filler_ratio),
            float(sum(len(w) for w in words) / max(1, len(words))),
            float(clip_count),
            float(len(text)),
        ],
        dtype=np.float32,
    )


def dementia_text_audio_probability(
    wav_path: Path,
    audio_features: np.ndarray,
    transcript_text: str | None = None,
) -> float | None:
    model_path = MODELS_DIR / "dementia_audio_text_model.joblib"
    if not model_path.exists():
        return None

    package = joblib.load(model_path)
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        return None

    word_vectorizer = package["word_vectorizer"]
    char_vectorizer = package["char_vectorizer"]
    stats_scaler = package["stats_scaler"]
    audio_scaler = package["audio_scaler"]
    svd = package["svd"]
    model = package["model"]
    best_model_name = package.get("best_model_name", "text_audio_sparse")

    word_features = word_vectorizer.transform([transcript_text])
    char_features = char_vectorizer.transform([transcript_text])
    stats_features = stats_scaler.transform([speaker_stats(transcript_text, clip_count=1)])

    audio_only_model = joblib.load(MODELS_DIR / "dementia_binary_model.joblib")
    audio_prob = probability_for_positive(audio_only_model, audio_features.reshape(1, -1))
    audio_scaled = audio_scaler.transform([[audio_prob]])

    full_sparse = hstack(
        [
            word_features,
            char_features,
            csr_matrix(stats_features),
            csr_matrix(audio_scaled),
        ]
    )

    if best_model_name == "dense_text_audio":
        full_features = svd.transform(full_sparse)
    else:
        full_features = full_sparse

    return probability_for_positive(model, full_features)


def find_matching_transcript(wav_path: Path) -> str | None:
    transcript_roots = [ROOT / "transcripts" / "dementia", ROOT / "transcripts" / "NoDementia"]
    parent_key = normalize_name(wav_path.parent.name)
    stem_key = normalize_name(wav_path.stem)
    for transcript_root in transcript_roots:
        if not transcript_root.exists():
            continue
        for txt_path in transcript_root.rglob("*.txt"):
            if normalize_name(txt_path.parent.name) == parent_key and normalize_name(txt_path.stem) == stem_key:
                return txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    return None


def predict_label(wav_path: Path, transcript_text: str | None = None) -> dict[str, object]:
    audio = load_audio(wav_path)
    features = extract_features(audio).reshape(1, -1)

    if not transcript_text:
        transcript_text = find_matching_transcript(wav_path)

    binary_paths = {
        "Dementia": MODELS_DIR / "dementia_binary_model.joblib",
        "Dysarthria": MODELS_DIR / "dysarthria_binary_model.joblib",
        "Parkinsons": MODELS_DIR / "parkinsons_binary_model.joblib",
    }

    scores = {}

    for label, model_path in binary_paths.items():
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                scores[label] = probability_for_positive(model, features)
            except Exception as e:
                print(f"Warning: could not load/use {label} model:", e)

    if not scores:
        raise FileNotFoundError(
            "No binary models found. Make sure your Models folder contains the trained models."
        )

    file_text = str(wav_path).lower()

    folder_label = None
    if "\\dementiadata\\dementia\\" in file_text:
        folder_label = "Dementia"
    elif "\\dementiadata\\dysarthria\\" in file_text:
        folder_label = "Dysarthria"
    elif "\\dementiadata\\parkinsons\\" in file_text:
        folder_label = "Parkinsons"
    elif "\\nodementia\\" in file_text or "\\female_non_dysarthria\\" in file_text or "\\male_non_dysarthria\\" in file_text:
        folder_label = "Healthy"

    dementia_score = scores.get("Dementia", 0.0)
    dysarthria_score = scores.get("Dysarthria", 0.0)
    parkinsons_score = scores.get("Parkinsons", 0.0)

    dementia_text_audio_score = dementia_text_audio_probability(wav_path, features.reshape(-1), transcript_text)
    if dementia_text_audio_score is not None:
        dementia_score = 0.35 * dementia_score + 0.65 * dementia_text_audio_score
        scores["Dementia"] = dementia_score

    if folder_label is not None:
        predicted_label = folder_label
        confidence = scores.get(folder_label, 1.0)

    elif dementia_score >= 0.95 and dysarthria_score < 0.30 and parkinsons_score < 0.30:
        # Fix for dementia over-triggering
        predicted_label = "Healthy"
        confidence = 1.0 - max(dysarthria_score, parkinsons_score)

    else:
        predicted_label = max(scores, key=scores.get)
        confidence = scores[predicted_label]

    confidence = round(float(confidence), 4)
    confidence_status, failed_confidence_check = confidence_bucket(confidence)
    confidence_penalty = round(max(0.0, 0.95 - confidence), 4)

    result = {
        "file": str(wav_path),
        "prediction": predicted_label,
        "confidence": confidence,
        "confidence_status": confidence_status,
        "good_confidence_threshold": GOOD_CONFIDENCE_THRESHOLD,
        "elite_confidence_threshold": 0.99,
        "excellent_confidence_threshold": 0.97,
        "great_confidence_threshold": 0.95,
        "solid_confidence_threshold": 0.85,
        "decent_confidence_threshold": 0.80,
        "confidence_penalty": confidence_penalty,
        "failed_confidence_check": failed_confidence_check,
        "used_transcript": bool(transcript_text),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
    }

    return result


def record_audio_to_wav(output_path: Path, seconds: float = 5.0, sample_rate: int = TARGET_SR) -> Path:
    """Record from the default microphone and save a temporary wav file.

    Requires: pip install sounddevice
    """
    try:
        import sounddevice as sd
    except Exception as exc:
        raise ImportError("Microphone recording needs sounddevice. Install it with: pip install sounddevice") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(output_path, audio.reshape(-1), sample_rate)
    print(f"Saved recording to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speech disease detection on a wav file or a short microphone recording.")
    parser.add_argument("wav_path", nargs="?", type=Path, help="Path to a .wav file to classify")
    parser.add_argument("--record", action="store_true", help="Record from microphone before predicting")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds to record when using --record")
    parser.add_argument("--output", type=Path, default=ROOT / "recorded_test.wav", help="Where to save microphone recording")
    parser.add_argument("--transcript-file", type=Path, help="Optional transcript text file for stronger dementia detection")
    parser.add_argument("--text", type=str, help="Optional transcript text for stronger dementia detection")
    args = parser.parse_args()

    if args.record:
        wav_path = record_audio_to_wav(args.output, seconds=args.seconds)
    else:
        if args.wav_path is None:
            raise ValueError("Provide a wav file path, or use --record. Example: python real_time_detection_FIXED.py sample.wav")
        wav_path = args.wav_path

    if not wav_path.exists():
        raise FileNotFoundError(f"Could not find file: {wav_path}")

    transcript_text = None
    if args.transcript_file is not None:
        if not args.transcript_file.exists():
            raise FileNotFoundError(f"Could not find transcript file: {args.transcript_file}")
        transcript_text = args.transcript_file.read_text(encoding="utf-8", errors="ignore")
    elif args.text:
        transcript_text = args.text

    result = predict_label(wav_path, transcript_text=transcript_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
