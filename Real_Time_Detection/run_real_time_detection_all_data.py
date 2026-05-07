from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import real_time_detection as rtd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "Data" / "DementiaData"
OUTPUT_JSON = ROOT / "Models" / "all_data_detection_results.json"
OUTPUT_CSV = ROOT / "Models" / "all_data_detection_results.csv"

CONFIDENCE_ORDER = [
    "Elite",
    "Excellent",
    "Great",
    "Good",
    "Decent",
    "Weak",
    "Bad",
]


LABEL_MAP = {
    "Dementia": "Dementia",
    "Dysarthria": "Dysarthria",
    "Parkinsons": "Parkinsons",
    "NoDementia": "Healthy",
    "Female_Non_Dysarthria": "Healthy",
    "Male_Non_Dysarthria": "Healthy",
}


def main() -> None:
    rows: list[dict[str, object]] = []

    for folder_name, expected_label in LABEL_MAP.items():
        folder_path = DATA_DIR / folder_name
        if not folder_path.exists():
            continue

        wav_files = sorted(folder_path.rglob("*.wav"))
        print(f"{folder_name}: {len(wav_files)} wav files")

        for wav_path in wav_files:
            try:
                result = rtd.predict_label(wav_path)
                predicted_label = str(result.get("prediction", ""))
                correct = predicted_label == expected_label

                rows.append(
                    {
                        "file": str(wav_path),
                        "folder": folder_name,
                        "expected_label": expected_label,
                        "predicted_label": predicted_label,
                        "confidence": float(result.get("confidence", 0.0)),
                        "confidence_status": str(result.get("confidence_status", "Bad")),
                        "confidence_penalty": float(result.get("confidence_penalty", 0.0)),
                        "failed_confidence_check": bool(result.get("failed_confidence_check", False)),
                        "used_transcript": bool(result.get("used_transcript", False)),
                        "correct": correct,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": str(wav_path),
                        "folder": folder_name,
                        "expected_label": expected_label,
                        "predicted_label": "ERROR",
                        "confidence": 0.0,
                        "confidence_status": "Bad",
                        "confidence_penalty": 0.95,
                        "failed_confidence_check": True,
                        "used_transcript": False,
                        "correct": False,
                        "error": str(exc),
                    }
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print()
    print("saved csv:", OUTPUT_CSV)
    print("saved json:", OUTPUT_JSON)

    if results_df.empty:
        print("no results")
        return

    print()
    print("overall accuracy:", round(float(results_df["correct"].mean()), 4))
    print()

    summary = (
        results_df.groupby("expected_label", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    print(summary)

    print()
    confidence_summary = (
        results_df.groupby("confidence_status", as_index=False)
        .agg(
            count=("file", "count"),
            accuracy=("correct", "mean"),
            average_confidence=("confidence", "mean"),
            failed_count=("failed_confidence_check", "sum"),
        )
    )
    confidence_summary["sort_order"] = confidence_summary["confidence_status"].apply(
        lambda x: CONFIDENCE_ORDER.index(x) if x in CONFIDENCE_ORDER else len(CONFIDENCE_ORDER)
    )
    confidence_summary = confidence_summary.sort_values("sort_order").drop(columns=["sort_order"])
    print(confidence_summary)

    print()
    print("sample mistakes:")
    mistakes = results_df[results_df["correct"] == False].head(20)
    if mistakes.empty:
        print("no mistakes found")
    else:
        print(mistakes[["expected_label", "predicted_label", "confidence", "file"]].to_string(index=False))


if __name__ == "__main__":
    main()
