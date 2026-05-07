# Audio Disease Detection Project

This repository contains a machine learning project for speech-based detection of Dementia, Dysarthria, and Parkinson?s disease using audio features, transcript-aware dementia modeling, and real-time `.wav` file prediction.

## Project Structure
- `Data/`
- `Models/`
- `transcripts/`
- `Training_Files/`
  - `train_dementia.ipynb`
  - `train_dysarthria.ipynb`
  - `train_parkinsons.ipynb`
  - `train_multiclass.ipynb`
- `Real_Time_Detection/`
  - `real_time_detection.py`
  - `run_real_time_detection_all_data.py`
- `transcribe_folder.py`
- `Final Project Report - Viprav Lipare.pdf`

## Data Layout
The project expects the main data under:
- `Data/DementiaData/Dementia`
- `Data/DementiaData/NoDementia`
- `Data/DementiaData/Dysarthria`
- `Data/DementiaData/Parkinsons`
- `Data/DementiaData/Female_Non_Dysarthria`
- `Data/DementiaData/Male_Non_Dysarthria`
- `Data/MDVR-KCL`
- `Data/UCI_189`
- `Data/UCI_301`

Transcript files for dementia are stored under:
- `transcripts/dementia`
- `transcripts/NoDementia`

## How To Reproduce
1. Open the notebooks from the `Training_Files/` folder.
2. Run:
   - `train_dementia.ipynb`
   - `train_dysarthria.ipynb`
   - `train_parkinsons.ipynb`
   - `train_multiclass.ipynb`
3. Trained models will be saved in `Models/`.

## Real-Time Detection
Run the live detector on a single file:

```powershell
python Real_Time_Detection/real_time_detection.py "C:\path	o\sample.wav"
```

Use transcript support for dementia if needed:

```powershell
python Real_Time_Detection/real_time_detection.py "C:\path	o\sample.wav" --transcript-file "C:\path	o\sample.txt"
```

Or:

```powershell
python Real_Time_Detection/real_time_detection.py "C:\path	o\sample.wav" --text "transcript text here"
```

## Batch Evaluation
Run the detector across all available dataset files:

```powershell
python Real_Time_Detection/run_real_time_detection_all_data.py
```

This saves:
- `Models/all_data_detection_results.csv`
- `Models/all_data_detection_results.json`

## Final Results Summary
- Dementia multimodal model:
  - validation accuracy: `0.95`
  - test accuracy: `0.9394`
  - test balanced accuracy: `0.90`
- Dysarthria best model:
  - accuracy: `0.9713`
  - balanced accuracy: `0.9671`
- Parkinson?s main audio model:
  - accuracy: `0.9945`
  - balanced accuracy: `0.9969`
- Multiclass detector:
  - accuracy: `0.9256`
  - balanced accuracy: `0.7863`
- Batch real-time detector accuracy over all available files:
  - overall accuracy: `0.9996`

## Notes
- Dementia is handled differently from the other disorders because transcript meaning matters more.
- Parkinson?s and Dysarthria are primarily audio-first models.
- Predictions below 80% confidence are flagged as failed confidence checks.
