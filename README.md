# Pneumonia Detection Using Deep Learning

# Overview
- This project uses deep learning to identify chest x-ray images as either
- NORMAL / PNEUMONIA

  ---

  # Model
- Architecture: EfficientNetB0 (Transfer Learning)
- Framework: Tensorflow / Keras
- Input size: 224 x 224 images
- Training class weighting to handle imbalance

  ---

 ## Dataset
- Source: Kaggle Chest X-Ray Dataset
- Classes:
  - NORMAL / PNEUMONIA (this includes bacterial and viral)

---

## Results

| Metric        | Score |
|--------------|------|
| Accuracy      | 83.6% |
| Precision     | 0.86 |
| Recall        | 0.86 |

---

## Confusion Matrix

[[168 66]
[ 23 367]]

---

## Observations
- The model performs better at identifying pneumonia than normal cases
- Slight overfitting observed after several epochs
- Class imbalance handled using class weights

---

## Tools Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Kaggle (dataset source)
- Google Colab (development environment)
  
---

## Notes
Google Colab was used instead of a local virtual environment to speed up development and leverage GPU support.
It wasn't usable on Mac for Tensorflow as speed was too slow
GPU speed via Google Colab

---

## Files
- `Vision.ipynb` → Main notebook
- `best_model.keras` → Trained model
- `history.json` → Training history

---
