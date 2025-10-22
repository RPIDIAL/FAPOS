# FAPOS  
**Facial Appearance Prediction for Orthognathic Surgery with Diffusion Models**

This repository provides the facial landmark index set used in the FAPOS framework, developed for orthognathic surgery-related facial analysis and prediction tasks.

## 📌 Description

The landmark indices shared here are derived from the 478-point facial annotation scheme defined by [Google’s MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), and have been carefully refined in our study to focus on regions that are anatomically and clinically relevant to orthognathic surgery.

In particular:
- Redundant or unrelated regions (e.g., periocular areas, eyebrows) were excluded.
- Landmarks were grouped into five anatomical regions:
  - **Upper Midline**
  - **Upper Left / Upper Right**
  - **Lower Midline**
  - **Lower Left / Lower Right**
- These landmarks are visualized at Landmark_figure.pdf.

The index set includes:
- A total of **282 selected landmark indices** (upper face 119 points, lower face 163 points)
- Format: plain text (.txt)

## 📁 File Contents

- `fapos_landmark_indices.txt`:  
  Contains Python-style dictionary lists of landmark indices grouped by region.

## 📖 Citation

If you use this landmark set in your research or applications, **please cite our paper**:

> [Full citation will go here when available]
