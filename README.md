# Object-Detection-Model-
# RoadVision: Vehicle Detection and Classification in Bangladesh Highway Surveillance Imagery

## Team Information

Team Name: **EWU AI Innovations**

Institution: East West University

Department: Computer Science & Engineering

Members:

1. MD Sirajul Islam  
2. HadiuL Alam Hredoy  
3. Shakkar Chowdhury

Submission Date:24.06.26


## Abstract

This project proposes an intelligent vehicle detection and classification system designed specifically for Bangladesh highway surveillance imagery. Due to the diversity of local transportation systems and challenging road conditions, conventional pretrained object detection models often fail to generalize effectively.

To address this challenge, our team developed a deep learning-based object detection framework capable of detecting and classifying 13 categories of vehicles using transfer learning and advanced augmentation techniques. The system predicts object locations, categories, and confidence scores to support traffic analytics and intelligent transportation applications.

Keywords:
Artificial Intelligence, Computer Vision, Object Detection, Vehicle Classification, YOLO, Traffic Analytics

---

## 1. Introduction

Bangladesh experiences highly heterogeneous traffic conditions where multiple vehicle categories operate simultaneously on roads and highways.

This project aims to build an automated surveillance analysis system that can:

* Detect vehicles automatically
* Classify vehicle categories
* Improve traffic monitoring
* Enable intelligent transportation analysis

The proposed solution leverages modern deep learning techniques for robust object detection.

---

## 2. Objectives

Primary objectives:

* Detect vehicles accurately
* Generate bounding box predictions
* Classify vehicles into 13 categories
* Improve detection under real-world conditions
* Maximize mAP@0.5 performance

---

## 3. Dataset Overview

Dataset Characteristics:

* Approximately 900 labeled images
* Real highway surveillance environment
* 13 vehicle classes
* Complex traffic scenes

Dataset Components:

data/
├── train/
├── valid/
├── images/
├── labels/
└── dataset.yaml

---

## 4. Methodology

### Data Preparation

* Data cleaning
* Annotation verification
* Dataset split

### Data Augmentation

* Mosaic
* Random flip
* Brightness adjustment
* Blur simulation

### Model Development

Selected Model:
YOLO-based Object Detection

Training Setup:

* Epoch: ______
* Batch Size: ______
* Image Size: ______
* Learning Rate: ______

---

## 5. Experimental Results

Evaluation Metric:
mAP@0.5

Results:

Training Accuracy: ______

Validation Accuracy: ______

Precision: ______

Recall: ______

Final mAP: ______

----

## 6. Challenges

* Small training dataset
* Domain adaptation issue
* Vehicle overlap
* Weather variability
* Low image resolution

---

## 7. Future Scope

Future enhancements include:

* Semi-supervised learning
* Ensemble models
* Larger datasets
* Real-time deployment

---

## 8. Conclusion

The project demonstrates an AI-driven approach for detecting and classifying vehicles from Bangladesh highway surveillance imagery. The developed solution can contribute to intelligent traffic systems and future smart city applications.

---

## Team Contribution

Member 01:MD Sirajul Islam  
Data Preparation

Member 02:Hadiul Alam Hredoy  
Model Training

Member 03:Shakkor Chowdhury  
Evaluation




## References

[1] YOLO Documentation

[2] Object Detection Research Literature

[3] Traffic Surveillance Research

END

