# Face Detection Methods

A collection of important face detection methods used in computer vision, from early neural-network approaches to modern deep-learning models.

This repository is intended for students and beginners who want a short reference explaining how face detection evolved over time.

---

## Table of Contents

1. Introduction
2. Methods Covered
3. Comparison Table
4. Requirements
5. Usage
6. Applications
7. Conclusion

---

## Introduction

Face detection is the task of finding the location of one or more human faces in an image or video. It is one of the first stages in many computer vision systems such as face recognition, attendance systems, security cameras, and mobile phone face unlock.

Older methods relied on handcrafted features and simple classifiers. Modern methods use deep learning and are much more accurate, especially when faces are rotated, blurred, partially hidden, or far from the camera.

---

## Methods Covered

This repository includes the following major face detection methods:

* Haar Cascade Face Detection
* Neural Network Face Detection
* Improved Haar Features
* Histogram of Oriented Gradients
* Multi-task CNN Face Detection
* YOLO Object Detection
* RetinaFace

---

## Comparison Table

| Author(s)                    | Year | Method                          | Main Contribution                                                  |
| ---------------------------- | ---- | ------------------------------- | ------------------------------------------------------------------ |
| Paul Viola and Michael Jones | 2001 | Haar Cascade Face Detection     | Introduced a fast and efficient real-time face detection algorithm |
| Rowley, Baluja, and Kanade   | 1998 | Neural Network Face Detection   | One of the earliest neural-network-based face detection methods    |
| Lienhart and Maydt           | 2002 | Improved Haar Features          | Improved Haar Cascade by adding rotated features                   |
| Dalal and Triggs             | 2005 | Histogram of Oriented Gradients | Used gradient features for more reliable detection                 |
| Zhang et al.                 | 2016 | Multi-task CNN Face Detection   | Combined face detection and alignment in one network               |
| Redmon et al.                | 2016 | YOLO Object Detection           | High-speed real-time detection using a single neural network       |
| Deng et al.                  | 2020 | RetinaFace                      | Accurate face detection with landmark prediction                   |

---

## Requirements

To experiment with these methods in Python, install the following libraries:

```bash
pip install opencv-python numpy matplotlib
```

For deep learning based methods such as YOLO, RetinaFace, and CNN-based face detection:

```bash
pip install tensorflow torch torchvision
```

---

## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/face-detection-methods.git
cd face-detection-methods
```

Run a sample Python file:

```bash
python main.py
```

Example OpenCV face detection:

```python
import cv2

img = cv2.imread("face.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_detector.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
```

---

## Method Summary

### Haar Cascade Face Detection

This method uses Haar-like features and a cascade of classifiers. It is very fast and works well for simple frontal face detection. However, it struggles when the face is rotated or partially hidden.

### Neural Network Face Detection

This was one of the first machine-learning-based face detection methods. It used early neural networks and performed better than rule-based approaches at that time.

### Improved Haar Features

This method improved the original Haar Cascade technique by introducing rotated Haar features. Because of this, it became better at detecting tilted faces.

### Histogram of Oriented Gradients

This method extracts edge and gradient information from the image. It is more robust to lighting changes and is often used with a Support Vector Machine classifier.

### Multi-task CNN Face Detection

This method uses convolutional neural networks to detect both faces and facial landmarks together. It performs much better than older approaches in difficult images.

### YOLO Object Detection

YOLO stands for "You Only Look Once". It detects objects in a single pass through the image, making it very fast. It can also be trained specifically for face detection.

### RetinaFace

RetinaFace is a modern deep-learning-based method that provides very high accuracy. It can detect small faces, blurred faces, and faces at different angles.

---

## Applications

* Face unlock on smartphones
* Student attendance systems
* CCTV and security monitoring
* Social media face tagging
* Driver monitoring systems
* Smart cameras and surveillance

---

## Conclusion

Face detection has changed significantly over the years. Traditional methods such as Haar Cascade are still useful when speed and low resource usage are important. Modern methods such as YOLO and RetinaFace are better when higher accuracy is required.

If you are creating a lightweight beginner project, Haar Cascade is often enough. If you are building a modern AI application, RetinaFace or YOLO is usually the better choice.
