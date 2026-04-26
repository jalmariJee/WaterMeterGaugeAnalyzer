\# Water Meter Gauge Analyser



\## Overview

This project focuses on developing an AI-based system capable of detecting and analyzing water meter gauge dials from images. The goal is to automate the identification and localization of the dial area using deep learning models, primarily CNN and YOLO architectures.



\## Project Evolution



\### 1. Image Cropper

The initial phase involved creating the image_cropper module, which was designed to crop the first dial from the water meter image. This cropped data was used for training and testing a Convolutional Neural Network (CNN) model.



\- Training `For_testing_CNN\train_resnet.py`  

\- Prediction `For_testing_CNN\predict.py`



This stage focused on learning dial features and improving classification accuracy for cropped images.



\### 2. YOLO Training

After the CNN phase, the project transitioned to YOLO (You Only Look Once) for object detection. The YOLO\yolo_train directory contains scripts and configurations for training the YOLO model to perform inference directly on full images.



This approach allowed the model to detect dial locations without requiring prior cropping, enabling end-to-end detection and classification.



\### 3. Final YOLO Project

The final stage, located in Final_YOLO_Project, integrates YOLO for full-image detection, alignment, and classification. This version represents the culmination of the project — a robust pipeline capable of:



\- Detecting water meter dials  

\- Aligning detected regions  

\- Classifying dial readings  



\## Folder Structure

Water meter gauge analyser

│

├── image_cropper                # Cropping tool for dial extraction

├── For_testing_CNN              # CNN training and prediction scripts

│   ├── train_resnet.py

│   └── predict.py

├── YOLO                         # YOLO training scripts and configs

│   └── yolo_train

├── Final_YOLO_Project           # Full detection and classification pipeline

└── runs\detect

└── runs\predict          # Example detection outputs



Code



\## Example Output

Below is an example detection result from the YOLO model



<img width="1080" height="1920" alt="WaterMeter_16" src="https://github.com/user-attachments/assets/8f58d633-f435-4d8d-8dab-e68b9a32bcb7" />




(This image demonstrates the YOLO model successfully detecting the water meter dial location and reading.)



\## How to Use

1\. Prepare Dataset Place raw water meter images in the designated dataset folder.  

2\. Train CNN Run `train_resnet.py` to train the CNN on cropped dial images.  

3\. Train YOLO Use `YOLO\yolo_train` scripts to train the YOLO model for full-image detection.  

4\. Run Inference Execute the final YOLO pipeline in `Final_YOLO_Project` to perform detection and classification.  



\## Future Improvements

\- Design hardware for acquisation of the images

\- Expand dataset diversity for better generalization.  

\- Optimize YOLO model for real-time inference.  



\## Authors

Developed by Ilmari Harilainen, Aryal Pawan, Miikka Rautapää and Sari Vuoskoski



\---



This README provides a structured overview of the Water Meter Gauge Analyser projec

