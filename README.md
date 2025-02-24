# Road Signs and Traffic Lights Detection

## **1. Project Overview**  
This repository contains the materials about the project of the "Computer vision" course in the "Physics of Data" master program, University of Padova.

This project focuses on **object detection in street images**, specifically for **traffic lights** and **road signs** (with an emphasis on speed limit signs). The goal is to accurately detect, classify, and extract relevant information from these objects using essentialy **Deep Learning models**. 

The best description which also shows some results is the report in file LorenzoVigorelliReport.pdf.

Here following instead I describe the main steps of the analysis.
All the functions I used are in the file Library, which includes also the requirement file.

### **Key tasks include:**
- **Traffic Lights Detection** → Detect and classify based on **color and shape**.
- **Road Signs Detection** → Detect and classify **speed limits, stop signs, crosswalks**.
- **Speed Limit Extraction** → Use **OCR (Optical Character Recognition)** to extract numbers from speed limit signs.

The **main object detection algorithm** used in this project is **YOLO (You Only Look Once)**, specifically **YOLOv5 and YOLOv9**.  



## **2. Dataset and Structure**  
The dataset consists of images with **annotated bounding boxes** for road objects. It is divided into:
- **Easy Set** → Images with clear, well-lit, and easily recognizable objects.
- **Hard Set** → Images with occlusions, poor lighting, and challenging detection conditions.

### **Dataset Structure for YOLO**
To train and evaluate YOLO models, the dataset follows this structure:

dataset/  
├── images/  
│   ├── train/  
│   ├── test/  
├── labels/  
│   ├── train/  
│   ├── test/  
Each image must have an associated label file with the format:
<class_id> <x_center> <y_center> <width> <height>

But generally when downloading a dataset we have an annnotation file with all of the information of the images in a single file, where the information aren't the one we need in YOLO, but the class, bounding box limits (vertical and horizontal) and path of the file.
So we need to convert our raw data in the correct format.

However, **raw datasets** often use **XML (Pascal VOC) or YAML** formats, requiring conversion.

---

## **3. Data Processing and Annotation Conversion**  
Since datasets might come in **different annotation formats**, functions are provided to **convert annotations into YOLO format**.

### **YAML to YOLO Conversion**
This function converts a **YAML annotation file** to YOLO format and copies the images to the correct directory.

### **XML (Pascal VOC) to YOLO Conversion**
Extracts **bounding box annotations** from an XML file and converts them to **YOLO format**.

### **Label Processing**
Replaces **text labels** with numeric **class IDs** in YOLO label files.

These functions are defined in convFunc.py file.
---

## **4. Image Preprocessing**  
To enhance object detection performance, images undergo **preprocessing** using:
- **Gamma Correction** → Adjust brightness.
- **Histogram Equalization** → Enhance contrast.
- **Grayscale Conversion** → Improve edge detection.
- **Bilateral Filtering** → Reduce noise while preserving edges.
- **Gaussian Blur** → Smooth the image.

These preprocessing techniques could help optimize the images before feeding them into the model.

Preprocessing functions are defined in hough.py file.

---

## **5. Object Detection with YOLO**  

### **YOLO Model Selection**
| Model  | Parameters | Speed | Accuracy |
|--------|-----------|--------|----------|
| **YOLOv5** | 9M | Fast | Moderate |
| **YOLOv9** | 25M | Slower | Higher |

- **YOLOv5** → Faster, lightweight, real-time applications.
- **YOLOv9** → More accurate but computationally expensive.

### **YOLO Object Detection**
- **Traffic lights detection** → Detect and classify the traffic lights based on **color**.
- **Traffic signs detection** → Recognize stop signs, speed limits, and crosswalks.

---

## **6. Speed Limit Sign Detection & OCR**
For **speed limit extraction**, we use **OCR**, which extracts numerical values from speed limit signs.

### **Speed Limit Detection via Hough Transform**
An alternative method to detect circular road signs using **Hough Circle Transform**.

---

## **7. YOLO Model Training**

### **Training a YOLO Model**
Training a YOLO model involves:
- Providing a correctly formatted **dataset**.
- Adjusting **hyperparameters** for optimal accuracy.
- Running the training process to fine-tune the model.

### **Training Modes**
- **Night Mode** → Optimized for low-light conditions.
- **Low Data Mode** → Optimized for small datasets.
- **Regular Mode** → Balanced settings.

The YOLO training functions are in the file yoloTrainingProcess.py

---

## **8. Model Evaluation**

To evaluate model performance, we use:
- **Loss Functions** → Measures errors in detection.
- **Precision & Recall** → Evaluates classification accuracy.
- **F1 Score** → Harmonic mean of precision and recall.
- **mAP (Mean Average Precision)** → Measures detection accuracy.
- **Confusion Matrix Analysis** → Helps visualize classification performance.

---

## **9. Results & Performance Analysis**

### **Key Findings**
- **Fine-tuning YOLO significantly improved accuracy.**  
- **YOLOv9 outperformed YOLOv5 but is computationally expensive.**  
- **OCR successfully extracted speed limits.**  
- **Detecting left/right-turn traffic lights remains challenging.**  
- **Limited dataset availability affects performance.**  

---

## **10. Future Improvements**
- Optimize **hyperparameter tuning**.  
- Expand the dataset with **more diverse traffic signs**.  
- Improve **real-world testing** in varied conditions.  
- Address **occlusions and partially visible objects**.  


