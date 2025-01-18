# Object Recognition in Street Images

This project focuses on object detection in street images, specifically for detecting traffic lights and traffic signals. The process is divided into several key steps, from loading images to applying object detection models.

## Steps

### 1. **Load Images**

There are two sets of images:

- **Easy Set**: Images where object detection is relatively straightforward.
- **Hard Set**: Images with more complex conditions for object detection.

Functions for this step:
- **`importImages(path)`**: Imports all images from a specified folder.
- **`showImage(preprocessed=[False], number=1)`**: Displays an image (optionally preprocessed).

### 2. **Preprocess Images**

Preprocessing is crucial to improve the quality and consistency of input data for object detection. By adjusting various parameters, we can optimize images for specific conditions.

The preprocessing function accepts the following options:
- **`gamma=[False, 1.5]`**: Apply a gamma transformation for brightness adjustment.
- **`equalize=False`**: Equalize the histogram to improve contrast.
- **`grayscale=False`**: Convert the image to grayscale.
- **`bilateralFilter=False`**: Apply a bilateral filter for noise reduction while preserving edges.
- **`gaussianBlur=False`**: Apply Gaussian blur to smooth the image.

### 3. **Object Detection**

We implement two models for detecting:
1. **Traffic Lights**
2. **Traffic Signals**

The detection models use the **YOLOv5** architecture, pretrained on the **COCO** dataset and fine-tuned for these specific use cases.

#### **Traffic Lights**
- The dataset includes various traffic light types, classified by color (red, yellow, green) and shape (left-turn, right-turn).
- **Alternative Approach**: If computational power is limited, use the YOLO model pretrained on COCO and divide the image into three sections. Classify the color based on the prevalent color in the section.

#### **Traffic Signals**
- The dataset detects multiple traffic signals: traffic lights, stop signs, speed limits, and crosswalks.
- For speed limit detection, a **CNN** or **OCR** (Tesseract) approach can be applied to extract the text on the signs.
- **Alternative Approach**: Use the **Hough Transform** to detect circles (useful for speed limit signs) and classify them based on their features. Preprocessing is crucial for optimal results in this approach.

### 4. **YOLO (You Only Look Once)**

YOLO is the primary architecture used for object detection in this project. The choice between different versions of YOLO is an important consideration:

- **YOLOv5**: This model has approximately 8 million parameters. It is lighter and faster, which makes it suitable for real-time detection or cases with limited computational resources. However, it is generally less precise than newer versions.
  
- **YOLOv9**: This newer version of YOLO is more accurate but comes with a tradeoff in speed. YOLOv9 has more parameters and is slower to train, making it more suitable for cases where high accuracy is required, and computational resources are available.

For the best results, fine-tuning a pretrained YOLO model on your specific dataset is recommended.

#### **Training the YOLO Model**
- Fine-tuning a pretrained YOLO model (either YOLOv5 or YOLOv9) on a relevant dataset significantly improves detection performance.
- For speed limit signs, a dataset with speed limit information as class labels will be required.
- Training involves providing correctly formatted data and training the model for optimal accuracy.

#### **Dataset Format for YOLO**
To train and validate the YOLO model, the dataset must be structured as follows:
dataset/
    images/
        train/
        test/
    labels/
        train/
        test/
Each image must have an associated label file with the format:
<class_id> <x_center> <y_center> <width> <height>

But generally when downloading a dataset we have an annnotation file with all of the information of the images in a single file, where the information aren't the one we need in YOLO, but the class, bounding box limits (vertical and horizontal) and path of the file.
So we need to convert our raw data in the correct format.
