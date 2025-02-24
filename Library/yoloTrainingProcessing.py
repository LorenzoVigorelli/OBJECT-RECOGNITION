from ultralytics import YOLO  
import easyocr 
import cv2 
import numpy as np  
import os  
from datetime import datetime  

def process_images_yolo(use_inference, input_directory, output_directory, trained_model_path, base_model="yolov5s.pt"):
    """
    Processes all images in a directory and subdirectories using YOLO and saves the results.

    Args:
        use_inference (bool): If True, uses the best-trained weights; if False, uses the base model.
        base_model (str): Base YOLO model file to use (e.g., 'yolov5s.pt' or other versions).
        input_directory (str): Directory containing the input images (and subfolders).
        output_directory (str): Directory where the results will be saved.

    Returns:
        None
    """
    # Determine the model to use based on the mode
    model_path = trained_model_path if use_inference else base_model
    model = YOLO(model_path)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Recursively process all image files in the input directory
    for root_dir, _, file_list in os.walk(input_directory):
        for file_name in file_list:
            if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
                full_image_path = os.path.join(root_dir, file_name)  # Get the complete path to the image

                # Perform inference using the YOLO model
                results = model.predict(full_image_path)

                # Generate output file name
                model_name = os.path.basename(model_path).split('.')[0]
                image_name = os.path.splitext(file_name)[0]
                output_file_name = f"{model_name}_{image_name}.jpg"
                output_file_path = os.path.join(output_directory, output_file_name)

                # Save results for each detection
                for result in results:
                    result.save(output_file_path)

                print(f"Result saved at: {output_file_path}")


def train_yolo(data_directory, output_directory, epochs=50, device=torch.device("mps"), base_model="yolov5s.yaml",
               mode="manual", batch_size=16, img_size=640, learning_rate=0.01, momentum=0.937, weight_decay=0.0005, 
               patience=7, seed=0, workers=8, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, 
               scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0):
    """
    Trains a YOLO model with data from a directory, using either default or custom configurations.

    Args:
        data_directory (str): Directory containing the training data and YAML file.
        output_directory (str): Directory where the training results will be saved.
        epochs (int): Number of training epochs.
        device: Device to use for training (e.g., CPU, GPU, MPS).
        base_model (str): Base YOLO model configuration (e.g., 'yolov5s.yaml').
        mode (str): Mode for hyperparameter configuration ("manual", "night", "low_data", "balanced", "automatic").
        batch_size (int): Batch size for training.
        img_size (int): Image size for training.
        learning_rate (float): Initial learning rate.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for regularization.
        patience (int): Early stopping patience for learning rate reduction.
        seed (int): Random seed for reproducibility.
        workers (int): Number of workers for data loading.
        hsv_h, hsv_s, hsv_v: HSV augmentation parameters.
        degrees, translate, scale, shear, perspective: Geometric augmentation parameters.
        flipud, fliplr: Flip augmentation probabilities.
        mosaic, mixup: Mosaic and MixUp augmentation probabilities.

    Returns:
        model: Trained YOLO model object.
    """
    # Create a unique output subdirectory for results
    model_name = os.path.basename(base_model).split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = os.path.join(output_directory, f"{model_name}_{mode}_{timestamp}")
    os.makedirs(results_subdir, exist_ok=True)

    # If in automatic mode, use the default settings
    if mode == "automatic":
        model = YOLO(base_model)
        model.train(
            data=data_directory,
            epochs=epochs,
            device=device,
            project=results_subdir
        )
        return model

    # Configure hyperparameters based on mode
    if mode == "night":
        batch_size, img_size, learning_rate = 16, 640, 0.001
        hsv_h, hsv_s, hsv_v = 0.005, 0.5, 0.2
        degrees, translate, scale, shear, perspective = 0.0, 0.05, 0.3, 0.0, 0.0
        flipud, fliplr, mosaic, mixup = 0.0, 0.5, 0.5, 0.2

    elif mode == "low_data":
        batch_size, img_size, learning_rate = 16, 512, 0.002
        hsv_h, hsv_s, hsv_v = 0.02, 0.8, 0.6
        degrees, translate, scale, shear, perspective = 10.0, 0.2, 0.6, 5.0, 0.02
        flipud, fliplr, mosaic, mixup = 0.1, 0.5, 1.0, 0.5

    elif mode != "manual":
        raise ValueError("Invalid mode. Use 'manual', 'night', 'low_data', or 'automatic'.")

    # Train the model with custom configurations
    model = YOLO(base_model)
    model.train(
        data=data_directory,
        epochs=epochs,
        device=device,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        workers=workers,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        project=results_subdir
    )
    return model

def extract_bounding_boxes(image_path, model, confidence_threshold=0.5):
    """
    Extracts bounding boxes for detected objects in an image using a YOLO model.

    Args:
        image_path (str): Path to the input image.
        model: Loaded YOLO model object.
        confidence_threshold (float): Confidence threshold for filtering detections.

    Returns:
        cropped_images (list): List of cropped images of detected regions.
        detections (list): List of tuples containing bounding box coordinates, confidence, and class ID.
    """
    # Load the input image
    image = cv2.imread(image_path)

    # Perform inference with the YOLO model
    results = model(image)

    # Retrieve bounding boxes from the first result
    first_result = results[0]
    bounding_boxes = first_result.boxes

    detections = []
    cropped_images = []

    if bounding_boxes is not None:
        for box in bounding_boxes:
            # Extract coordinates and details from the box
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
            confidence = float(box.conf[0])  # Detection confidence
            class_id = int(box.cls[0])  # Detected class ID

            # Filter based on confidence threshold
            if confidence > confidence_threshold:
                detections.append((x_min, y_min, x_max, y_max, confidence, class_id))
                # Crop the detected region from the image
                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_images.append(cropped_image)

    return cropped_images, detections


def analyze_traffic_light_color(region):
    """
    Analyzes the dominant color in a region of an image to determine traffic light status.

    Args:
        region: Image region (numpy array) to analyze.

    Returns:
        str: The dominant color in the region ('Red', 'Yellow', 'Green', or 'Unknown').
    """
    # Convert the image region to HSV color space
    hsv_image = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for traffic light colors
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([15, 150, 150])
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([35, 100, 50])
    green_upper = np.array([85, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Count non-zero pixels in each mask
    red_count = np.count_nonzero(red_mask)
    yellow_count = np.count_nonzero(yellow_mask)
    green_count = np.count_nonzero(green_mask)

    # Determine the dominant color
    if red_count > yellow_count and red_count > green_count:
        return "Red"
    elif yellow_count > red_count and yellow_count > green_count:
        return "Yellow"
    elif green_count > red_count and green_count > yellow_count:
        return "Green"
    else:
        return "Unknown"


def process_traffic_light_signals(input_directory, output_directory, model, confidence_threshold=0.5):
    """
    Processes images in a directory to detect traffic lights, analyzes their color,
    and writes the detected color on the image.

    Args:
        input_directory (str): Directory containing input images.
        output_directory (str): Directory to save processed images.
        model: Loaded YOLO model object.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all images in the input directory
    for root_dir, _, file_list in os.walk(input_directory):
        for file_name in file_list:
            if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
                full_image_path = os.path.join(root_dir, file_name)  # Full path to the image

                # Load the input image
                image = cv2.imread(full_image_path)

                # Perform inference using the YOLO model
                results = model(image)

                # Retrieve bounding boxes from the first result
                first_result = results[0]
                bounding_boxes = first_result.boxes

                if bounding_boxes is not None:
                    for box in bounding_boxes:
                        # Extract coordinates and details
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Check if the detection is a traffic light and passes the confidence threshold
                        if confidence > confidence_threshold and class_id == 9:  # Class 9 assumed to be 'traffic light'
                            cropped_region = image[y_min:y_max, x_min:x_max]

                            # Analyze the traffic light's color
                            height, width, _ = cropped_region.shape

                            # Split the cropped image into three vertical regions (top, middle, bottom)
                            top_region = cropped_region[0:height // 3, :]
                            middle_region = cropped_region[height // 3:2 * height // 3, :]
                            bottom_region = cropped_region[2 * height // 3:height, :]

                            # Determine the dominant color in each region
                            top_color = analyze_traffic_light_color(top_region)
                            middle_color = analyze_traffic_light_color(middle_region)
                            bottom_color = analyze_traffic_light_color(bottom_region)

                            # Determine the traffic light's status
                            traffic_light_color = "Unknown"
                            if top_color == "Red" or middle_color == "Red" or bottom_color == "Red":
                                traffic_light_color = "Red"
                            elif top_color == "Yellow" or middle_color == "Yellow" or bottom_color == "Yellow":
                                traffic_light_color = "Yellow"
                            elif top_color == "Green" or middle_color == "Green" or bottom_color == "Green":
                                traffic_light_color = "Green"

                            # Draw the bounding box and label on the image
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            label = f"{traffic_light_color} Traffic Light"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(image, label, (x_min, y_min - 10), font, 1, (0, 255, 0), 2)

                            # Save the processed image
                            output_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_{traffic_light_color}_light.jpg")
                            cv2.imwrite(output_file_path, image)

                            print(f"Processed and saved: {output_file_path}")

def extract_speed_limit_sign(image_path, model, confidence_threshold=0.5):
    """
    Extracts speed limit signs from an image using a YOLO model and performs OCR on the detected signs.

    Args:
        image_path (str): Path to the input image.
        model: Loaded YOLO model object.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        cropped_signs (list): List of cropped images of detected speed limit signs.
        detections (list): List of tuples with bounding box details and confidence levels.
        ocr_results (list): List of OCR results for each detected speed limit sign.
    """
    # Use the helper function to extract bounding boxes and cropped images
    cropped_signs, detections = extract_bounding_boxes(image_path, model, confidence_threshold=confidence_threshold)

    # Initialize the OCR reader
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])

    # Perform OCR on each cropped image
    ocr_results = []
    for sign_image in cropped_signs:
        if sign_image is not None:
            # Use OCR to extract text
            text_result = ocr_reader.readtext(sign_image)
            ocr_results.append(text_result)
        else:
            ocr_results.append(None)

    return cropped_signs, detections, ocr_results

def process_speed_signals(use_inference, input_directory, output_directory, trained_model_path, 
                          base_model="yolov5s.pt", confidence_threshold=0.5):
    """
    Processes images in a directory to detect speed limit signs using YOLO, performs OCR, and saves the results.

    Args:
        use_inference (bool): If True, uses the trained model; otherwise, uses the base YOLO model.
        base_model (str): Base YOLO model file to use.
        input_directory (str): Directory containing input images (and subdirectories).
        output_directory (str): Directory where the results will be saved.
        trained_model_path (str): Path to the trained YOLO model weights.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        None
    """
    # Determine the model to use
    model_path = trained_model_path if use_inference else base_model
    model = YOLO(model_path)

    # Initialize the OCR reader
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all images in the input directory
    for root_dir, _, file_list in os.walk(input_directory):
        for file_name in file_list:
            if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
                full_image_path = os.path.join(root_dir, file_name)

                # Load the input image
                image = cv2.imread(full_image_path)

                # Perform inference using the YOLO model
                results = model(image)
                first_result = results[0]
                bounding_boxes = first_result.boxes

                # Process each detected bounding box
                if bounding_boxes is not None:
                    for box in bounding_boxes:
                        # Extract box coordinates and confidence
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])

                        # Filter based on confidence threshold
                        if confidence > confidence_threshold:
                            # Crop the detected region from the image
                            cropped_sign = image[y_min:y_max, x_min:x_max]

                            # Perform OCR on the cropped image
                            ocr_result = ocr_reader.readtext(cropped_sign)

                            # Draw the bounding box on the original image
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                            # Add the detected text to the image, if available
                            if ocr_result:
                                detected_text = ocr_result[0][1]  # Extract the detected text
                                cv2.putText(image, f"Speed limit {detected_text} km/h", 
                                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, (0, 255, 0), 2)

                # Save the processed image with annotations
                output_file_path = os.path.join(output_directory, file_name)
                cv2.imwrite(output_file_path, image)

                print(f"Result saved at: {output_file_path}")