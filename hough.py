import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import easyocr

def importImages(path):
    """
    Imports all images from a specified folder.

    Args:
        path (str): Path to the folder containing images.

    Returns:
        list: A list of images loaded as OpenCV matrices.
    """
    images = []
    for filename in os.listdir(path):  # Loop through all files in the directory
        img = cv.imread(os.path.join(path, filename))  # Read each image
        if img is not None:  # Only append valid images
            images.append(img)
    return images

def showImage(preprocessed=[False, ], number=1):
    """
    Displays an image, either preprocessed or directly from the folder.

    Args:
        preprocessed (list): A flag and an optional preprocessed image.
        number (int): Index of the image to show if preprocessed is False.

    Returns:
        None
    """
    if preprocessed[0]:  # Check if a preprocessed image is provided
        cv.imshow("Image", preprocessed[1])
    else:  # Otherwise, import and show the image from the folder
        cv.imshow("Image", importImages("data/easy")[number])
    cv.waitKey(0)  # Wait for a key press to close the image
    cv.destroyAllWindows()

def preprocess(gamma=[False, 1.5], equalize=False, grayscale=True, bilateralFilter=False, gaussianBlur=False, medianBlur=True):
    """
    Preprocesses images with optional filters and transformations.

    Args:
        gamma (list): If True, applies gamma correction with the given value.
        equalize (bool): If True, applies histogram equalization.
        grayscale (bool): If True, converts images to grayscale.
        bilateralFilter (bool): If True, applies bilateral filtering.
        gaussianBlur (bool): If True, applies Gaussian blur.
        medianBlur (bool): If True, applies median blur.

    Returns:
        list: A list of preprocessed images.
    """
    images = importImages("data/easy")  # Load images from the folder
    preprocessed = []
    for img in images:
        if grayscale:  # Convert to grayscale if specified
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if medianBlur:  # Apply median blur
            img = cv.medianBlur(img, 5)
        if bilateralFilter:  # Apply bilateral filter
            img = cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        if gaussianBlur:  # Apply Gaussian blur
            img = cv.GaussianBlur(img, (9, 9), 2)
        if gamma[0]:  # Apply gamma correction
            img = img / 255.0  # Normalize to [0, 1]
            img = np.power(img, gamma[1])
            img = (img * 255).clip(0, 255).astype(np.uint8)  # Rescale to [0, 255]
        if equalize:  # Apply histogram equalization
            if len(img.shape) == 2:  # Grayscale image
                img = cv.equalizeHist(img)
            else:  # Color image
                img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
                img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
        preprocessed.append(img)
    return preprocessed

def detect_circles(image_path):
    """
    Detects circles in an image using the Hough Circle Transform.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    image = cv.imread(image_path)  # Load the image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    gray_blurred = cv.medianBlur(gray, 5)  # Apply median blur to reduce noise

    # Detect circles using the Hough Transform
    circles = cv.HoughCircles(
        gray_blurred,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=150,
        param2=50,
        minRadius=35,
        maxRadius=60
    )

    if circles is not None:  # If circles are detected
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the circle outline
            cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)  # Draw the circle center
            
            # Extract circle coordinates
            center_x, center_y, radius = i
            x_min = max(0, center_x - radius)
            y_min = max(0, center_y - radius)
            x_max = min(image.shape[1], center_x + radius)
            y_max = min(image.shape[0], center_y + radius)

            # Crop the circle and display it
            cropped_image = image[y_min:y_max, x_min:x_max]
            plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    # Display the final image with the detected circles
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_speed_signals_with_hough(input_directory, output_directory):
    """
    Processes all images in a directory and subdirectories using the Hough Transform to detect circles,
    performs OCR to read text inside detected circles, and saves annotated images.

    Args:
        input_directory (str): Directory containing the input images (and subfolders).
        output_directory (str): Directory where annotated images will be saved.

    Returns:
        None
    """
    # Initialize the OCR reader
    reader = easyocr.Reader(['ch_sim', 'en'])

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Recursively loop through all images in the input directory
    for root, _, files in os.walk(input_directory):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg')):  # Only process images
                image_path = os.path.join(root, filename)  # Full path to the image

                # Load the image
                image = cv.imread(image_path)

                # Convert the image to grayscale for circle detection
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # Apply median blur to reduce noise
                gray_blurred = cv.medianBlur(gray, 5)

                # Use the Hough Transform to detect circles
                circles = cv.HoughCircles(
                    gray_blurred,
                    cv.HOUGH_GRADIENT,
                    dp=1.2,           # Inverse ratio of resolution
                    minDist=200,      # Minimum distance between detected circles
                    param1=150,       # Higher threshold for edge detection
                    param2=50,        # Accumulator threshold for circle detection
                    minRadius=10,     # Minimum circle radius
                    maxRadius=50      # Maximum circle radius
                )

                # Initialize a list to store detected texts
                detected_texts = []

                # If circles are detected, process each circle
                if circles is not None:
                    circles = np.uint16(np.around(circles))  # Round circle parameters to integers
                    for i in circles[0, :]:
                        # Extract circle coordinates
                        center_x, center_y, radius = i
                        x_min = max(0, center_x - radius)
                        y_min = max(0, center_y - radius)
                        x_max = min(image.shape[1], center_x + radius)
                        y_max = min(image.shape[0], center_y + radius)

                        # Draw the circle outline and center on the original image
                        cv.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)  # Outline
                        cv.circle(image, (center_x, center_y), 2, (0, 0, 255), 3)       # Center

                        # Crop the circle region from the image
                        cropped_image = image[y_min:y_max, x_min:x_max]

                        # Perform OCR on the cropped image
                        ocr_result = reader.readtext(cropped_image)

                        # If OCR is successful, extract the text and annotate the image
                        if ocr_result:
                            text = f"The speed limit is {ocr_result[0][1]}"  # Extract detected text
                            detected_texts.append(text)

                            # Annotate the original image with the detected text
                            cv.putText(image, text, (x_min - 20, y_min - 10), 
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the annotated image to the output directory
                output_image_path = os.path.join(output_directory, filename)
                cv.imwrite(output_image_path, image)
                print(f"Processed and saved: {output_image_path}")