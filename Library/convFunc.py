import os
import shutil
import yaml
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_yaml_to_yolo_format(yaml_file, image_folder, output_image_folder, output_label_folder, class_mapping):
    """
    Converts a YAML file containing annotations to YOLO format. Copies the images to the output directory and creates corresponding YOLO-formatted label files.

    Args:
        yaml_file (str): Path to the YAML file containing annotations.
        image_folder (str): Path to the folder containing the images.
        output_image_folder (str): Path to the output folder for images.
        output_label_folder (str): Path to the output folder for YOLO labels.
        class_mapping (dict): Dictionary mapping class names to class IDs.
    """
    # Create output folders for images and labels
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    # Create 'classes.txt' file with unique classes
    classes = list(class_mapping.values())
    with open(os.path.join(output_image_folder, 'classes.txt'), 'w') as class_file:
        for cls in classes:
            class_file.write(f"{cls}\n")

    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # Process each entry in the YAML file
    for entry in data:
        img_path = entry['path']
        img_name = os.path.basename(img_path)

        # Copy image to the output folder
        src_image_path = os.path.join(image_folder, img_name)
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, os.path.join(output_image_folder, img_name))

        # Create YOLO label file for the image
        label_file_path = os.path.join(output_label_folder, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
        with open(label_file_path, 'w') as label_file:
            for box in entry['boxes']:
                label = box['label']
                x_min, y_min, x_max, y_max = box['x_min'], box['y_min'], box['x_max'], box['y_max']
                width = x_max - x_min
                height = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # Normalize the coordinates
                x_center /= 1280
                y_center /= 720
                width /= 1280
                height /= 720

                # Map the label to its class ID
                class_id = class_mapping.get(label.lower(), -1)
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def replace_labels_in_file(file_path, label_map):
    """
    Replaces string labels with numeric class IDs in a YOLO label file.

    Args:
        file_path (str): Path to the label file to be modified.
        label_map (dict): Dictionary mapping string labels to numeric class IDs.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            if parts[0] in label_map:
                parts[0] = str(label_map[parts[0]])
            file.write(" ".join(parts) + "\n")

def process_labels_in_directory(directory_path, label_map):
    """
    Processes all label files in a directory, replacing string labels with numeric class IDs.

    Args:
        directory_path (str): Path to the directory containing label files.
        label_map (dict): Dictionary mapping string labels to numeric class IDs.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            replace_labels_in_file(file_path, label_map)

def extract_info_from_xml(xml_file):
    """
    Extracts annotation information from an XML file in Pascal VOC format.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        dict: Dictionary containing annotation details, including bounding boxes and image metadata.
    """
    root = ET.parse(xml_file).getroot()
    info_dict = {'bboxes': []}

    for elem in root:
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
        elif elem.tag == "size":
            image_size = [int(subelem.text) for subelem in elem]
            info_dict['image_size'] = tuple(image_size)
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)

    return info_dict

def convert_xml_to_yolo_format(info_dict, output_dir, class_name_to_id_mapping):
    """
    Converts annotation information from Pascal VOC XML format to YOLO format.

    Args:
        info_dict (dict): Annotation details extracted from an XML file.
        output_dir (str): Directory where YOLO labels will be saved.
        class_name_to_id_mapping (dict): Dictionary mapping class names to class IDs.
    """
    print_buffer = []

    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
            continue

        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = (b["xmax"] - b["xmin"])
        b_height = (b["ymax"] - b["ymin"])

        image_w, image_h, _ = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        print_buffer.append(f"{class_id} {b_center_x:.6f} {b_center_y:.6f} {b_width:.6f} {b_height:.6f}")

    save_file_name = os.path.join(output_dir, info_dict["filename"].replace("png", "txt"))
    os.makedirs(output_dir, exist_ok=True)
    with open(save_file_name, "w") as f:
        f.write("\n".join(print_buffer))

def move_files_to_subfolders(file_list, images_dir, labels_dir, output_dirs, split):
    """
    Moves images and labels to specific subfolders for train, validation, or test splits.

    Args:
        file_list (list): List of file base names to be moved.
        images_dir (str): Path to the directory containing images.
        labels_dir (str): Path to the directory containing labels.
        output_dirs (dict): Dictionary containing output subfolders for each split.
        split (str): Split name (e.g., 'train', 'val', 'test').
    """
    for base_name in file_list:
        img_src = os.path.join(images_dir, base_name + ".jpg")
        lbl_src = os.path.join(labels_dir, base_name + ".txt")

        img_dst = os.path.join(output_dirs[split]["images"], base_name + ".jpg")
        lbl_dst = os.path.join(output_dirs[split]["labels"], base_name + ".txt")

        if os.path.exists(img_src) and os.path.exists(lbl_src):
            shutil.move(img_src, img_dst)
            shutil.move(lbl_src, lbl_dst)