import cv2
from ultralytics import YOLO

#%% convert data to yolo format
import os
import glob
import xml.etree.ElementTree as ET

# Define paths
xml_dir = 'datasets/labelimg'
img_dir = 'datasets/images'
output_dir = 'datasets/labels_yolo'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert XML annotation to YOLO format
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Iterate over all XML files
for xml_file in glob.glob(f"{xml_dir}/*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    file_name = root.find('filename').text.replace('.jpg', '').replace('.png', '')
    
    with open(f"{output_dir}/{file_name}.txt", 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            f.write(f"{cls} " + " ".join([str(a) for a in bb]) + '\n')


#%% split the data into train and validation data

import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
img_dir = 'datasets/images'
label_dir = 'datasets/labels_yolo'
output_base_dir = 'datasets/split/dataset'

# Create directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_base_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'labels', split), exist_ok=True)

# Get all image file paths
images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]
labels = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')]

# Ensure each image has a corresponding label file
images = [img for img in images if os.path.exists(os.path.join(label_dir, os.path.basename(img).replace('.png', '.txt')))]

# Split dataset into training and validation sets
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Copy files to respective directories
def copy_files(file_list, split):
    for file in file_list:
        shutil.copy(file, os.path.join(output_base_dir, 'images', split, os.path.basename(file)))
        label_file = os.path.join(label_dir, os.path.basename(file).replace('.png', '.txt'))
        shutil.copy(label_file, os.path.join(output_base_dir, 'labels', split, os.path.basename(label_file)))

copy_files(train_images, 'train')
copy_files(val_images, 'val')



#%% Load the model and train it on custom data

# Load a YOLO model
yolo = YOLO('yolov8s.pt')  # Use the appropriate model checkpoint

# Train the model on your custom dataset
yolo.train(
    data='custom_dataset.yaml',  # Path to the custom dataset config file
    epochs=10,  # Number of training epochs
    imgsz=640,  # Image size
    batch=1,  # Batch size
    device='cpu'  # Use CPU for training
)

# Validate the model
yolo.val(data='custom_dataset.yaml')


#%% usage of yolo

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Path to the image file
image_path = 'datasets/images/topology_plot_6.png'

# Load the image
frame = cv2.imread(image_path)
if frame is None:
    raise ValueError("Image not found. Please check the path.")

# Perform object detection
results = yolo(frame)

# Iterate over each result
for result in results:
    # Get the classes names
    classes_names = result.names

    # Iterate over each box
    for box in result.boxes:
        # Check if confidence is greater than 40 percent
        if box.conf[0] > 0.4:
            # Get coordinates
            [x1, y1, x2, y2] = box.xyxy[0]
            # Convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the class
            cls = int(box.cls[0])

            # Get the class name
            class_name = classes_names[cls]

            # Get the respective color
            colour = getColours(cls)

            # Draw the rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # Put the class name and confidence on the image
            cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

# Show the image
cv2.imshow('Image', frame)
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
