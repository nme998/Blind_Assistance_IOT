#import sys
#sys.path.append("D:/opencv/build/bin")

import json
import cv2
import random
from keras.utils import load_img, img_to_array
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data_file_path = "D:/IOT/train/data"
labels_file_path = "D:/IOT/train/labels/preprocessed_labels.csv"
classes_file_path = "D:/IOT/train/metadata/fixed_classes.csv"
hierarchy_file_path = "D:/IOT/train/metadata/hierarchy.json"
image_ids_file_path = "D:/IOT/train/metadata/filtered_image_ids.csv"

labels = pd.read_csv(labels_file_path)
classes = pd.read_csv(classes_file_path)
image_ids = pd.read_csv(image_ids_file_path)



class_distribution = labels['LabelName'].value_counts()

# Find the minimum number of samples per class
min_samples_per_class = class_distribution.min()
print("Class_distribution: " + str(class_distribution) + " min_samples_per_class: " + str(min_samples_per_class))

# Initialize lists to store data for training and testing
train_data = []
test_data = []

# Iterate over each class
for label, count in class_distribution.items():
    # Get samples for the current class
    class_samples = labels[labels['LabelName'] == label]
    
    # Split samples into training and testing sets
    train_samples, test_samples = train_test_split(class_samples, test_size=0.2, random_state=42)  # Adjust test_size as needed
    
    # Add the split samples to the respective lists
    train_data.append(train_samples)
    test_data.append(test_samples)

# Concatenate data from all classes for training and testing sets
train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

# Shuffle the data
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

print(train_df)

random_index = random.randint(0, len(train_df) - 1)
image_row = train_df.iloc[random_index]

# Get the ImageID or name of the selected image
image_id = image_row['ImageID']  # Assuming the column name is 'ImageID'
class_name = image_row['classes']
# Load the image
#image_path = data_file_path + "/" + image_ids[image_ids['ImageID'] == image_id].iloc[0]["ImageID"] + ".jpg"
image_path = data_file_path + "/" + image_id + ".jpg"
try:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Unable to load image")
    # Display the image
    cv2.imshow("Image", image)
    print("ImageID or Name:", image_id)
    print("Class: ", class_name)
except Exception as e:
    print(f"Error loading image '{image_path}': {e}")

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
image_index = 0

# Load the image
image_path = data_file_path + "/" + image_ids.iloc[image_index]["ImageID"] + ".jpg"
image = cv2.imread(image_path)

# Get the corresponding labels for the image
image_labels = labels[labels["ImageID"] == image_ids.iloc[image_index]["ImageID"]]

# Get the corresponding metadata for the image
image_classes = classes[classes["index"] == image_ids.iloc[image_index]["ImageID"]]

# Print the image along with the corresponding class and metadata
cv2.imshow("Image", image)
print("Labels:")
print(image_labels)
print("Metadata:")
print(image_classes)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows() 
'''