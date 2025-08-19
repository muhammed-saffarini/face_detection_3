import numpy as np
import cv2
import time
import hpelm
import pandas as pd
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the VGG16 model pre-trained on ImageNet, excluding the top layers
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Function to extract HOG features
def extract_hog_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray_img, block_norm='L2-Hys', visualize=True)
    return features


# Function to extract VGG16 features
def extract_vgg16_features(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    features = vgg16_model.predict(img_array)
    features = features.flatten()
    return features


# Function to train an ELM classifier
def train_elm_classifier(features, labels):
    features = np.array(features)  # Ensure features are NumPy arrays
    labels = np.array(labels)  # Ensure labels are NumPy arrays
    # Convert labels to 2D (required by hpelm)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)  # Make labels a 2D array

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create and train the ELM model using the hpelm library
    elm_model = hpelm.ELM(inputs=features_scaled.shape[1], outputs=labels.shape[1])  # Ensure correct output shape

    # elm_model = hpelm.ELM(inputs=features_scaled.shape[1], outputs=len(np.unique(labels)))
    elm_model.add_neurons(100, 'sigm')  # 100 neurons with sigmoid activation
    elm_model.train(features_scaled, labels)

    return elm_model, scaler


# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1,conf_matrix





# Assuming y_pred is the output from the model, we need to convert continuous predictions to discrete class labels
def convert_continuous_to_class_labels(y_pred, is_binary=True):
    if is_binary:
        # For binary classification, apply thresholding (e.g., threshold of 0.5)
        return (y_pred > 0.5).astype(int)
    else:
        # For multi-class classification, choose the class with the highest probability
        return np.argmax(y_pred, axis=1)



# Set up ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Directory containing training and testing images (update with your own dataset directory)
train_dir = '/root/face_project/dataset/dataset/final_dataset'
val_dir = '/root/face_project/dataset/dataset/validation'
test_dir = '/root/face_project/dataset/dataset/Test'  # Path for test data

batch_size = 32
img_size = (224, 224)

# Load training and testing data using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',  # 'sparse' for integer labels, 'categorical' for one-hot encoded
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',  # 'sparse' for integer labels, 'categorical' for one-hot encoded
)

# Measure the training time
start_training_time = time.time()

# Extract features from VGG16 and HOG for each batch in the training set
vgg16_features_train = []
hog_features_train = []
labels_train = []
for i in range(len(train_generator)):
    x_batch, y_batch = train_generator[i]

    for img, label in zip(x_batch, y_batch):
        vgg16_feat = extract_vgg16_features(img)
        hog_feat = extract_hog_features(img)
        vgg16_features_train.append(vgg16_feat)
        hog_features_train.append(hog_feat)
        labels_train.append(label)

    if i == len(train_generator) - 1 or i == 1:
        print("OKKKKKKKKK")
        break
print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
# Combine VGG16 and HOG features for training
combined_features_train = [np.concatenate((vgg, hog)) for vgg, hog in zip(vgg16_features_train, hog_features_train)]

# Train the ELM classifier
elm_classifier, scaler = train_elm_classifier(combined_features_train, labels_train)

all_results =[]
all_results_test =[]


# Extract features from VGG16 and HOG for each batch in the testing set
# vgg16_features_test = []
# hog_features_test = []
# labels_test = []
#
# for i in range(len(test_generator)):
#     x_batch, y_batch = test_generator[i]
#
#     for img, label in zip(x_batch, y_batch):
#         vgg16_feat = extract_vgg16_features(img)
#         hog_feat = extract_hog_features(img)
#         vgg16_features_test.append(vgg16_feat)
#         hog_features_test.append(hog_feat)
#         labels_test.append(label)
#
#     if i == len(test_generator) - 1:
#         break



# Assuming you have already defined your test_datagen, test_dir, img_size, and batch_size

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='sparse',  # 'sparse' for integer labels, 'categorical' for one-hot encoded
# )


def generate_region_proposals(edges):
    # Find contours from the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Generate bounding boxes for each contour
    proposals = [cv2.boundingRect(c) for c in contours]
    # Filter out proposals with area less than 200px
    filtered_proposals = [box for box in proposals if box[2] * box[3] >= 5000]
    print(len(filtered_proposals))
    return filtered_proposals


def preprocess_region(image, box):
    # Unpack the bounding box
    (x, y, w, h) = box

    # Add padding to each side (10 pixels more on each side)
    padding = 0
    x_new = max(x - padding, 0)  # Ensure x is not less than 0
    y_new = max(y - padding, 0)  # Ensure y is not less than 0
    w_new = min(x + w + padding, image.shape[1])  # Ensure the width doesn't exceed image size
    h_new = min(y + h + padding, image.shape[0])  # Ensure the height doesn't exceed image size

    # Extract the region of interest (ROI) with the new bounding box
    roi = image[y_new:h_new, x_new:w_new]

    return roi
def apply_canny(image):
    cv2.imwrite("/root/face_project/PhaseThree/img.jpg",image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)

    edges = cv2.Canny(gray, 100, 200)

    return edges

vgg16_features_test = []
hog_features_test = []
labels_test = []
# print(len(test_generator))

def load_data(faces_dir, non_faces_dir):
    """
    Load the data from directories, extract CNN and HOG features, and prepare the dataset for training.

    Args:
    - faces_dir: Directory containing face images.
    - non_faces_dir: Directory containing non-face images.

    Returns:
    - X: Combined CNN + HOG feature array.
    - y: Label array (0 for non-face, 1 for face).
    """
    X = []
    y = []

    # Load images from the faces directory (label = 1)
    for filename in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, filename)
        image_data = cv2.imread(img_path)
        if image_data is not None:
            # cnn_features = get_vgg16_features(image_data)
            # hog_features = extract_hog_features(image_data)
            # combined_features = combine_features(cnn_features, hog_features)
            X.append(image_data)
            y.append(1)  # Label for faces

    # Load images from the non-faces directory (label = 0)
    for filename in os.listdir(non_faces_dir):
        img_path = os.path.join(non_faces_dir, filename)
        image_data = cv2.imread(img_path)
        if image_data is not None:
            # cnn_features = get_vgg16_features(image_data)
            # hog_features = extract_hog_features(image_data)
            # combined_features = combine_features(cnn_features, hog_features)
            X.append(image_data)
            y.append(0)  # Label for non-faces

    # X = np.array(X)
    # y = np.array(y)

    return X, y




for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]

    for img, label in zip(x_batch, y_batch):
        # Convert the image to grayscale for Canny edge detection
       # img_gray = np.uint8(img)

        edges = apply_canny(img)
        proposals = generate_region_proposals(edges)

        print(proposals)
        # Filter out proposals with area less than 200px
        # filtered_proposals = [box for box in proposals if box[2] * box[3] >= 5000]
        # Filter out proposals with area less than 200px
        # filtered_proposals = [box for box in proposals if box[2] * box[3] >= 1000]
        # Process each contour to extract features
        # for contour in contours:
            # Create a mask for the current contour
        for box in proposals:
            print(label)

            roi = preprocess_region(img, box)
            # Extract features from the masked image
            vgg16_feat = extract_vgg16_features(roi)
            hog_feat = extract_hog_features(roi)

            # Append features and labels to the lists
            vgg16_features_test.append(vgg16_feat)
            hog_features_test.append(hog_feat)
            labels_test.append(label)

    if i == len(test_generator) - 1:
        print(i,"-----------------------------------------------------------------------------------")
        break

# After processing, you can close any open OpenCV windows
# cv2.destroyAllWindows()


for run in range(1,6):
    # End training time
    end_training_time = time.time()
    training_time = end_training_time - start_training_time

    # Measure testing time
    start_testing_time = time.time()


    print(vgg16_features_test)
    print(hog_features_test)
    # Combine VGG16 and HOG features for testing
    combined_features_test = [np.concatenate((vgg, hog)) for vgg, hog in zip(vgg16_features_test, hog_features_test)]

    # Predict on the test data
    y_true_test = labels_test

    combined_features_test = np.array(combined_features_test)
    print(combined_features_test)
    y_pred_test = elm_classifier.predict(scaler.transform(combined_features_test))

    # End testing time
    end_testing_time = time.time()
    testing_time = end_testing_time - start_testing_time

    # Calculate metrics for training
    y_true_train = labels_train
    #y_pred_train = elm_classifier.predict(scaler.transform(combined_features_train))

    # Assuming elm_classifier.predict returns continuous values, we apply the function above
    y_pred_train = elm_classifier.predict(scaler.transform(combined_features_train))
    y_pred_test = elm_classifier.predict(scaler.transform(combined_features_test))

    # Convert continuous predictions to class labels
    y_pred_train = convert_continuous_to_class_labels(y_pred_train,
                                                            is_binary=False)  # Set to False for multi-class
    y_pred_test = convert_continuous_to_class_labels(y_pred_test, is_binary=False)  # Set to False for multi-class

    # Calculate training metrics
    accuracy_train, precision_train, recall_train, f1_train,conf_matrix = calculate_metrics(y_true_train, y_pred_train)

    # Calculate testing metrics
    accuracy_test, precision_test, recall_test, f1_test,conf_matrix1 = calculate_metrics(y_true_test, y_pred_test)





    # Print metrics for training and testing
    print("Training Metrics:")
    print(f"Accuracy: {accuracy_train:.4f}")
    print(f"Precision: {precision_train:.4f}")
    print(f"Recall: {recall_train:.4f}")
    print(f"F1-Score: {f1_train:.4f}")

    print("\nTesting Metrics:")
    print(f"Accuracy: {accuracy_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1-Score: {f1_test:.4f}")

    # Print the time taken for training and testing
    print(f"\nTraining Time: {training_time:.2f} seconds")
    print(f"Testing Time: {testing_time:.2f} seconds")

    y_true_train = np.array(labels_train, dtype=np.float32)  # Cast to float32 for compatibility
    y_pred_train = np.array(y_pred_train, dtype=np.float32)  # Ensure predictions are float32

    # Ensure that y_pred_train is between 0 and 1
    y_pred_train = np.clip(y_pred_train, 0.0, 1.0)  # This ensures the probabilities are valid



    y_true_test = np.array(labels_test, dtype=np.float32)  # Cast to float32 for compatibility
    y_pred_test = np.array(y_pred_test, dtype=np.float32)  # Ensure predictions are float32

    # Ensure that y_pred_train is between 0 and 1
    y_pred_test = np.clip(y_pred_test, 0.0, 1.0)  # This ensures the probabilities are valid


    loss_train = tf.keras.losses.binary_crossentropy(y_true_train, y_pred_train)
    loss_test = tf.keras.losses.binary_crossentropy(y_true_test, y_pred_test)





    # Take the mean of the loss over all samples in the batch
    loss_train = tf.reduce_mean(loss_train).numpy()
    loss_test = tf.reduce_mean(loss_test).numpy()

    resultsTrain = {
        'Run-test': run,
        'Accuracy': accuracy_train,
        'Precision': precision_train,
        'Recall': recall_train,
        'F1 Score': f1_train,
        'Loss': loss_train,
        'Traning Time (s)': training_time,
        'Confusion Matrix': [conf_matrix]  # Storing as a list to keep it in a cell
    }
    all_results.append(resultsTrain)
    resultsTest = {
        'Run-test': run,
        'Accuracy': accuracy_test,
        'Precision': precision_test,
        'Recall': recall_test,
        'F1 Score': f1_test,
        'Loss': loss_test,
        'Testing Time (s)': testing_time,
        'Confusion Matrix': [conf_matrix1]  # Storing as a list to keep it in a cell
    }
    all_results_test.append(resultsTrain)




excel_path_final_Train = f'ModelELM/testing_metrics_ELM_5_runs_train_phase2.xlsx'
excel_path_final_Test = f'ModelELM/testing_metrics_ELM_5_runs_test_phase2.xlsx'

df_results = pd.DataFrame(all_results)

df_results.to_excel(excel_path_final_Train, index=False)


df_results_test = pd.DataFrame(all_results_test)

df_results_test.to_excel(excel_path_final_Test, index=False)
