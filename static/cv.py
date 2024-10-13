import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
def load_data(data_dir):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    for label in ["Fire", "NoFire"]:
        folder_path = os.path.join(data_dir, label)
        if not os.path.exists(folder_path):
            continue
        
        for img_name in os.listdir(folder_path):
            if img_name.startswith('.') or not img_name.lower().endswith(valid_extensions):
                continue
            
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                image = image.astype('float32') / 255.0
                images.append(image)
                labels.append(0 if label == "NoFire" else 1)
    
    return np.array(images), np.array(labels)

# Load data
images, labels = load_data("Fire-Detection")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create the model
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model using the data generator
model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_test, y_test),
          epochs=10)

# Save the trained model in the new Keras format
model.save("fire_detection_model.keras")
print("Model saved as 'fire_detection_model.keras'")

# Define the function to detect fire in multiple images and display results
def detect_fire_and_show_images(images_list):
    results = {}
    
    # Create a subplot with enough space for all images
    num_images = len(images_list)
    plt.figure(figsize=(15, 5 * num_images))  # Adjust size as needed

    for i, image_path in enumerate(images_list):
        if not os.path.isfile(image_path):
            results[image_path] = "Image file not found."
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            results[image_path] = "Failed to read the image."
            continue
        
        img_resized = cv2.resize(img, (128, 128)) / 255.0  # Resize and normalize
        img_expanded = np.expand_dims(img_resized, axis=0)  # Expand dimensions to fit model input
        prediction = model.predict(img_expanded)  # Predict using the model
        
        # Store the result in a dictionary
        result_label = "Fire Detected" if prediction[0][0] > 0.5 else "No Fire Detected"
        results[image_path] = result_label
        
        # Display the image and result
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(img_resized)
        plt.title(f"{result_label} ({image_path})")
        plt.axis('off')  # Turn off axis

    plt.tight_layout()  # Adjust spacing
    plt.show()  # Show the plot

    return results


# Example usage for fire detection on multiple images
image_paths = ["NOFIRE.jpg","FIRE.jpg"]  # Update this with your correct image paths
print("Checking images...")
results = detect_fire_and_show_images(image_paths)

# Print results
for img_path, result in results.items():
    print(f"Result for {img_path}: {result}")

