# 🩺 Pneumonia Detection from Chest X-Rays using Robotic Process Automation (RPA) 

An advanced deep learning pipeline designed to **automate the diagnosis of pneumonia** from chest X-ray images using state-of-the-art computer vision techniques.

---

## 🚀 Project Highlights

This project explores multiple deep learning architectures and optimization strategies for accurate medical image classification, including:

- ✅ Multi-model comparison: `CNN`, `VGG19`, `CheXNet`, and `YOLO`
- 🛠️ Transfer learning with pre-trained models
- 📊 Data preprocessing, augmentation, and imbalance handling
- 💾 Model weight saving & loading for efficient retraining
- 🔍 Exploratory data analysis and visual insights

---

## 🧠 Technical Stack

- **Programming Language:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, PyTorch, OpenCV, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Models Used:** Custom CNN, VGG19, CheXNet, YOLOv3

---

## 📂 Dataset Overview

- Chest X-ray image dataset with labeled samples:
  - `Pneumonia` vs `Normal`
- Images preprocessed and resized to a uniform shape
- Missing values handled using imputation strategies
- Balanced dataset via **data augmentation**

---

## 🔍 Exploratory Data Analysis (EDA)

- Visualized class distributions to understand imbalance
- Analyzed image quality, pixel intensity, and structure
- Identified data augmentation requirements

---

## 🧪 Augmentation Pipeline

Using Keras `ImageDataGenerator`:

```python
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
