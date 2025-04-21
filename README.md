**Pneumonia Detection from Chest X-Rays using RPA methology**
Automated Diagnosis System with CNN, VGG19, and CheXNet Transfer Learning
Project Overview
A deep learning pipeline for detecting pneumonia from chest X-ray images, implementing:
✔ Multi-model architecture comparison (CNN, VGG19, CheXNet, YOLO)
✔ Advanced hyperparameter optimization (learning rate, batch size, early stopping)
✔ Class imbalance mitigation through data augmentation and sampling
✔ Model persistence with weight saving/loading for iterative training

Technical Implementation
Data Preprocessing
Exploratory Data Analysis: Visualized class distributions and image quality

Missing Value Handling: Implemented robust data imputation strategies

Augmentation Pipeline:

python
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
Model Architectures
Model	Val Accuracy	Precision	Recall
Baseline CNN	82.3%	0.79	0.85
VGG19-FT	89.1%	0.87	0.91
CheXNet	91.4%	0.89	0.93
Optimization Strategies
Learning Rate Scheduling:

python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
Early Stopping:

python
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True
