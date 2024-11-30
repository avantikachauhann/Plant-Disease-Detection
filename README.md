

# Plant Disease Detection Using CNN and Fuzzy Logic

This project leverages Convolutional Neural Networks (CNNs) and Fuzzy Logic to develop an automated system for detecting plant diseases from leaf images. The model is designed to provide early and accurate identification of diseases, which can help farmers minimize crop losses and improve agricultural productivity.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Model Architecture](#model-architecture)
5. [Implementation Details](#implementation-details)
6. [Results](#results)
7. [Challenges Faced](#challenges-faced)
8. [Future Scope](#future-scope)
9. [How to Run the Project](#how-to-run-the-project)

---

## Introduction

Plant diseases are a significant threat to global agricultural productivity. Traditional methods of disease identification involve manual inspection, which is time-consuming, prone to error, and requires expert knowledge. This project provides a scalable and automated solution using CNNs for feature extraction and Fuzzy Logic for refining classification decisions.

**Key Features:**
- **Deep Learning:** A CNN model trained on 83,000+ images of plant leaves.
- **Fuzzy Logic:** Added to improve decision-making in uncertain cases.
- **Real-Time Predictions:** Capable of making fast and accurate classifications.

---

## Dataset

The dataset was sourced from Kaggle and contains:
- **Images:** Over 83,000 labeled images of healthy and diseased plant leaves.
- **Classes:** Multiple plant species and disease categories.

### Data Preprocessing:
1. Cleaning and Normalization: Rescaled pixel values to [0, 1].
2. Augmentation: Applied techniques like rotation, flipping, zooming, and shifting to enhance dataset diversity.
3. Resizing: All images were resized to 128x128 pixels for uniformity.

---

## Methodology

1. **Data Collection:** Kaggle dataset containing labeled leaf images.
2. **Data Preprocessing:** Cleaning, normalization, and augmentation.
3. **Model Training:**
   - Built a CNN model to classify plant diseases.
   - Integrated Fuzzy Logic to improve decision-making.
4. **Model Evaluation:** Used metrics like accuracy, precision, recall, and F1-score.
5. **Results Visualization:** Visualized predictions, confusion matrix, and training/validation performance.

---

## Model Architecture

The CNN architecture includes:
- **Input Layer:** Accepts 128x128 RGB images.
- **Convolutional Layers:** Extract features using filters (32, 64, 128, 256).
- **Pooling Layers:** Downsample feature maps to reduce spatial dimensions.
- **Flatten Layer:** Converts 2D feature maps into a 1D vector.
- **Dense Layers:** Fully connected layers for classification.
- **Output Layer:** Softmax activation for multi-class probability prediction.

---

## Implementation Details

### Libraries and Tools:
- **TensorFlow/Keras:** For building and training the CNN.
- **OpenCV:** For image processing.
- **SciKit-Fuzzy:** For implementing Fuzzy Logic.
- **NumPy, Pandas:** For data manipulation.
- **Matplotlib, Seaborn:** For visualizations.

### Fuzzy Logic:
- Used to interpret CNN confidence scores and classify diseases into severity levels (e.g., mild, moderate, severe).
- Enhanced the model's performance in ambiguous cases by refining classification thresholds.

---

## Results

The performance of the CNN-based model was evaluated on a test set of **17,572 images** across multiple plant species and diseases.

### **Overall Metrics**
- **Accuracy:** 93%  
- **Precision (Macro Average):** 94%  
- **Recall (Macro Average):** 94%  
- **F1-Score (Macro Average):** 93%

---

### **Class-Wise Performance**

| **Class**                                  | **Precision** | **Recall** | **F1-Score** | **Support** |
|--------------------------------------------|---------------|------------|--------------|-------------|
| Apple___Apple_scab                         | 97%           | 93%        | 95%          | 504         |
| Apple___Black_rot                          | 100%          | 87%        | 93%          | 497         |
| Apple___Cedar_apple_rust                   | 88%           | 97%        | 92%          | 440         |
| Apple___healthy                            | 97%           | 87%        | 92%          | 502         |
| Blueberry___healthy                        | 90%           | 97%        | 93%          | 454         |
| Cherry (sour)___Powdery_mildew             | 84%           | 99%        | 91%          | 421         |
| Cherry (sour)___healthy                    | 98%           | 97%        | 98%          | 456         |
| Corn (maize)___Cercospora_leaf_spot        | 89%           | 89%        | 89%          | 410         |
| Corn (maize)___Common_rust                 | 99%           | 99%        | 99%          | 477         |
| Corn (maize)___Northern_Leaf_Blight        | 93%           | 92%        | 93%          | 477         |
| Corn (maize)___healthy                     | 96%           | 100%       | 98%          | 465         |
| Grape___Black_rot                          | 83%           | 97%        | 90%          | 472         |
| Grape___Esca (Black Measles)               | 100%          | 82%        | 90%          | 480         |
| Grape___Leaf_blight (Isariopsis_Leaf_Spot) | 95%           | 99%        | 97%          | 430         |
| Grape___healthy                            | 99%           | 98%        | 98%          | 423         |
| Orange___Haunglongbing (Citrus greening)   | 94%           | 99%        | 96%          | 503         |
| Peach___Bacterial_spot                     | 96%           | 93%        | 94%          | 459         |
| Peach___healthy                            | 99%           | 96%        | 98%          | 432         |
| ...                                        | ...           | ...        | ...          | ...         |

> **Note:** Only a subset of the class-wise performance metrics is shown here for brevity. See the full confusion matrix for detailed insights.

---

### **Key Observations**
1. **High Accuracy for Most Classes**:  
   The model achieves near-perfect accuracy for diseases with distinct symptoms, such as *Corn (maize)___Common_rust* and *Tomato___Tomato_Yellow_Leaf_Curl_Virus*.
   
2. **Struggles with Similar Diseases**:  
   Some diseases, such as *Tomato___Early_blight* and *Tomato___Late_blight*, had overlapping symptoms, resulting in slightly lower precision and recall scores.
   
3. **Fuzzy Logic Contribution**:  
   Fuzzy logic improved predictions for borderline confidence scores, enhancing reliability for diseases with subtle differences.


**Visualizations:**
- Confusion matrix, accuracy/loss curves, and sample predictions are included in the results.

---

## Challenges Faced

1. **Similar Symptoms:** Differentiating between diseases with similar visual symptoms.
2. **Data Imbalance:** Some classes had significantly fewer samples, affecting performance.
3. **High-Resolution Images:** Computationally intensive processing of high-resolution images.

---

## Future Scope

1. **Expand Dataset:** Include more plant species and diseases.
2. **Environmental Data Integration:** Add factors like temperature and humidity to enhance predictions.
3. **Deploy Model:** Create a mobile app or web interface for real-time disease detection.
4. **Advanced Architectures:** Experiment with transfer learning models like ResNet or EfficientNet.

---

## How to Run the Project

### Prerequisites:
- Python 3.8
- TensorFlow
- OpenCV
- SciKit-Fuzzy
- NumPy, Pandas, Matplotlib, Seaborn

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/username/plant-disease-detection.git

   Here’s a well-structured **README.md** file for your plant disease detection project. It includes all the necessary sections to explain your project on GitHub effectively.

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    Prepare the dataset:
    Download the Kaggle dataset and place it in the dataset/ directory.
4. Train the model:
    ```bash
    python train_model.py
5. Test the model:
    ```bash
    python test_model.py
6. Visualize results:
    - View accuracy/loss plots and sample predictions in the output directory.


