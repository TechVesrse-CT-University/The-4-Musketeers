# 🧠 InVisionX - Deepfake Face Image Detection System

## 🚀 Project Overview

Deepfake images are becoming increasingly realistic and are widely used to spread misinformation and deceive people. Manual detection is nearly impossible, making it a growing digital threat.

**InVisionX** aims to tackle this issue using a **Convolutional Neural Network (CNN)**-based model. The system takes an image as input and provides a **confidence percentage** indicating how real or fake the image is.

With an intuitive interface and fast analysis, InVisionX empowers users to verify image authenticity and stay protected from visual manipulation.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Frontend Framework:** Bootstrap  
- **Backend:** Python  
- **Backend Framework:** Flask  
- **Other Tools:** Visual Studio Code, GitHub Copilot  

---

## 💻 GUI Screenshots

**Index Page**  
![Index Page](images/example1.png)

**After Image Recognition and Analysis**  
![Evaluation Page](images/example2.png)

---

## 📂 Project Structure and Code Explanation

### 📁 1. `index.py` - Model Training

#### 🔸 Data Management
- Loads datasets (training, validation, test) using `image_dataset_from_directory`.
- Resizes images to **128x128**, with a batch size of **32**.
- Applies optimizations like caching and prefetching.

#### 🔸 Data Augmentation
- Adds random flips and rotations to make the model more robust.

#### 🔸 CNN Architecture
- Rescaling (normalization)
- 3 Convolutional Layers (filters: 32, 64, 128)
- MaxPooling after each layer
- Flatten → Dense(128) with ReLU → Dropout(0.5) → Sigmoid Output (Binary Classification)

#### 🔸 Training
- Optimizer: **Adam**
- Loss: **Binary Crossentropy**
- Includes **EarlyStopping** and **ReduceLROnPlateau**
- Trains for up to **20 epochs**

#### 🔸 Evaluation
- Plots training/validation accuracy & loss
- Saves trained model as `face_classifier.keras`
- Visualizes predictions on test images

---

### 📁 2. `app.py` - Web Backend (Flask)

#### 🔸 Setup
- Loads the trained model
- CORS-enabled Flask app

#### 🔸 Image Preprocessing
- Converts to RGB
- Resizes to 128x128
- Normalizes and batches input

#### 🔸 Routes
- `/`: Serves the homepage
- `/predict`: Accepts an image, runs prediction, and returns:
  - **Prediction**: Real or Fake
  - **Confidence Level**

#### 🔸 Error Handling
- Checks for missing or invalid files
- Returns appropriate error messages

---

### 📁 3. `index.html` - Frontend UI

#### 🔸 Design
- Clean, modern look using Bootstrap
- Custom CSS for gradients and hover effects

#### 🔸 Features
- Drag-and-drop or browse image upload
- Display selected file
- Show prediction result with icon
- Highlights: Speed, Security, and AI-powered insights

#### 🔸 Functionality
- JavaScript handles async fetch requests to `/predict`
- Parses response and displays it dynamically
- Error handling for user input

---

## 🔍 Understanding Accuracy & Loss in InVisionX

During training, InVisionX tracks four key metrics:

### ✅ Accuracy
- **Training Accuracy**: How well the model fits the training data.
- **Validation Accuracy**: Key metric for generalization to unseen data.

### 📉 Loss
- **Training Loss**: Measures error on training data.
- **Validation Loss**: Indicates overfitting if it increases while training loss decreases.

### 🛡️ Optimization Techniques
- **EarlyStopping**: Stops training if validation loss doesn’t improve.
- **ModelCheckpoint**: Saves the best-performing model.

---

## 📊 Visual Results

**📈 Model Accuracy Output**  
![Accuracy of the Model](images/accuracyterminal.png)

**🧑‍🦰 Sample Faces & Prediction Output**  
![Test Faces and Results](images/Faces_1.png)

**📉 Accuracy & Loss Graphs**  
![Graph 1](images/Figure_1.png)

**📈 Model Evauluation Metrics**  <br>
![Graph 2](images/example3.png)

---

## 📊 Model Evaluation Metrics Explained

## 🔍 1. Support
🧮 **Support** refers to the number of actual samples for each class in the dataset.

- **Fake**: 120 examples labeled as Fake.
- **Real**: 135 examples labeled as Real.

It helps to understand how balanced the dataset is.

---

## 🧠 2. Precision
**Precision** tells us: _Out of all the images the model predicted as "Real", how many were actually real?_

### Formula:

- **TP** = True Positives (correctly predicted Real images)
- **FP** = False Positives (images incorrectly predicted as Real)

📌 **High precision** means **fewer false positives**.

Example:
- For "Real": If the model predicted 100 images as Real, and 59 of them were actually Real,  
  ➤ Precision = **59%**

---

## 🔍 3. Recall
**Recall** tells us: _Out of all the actual "Real" images, how many did the model correctly identify?_

### Formula:

- **TP** = True Positives  
- **FN** = False Negatives (Real images wrongly predicted as Fake)

📌 **High recall** means **fewer false negatives**.

Example:
- For "Real": If there were 100 actual Real images, and the model correctly identified 61 of them,  
  ➤ Recall = **61%**

---

## ⚖️ 4. F1-Score
The **F1-score** is the **harmonic mean** of precision and recall. It's a single metric that balances both.

### Formula:

- Useful when you care about both **precision and recall**, especially in imbalanced datasets.

📌 A **high F1-score** means the model is doing well on both avoiding false positives and false negatives.

---

### ✅ Summary Table (Example)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fake  | 0.54      | 0.53   | 0.53     | 120     |
| Real  | 0.59      | 0.61   | 0.60     | 135     |
| **Avg** | **0.57** | **0.57** | **0.57** | 255     |

---

## ✅ System Integration Summary

| Component | Role |
|----------|------|
| `index.py` | Trains the AI model |
| `app.py` | Hosts the prediction service |
| `index.html` | Enables user interaction |

ThankYou for giving us the Opportunity to learn and collaborate with most innovative minds of the Country.


