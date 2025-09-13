# 🌿 Plant Disease Detection Web Application

An AI-powered tool built with PyTorch and Flask to identify plant diseases from leaf images. This application helps farmers and gardeners diagnose crop issues early, providing detailed information, prevention steps, and even supplement recommendations.



---

## ✨ Features

* **AI-Powered Detection**: Utilizes a Convolutional Neural Network (CNN) to classify plant leaf images into **39 different categories** of diseases and healthy plants.
* **Detailed Diagnosis**: After prediction, the app displays the disease name, a detailed description, and an example image.
* **Actionable Advice**: Provides clear, actionable steps to prevent and treat the identified disease.
* **Supplement Market**: Includes a marketplace page suggesting relevant supplements for plant health, complete with purchase links.
* **User-Friendly Interface**: A clean and simple web interface for uploading images and viewing results.

---

## 💻 Tech Stack

* **Backend**: **Flask** (Python)
* **Deep Learning Framework**: **PyTorch**
* **Frontend**: HTML, CSS, **Bootstrap 5**
* **Core Libraries**: Pandas, NumPy, Pillow

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.7+
* pip (Python package installer)

### 🔧 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DenCT101/Orbit-IQ.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Pre-trained Model:**
    The pre-trained model file `plant_disease_model_1_latest.pt` is required. Make sure it is present in the root directory of the project.
    *(Note: If the model is not in the repo, add instructions here on where to download it from, e.g., Google Drive, GitHub Releases, etc.)*

5.  **Run the Flask application:**
    ```bash
    python app.py
    ```

6.  **Open your browser** and navigate to `http://127.0.0.1:5000`.

---

## 📖 How to Use

1.  Navigate to the **AI Engine** page from the navigation bar.
2.  Click on the **"Choose File"** button to upload an image of a plant leaf.
3.  Press the **"Predict"** button.
4.  The application will process the image and display the prediction results, including the disease name, description, and recommended actions.

---

## 🧠 Model Architecture

The core of this project is a Convolutional Neural Network (CNN) built using PyTorch. The architecture consists of:
* **Four Convolutional Blocks**: Each block contains `Conv2d` layers, `ReLU` activation, and `BatchNorm2d` for normalization, followed by a `MaxPool2d` layer to downsample the feature maps. The channel sizes progressively increase (32 -> 64 -> 128 -> 256).
* **Fully Connected Layers**: After flattening the output from the convolutional layers, the data passes through dense layers with `Dropout` for regularization to prevent overfitting.
* **Output Layer**: The final layer outputs logits for the 39 possible classes.

The model was trained on the public **PlantVillage dataset**.

---

