# 🌿 Orbit-IQ — Plant Disease Detection with RAG Advisory

An AI-powered tool built with PyTorch and Flask to identify plant diseases from leaf images. After CNN-based classification, a **RAG (Retrieval-Augmented Generation) advisory layer** retrieves relevant knowledge from agricultural extension sources and uses the **Gemini API** to generate grounded, source-cited treatment advice.



---

## ✨ Features

* **AI-Powered Detection**: Utilizes a Convolutional Neural Network (CNN) to classify plant leaf images into **39 different categories** of diseases and healthy plants.
* **Detailed Diagnosis**: After prediction, the app displays the disease name, a detailed description, and an example image.
* **🆕 RAG Advisory Layer**: Retrieves the most relevant passages from a curated knowledge base of agricultural extension documents using **ChromaDB** vector search, then uses the **Gemini API** (via **LangChain**) to generate grounded, actionable advice with cited sources.
* **Actionable Advice**: Provides clear, actionable steps to prevent and treat the identified disease.
* **Supplement Market**: Includes a marketplace page suggesting relevant supplements for plant health, complete with purchase links.
* **User-Friendly Interface**: A clean and simple web interface for uploading images and viewing results.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Upload                              │
│                     (Plant Leaf Image)                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              CNN Model (PyTorch, 39 classes)                     │
│           Predicts disease class index (0-38)                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌────────────────────────────────────────┐
│  disease_info.csv │    │        RAG Advisory Pipeline           │
│  (Basic info)     │    │                                        │
└──────────────────┘    │  1. Query ChromaDB (sentence-transformers)│
                        │  2. Retrieve top-5 passages              │
                        │  3. LangChain RetrievalQA chain          │
                        │  4. Gemini API generates grounded advice │
                        │  5. Return advice + source citations     │
                        └────────────────────────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Submit Results Page   │
                        │  • Disease description   │
                        │  • Prevention steps      │
                        │  • 🤖 AI Advisory        │
                        │  • 📚 Cited sources      │
                        │  • Supplement links       │
                        └────────────────────────┘
```

---

## 💻 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask (Python) |
| **Deep Learning** | PyTorch (CNN) |
| **Vector Database** | ChromaDB (persistent, serverless) |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **LLM** | Google Gemini API (`gemini-2.0-flash`) |
| **Orchestration** | LangChain (`RetrievalQA` chain) |
| **Frontend** | HTML, CSS, Bootstrap 5 |
| **Core Libraries** | Pandas, NumPy, Pillow |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### 🔧 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DenCT101/Orbit-IQ.git
    cd Orbit-IQ/OrbitIQ-DTI-
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your API key:**
    ```bash
    cp .env.example .env
    # Edit .env and add your Google Gemini API key
    ```

5.  **Download the Pre-trained Model:**
    The pre-trained model file `plant_disease_model_1_latest.pt` is required. Make sure it is present in the root directory of the project.

6.  **Ingest the knowledge base into ChromaDB:**
    ```bash
    python rag_ingest.py
    ```

7.  **(Optional) Run the RAG evaluation:**
    ```bash
    python rag_evaluate.py
    ```

8.  **Start the Flask app:**
    ```bash
    python app.py
    ```

9.  **Open your browser** and navigate to `http://127.0.0.1:5000`.

---

## 📖 How to Use

1.  Navigate to the **AI Engine** page from the navigation bar.
2.  Click on the **"Choose File"** button to upload an image of a plant leaf.
3.  Press the **"Predict"** button.
4.  The application will display:
    - **Disease identification** and description
    - **Prevention steps** from the CSV database
    - **🤖 AI-Powered Advisory** — RAG-generated advice grounded in agricultural extension sources, with citations
    - **Supplement recommendations** with purchase links

---

## 🔬 RAG Advisory Layer — Details

### Knowledge Base
- **39 curated documents** covering all disease classes, sourced from university agricultural extension publications (UMN Extension, Penn State, Cornell, UC Davis, UF IFAS, etc.)
- Each document contains symptoms, disease cycle, treatment options, and prevention strategies
- Total of **200+ chunks** indexed in ChromaDB

### Ingestion Pipeline (`rag_ingest.py`)
- Reads `.txt` files from `knowledge_base/` directory
- Chunks documents into ~800-character segments with 150-character overlap
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores in **ChromaDB** (persistent, serverless — no separate database needed)

### Query Pipeline (`rag_query.py`)
- Uses **LangChain** `RetrievalQA` chain for the retrieval-generation flow
- After CNN predicts a class, retrieves **top-5 most relevant passages**
- **Gemini API** (`gemini-2.0-flash`) generates advisory text grounded in retrieved passages only
- Returns advice with source citations (university extension documents)

### Evaluation (`rag_evaluate.py`)
- **20 hand-crafted test questions** spanning diverse disease classes
- Measures **hit rate**: whether retrieved passages match the expected disease class
- Results saved to `rag_evaluation_results.json`

---

## 🧠 Model Architecture

The core of this project is a Convolutional Neural Network (CNN) built using PyTorch. The architecture consists of:
* **Four Convolutional Blocks**: Each block contains `Conv2d` layers, `ReLU` activation, and `BatchNorm2d` for normalization, followed by a `MaxPool2d` layer to downsample the feature maps. The channel sizes progressively increase (32 -> 64 -> 128 -> 256).
* **Fully Connected Layers**: After flattening the output from the convolutional layers, the data passes through dense layers with `Dropout` for regularization to prevent overfitting.
* **Output Layer**: The final layer outputs logits for the 39 possible classes.

The model was trained on the public **PlantVillage dataset**.

---

## 📁 Project Structure

```
OrbitIQ-DTI-/
├── app.py                      # Flask application (main entry point)
├── CNN.py                      # CNN model architecture
├── rag_ingest.py               # ChromaDB ingestion script
├── rag_query.py                # RAG query module (LangChain + Gemini)
├── rag_evaluate.py             # Retrieval evaluation (20 test questions)
├── disease_info.csv            # Disease descriptions & prevention steps
├── supplement_info.csv         # Supplement recommendations
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── knowledge_base/             # 📚 Source documents for RAG
│   ├── apple_scab.txt
│   ├── apple_black_rot.txt
│   ├── ...                     # (39 documents, one per class)
│   └── tomato_healthy.txt
├── chroma_db/                  # ChromaDB persistent storage (generated)
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── home.html
│   ├── index.html
│   ├── submit.html             # Results page with RAG advisory
│   ├── market.html
│   └── contact-us.html
└── static/                     # Static assets
    └── uploads/                # Uploaded images (generated)
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
