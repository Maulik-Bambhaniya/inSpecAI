
# ⚡ inSpecAI – AI-Powered Visual Inspection for QC & Assurance

**inSpecAI** is a real-time, AI-powered quality inspection tool for manufacturing environments. Built with Streamlit and TensorFlow, it classifies component images as defective or non-defective using a Teachable Machine-trained model. The UI is sleek, interactive, and modularized for clean development.

---

## 🚀 Features

- 🖼️ Upload or capture images via camera
- 🔍 Deep learning-based defect classification
- 📊 Plotly-based confidence and defect analytics
- 📋 Inspection history with export to CSV
- 🧠 Customizable model (`keras_model.h5`) and labels
- 🎨 Modular CSS styling (via `styles.css`)
- 📈 Real-time dashboard for inspections and pass rate

---

## 🛠 Tech Stack

- **Frontend**: Streamlit, HTML/CSS, Plotly
- **Backend**: TensorFlow / Keras
- **Image Processing**: PIL, NumPy
- **Data Handling**: Pandas, Session State
- **Language**: Python 3.8–3.10

---

## 📁 Project Structure

```
inSpecAI/
├── streamlit_app.py     # Main Streamlit app
├── keras_model.h5       # Exported model from Teachable Machine
├── labels.txt           # Class labels for predictions
├── styles.css           # Modular custom CSS
├── requirements.txt     # Python dependencies
└── README.md            # This documentation file
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/maulik-bambhaniya/inSpecAI.git
cd inSpecAI
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run streamlit_app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 📦 requirements.txt

```
streamlit
tensorflow
pillow
numpy
pandas
plotly
```

---

## 🧠 Using Your Own Model

1. Train a model using [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Export it as TensorFlow → download the `keras_model.h5` and `labels.txt`
3. Replace the files in this directory

---

## 🧪 How It Works

1. Upload or capture a component image
2. Click "🚀 INITIATE DEEP SCAN"
3. View:
   - Classification label
   - Confidence percentage
   - Pass/Fail status
4. Inspect the analytics & history tabs for deeper insights

---


---

## 📸 Screenshots

![Demo Screenshot](https://github.com/user-attachments/assets/b6c6c24d-9b51-483b-a638-1125232db9dd)



> Built as part of the Intel® AI for Manufacturing Certification Project
