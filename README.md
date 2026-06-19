<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Sarvam%20%7C%20Groq-purple" alt="AI">
  <img src="https://img.shields.io/badge/ML-SciKit%20Learn-yellow" alt="ML">
</div>

# 🧬 Healthcare AI Suite : A Next-Gen Clinical Intelligence Platform

**🏆 Built for Hackathons. Engineered to Save Lives.**

## 🚨 The Global Healthcare Crisis (The Problem)
The modern healthcare system is facing a breaking point:
- **Severe Doctor Shortages:** The WHO estimates a projected shortfall of 10 million health workers by 2030. People living in remote or underfunded areas lack basic access to primary diagnostics.
- **Skyrocketing Medical Costs:** Routine consultations and preventative screenings are becoming increasingly unaffordable, leading to delayed treatments and fatal complications.
- **The "Illegible Prescription" Epidemic:** Countless deaths and adverse drug events occur annually because pharmacists and patients cannot decipher messy doctor handwriting. The issue is so severe that high courts globally have had to repeatedly intervene, mandating legible prescriptions.
- **Fragmented Data:** Patients receive paper reports but have no way to track their longitudinal health trajectory, meaning they often miss the early warning signs of chronic diseases.

## 💡 Our Solution
An advanced, multi-modal Healthcare AI application designed to democratize clinical intelligence. Built for scale, this suite combines highly optimized Machine Learning (ML) predictive models, Vision AI, and Large Language Models (LLMs) to put a digital diagnostic assistant in every pocket.

---

## 🌟 Why This Project Wins (The "Wow" Factor)

- **Robust Machine Learning Pipelines:** We don't just rely on API wrappers. The core of our diagnostic engine is powered by local, rigorously trained ML models (using SciKit-Learn and Joblib) for multi-disease prediction. 
- **Time-Series Medical Analytics:** We don't just read a single report. Users can upload years of medical history, and our Vision AI automatically extracts temporal data to plot interactive health trajectory charts.
- **Dual-Engine AI Architecture:** Implements a robust `Sarvam -> Groq` fallback protocol. If the primary Vision/NLP engine faces rate limits, it seamlessly fails over to Groq's ultra-fast Llama 3 models, ensuring 100% uptime in critical healthcare scenarios.

---

## 🚀 Core Features

### 1. 🧬 Multi-Disease ML Predictor (Core Engine)
Integrated Machine Learning models capable of predicting the likelihood of various diseases (Diabetes, Heart Disease, PCOS, Parkinson's, etc.) based on user-inputted physiological metrics. These models run instantaneously and act as a critical first-line screening tool for patients who cannot afford immediate doctor consultations.

### 2. 📝 RxPlain Intelligence (Prescription Decoder)
Directly tackling the illegible handwriting crisis. Users simply upload a photo of a messy prescription. The dual-engine Vision AI extracts the text, identifies medications, flags potential warnings, and translates the clinical jargon into a simplified, native language summary. 

### 3. 📈 Longitudinal Health Trajectory (Time-Series)
Upload multiple past lab reports (e.g., Lipid Profiles, Complete Blood Counts). The Vision AI automatically extracts collection dates and numerical parameters, compiling them into a Pandas DataFrame to plot interactive trajectory graphs. The LLM then acts as a digital doctor, generating a plain-language summary of how your health is trending over time.

### 4. 🤖 AI Health Assistant with Vision
A conversational medical assistant. Not only can it answer general health queries using Groq's LLaMA 3, but it also has "eyes". Users can upload images of reports directly into the chat for contextual, real-time analysis.

---

## 🏗️ Technical Architecture

- **Machine Learning**: Scikit-Learn, NumPy, Pandas, Joblib (for serialized model deployment).
- **Primary AI (India-Native)**: Sarvam AI (`sarvam-m` for Vision, `sarvam-105b` for NLP).
- **Secondary AI (Ultra-Fast Fallback)**: Groq API (`llama-3.3-70b-versatile` & `llama-3.2-90b-vision-preview`).
- **Frontend & Routing**: Streamlit.
- **Data Visualization**: Matplotlib.

---

## 💻 Local Setup & Installation

Follow these steps to run the application locally on your machine.

### 1. Clone & Environment
```bash
git clone <your-repo-url>
cd 77
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
The application uses AI APIs. Export them in your terminal or create a `.env` file:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export SARVAM_API_KEY="your_sarvam_api_key_here"
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🛡️ Disclaimer
*This application is built for educational and demonstration purposes. The ML predictions and AI analyses are not substitutes for professional medical advice, diagnosis, or treatment.*
