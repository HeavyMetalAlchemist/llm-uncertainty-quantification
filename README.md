# LLM Uncertainty Quantification

This project explores methods for **uncertainty quantification** in large language models (LLMs) using **Phi-3-Mini-4k-Instruct (4-bit)**.  
It analyzes both **token-level** and **sentence-level** uncertainty signals across two distinct tasks:
- Mathematical reasoning  
- Factual question answering  

The experiments investigate how uncertainty metrics such as entropy, confidence, calibration error, and AUROC vary between reasoning and factual tasks, providing insights into model reliability and calibration.

---

## 📘 Project Overview

Uncertainty quantification aims to measure how *confident* or *uncertain* a model is in its own predictions.  
This notebook evaluates:
- **Token-level metrics:** entropy, probability margins, chosen-token probability  
- **Sentence-level metrics:** mean confidence, mean entropy, variance of log-probabilities  
- **Calibration metrics:** Expected Calibration Error (ECE), Brier Score, AUROC, and Risk–Coverage curves  

By comparing reasoning with factual QA, this project highlights how uncertainty behaves differently across task types — structured reasoning vs. knowledge recall.

---

## 🧩 Repository Structure
```
llm-uncertainty-quantification/
│
├── LLM_Uncertainty_Quantification.ipynb # Main research notebook
├── requirements.txt # Python dependencies
├── visualization/ # Generated plots and figures
├── output_csv/ # Saved csv files for data exploration
├── LICENSE 
└── README.md
```
---

## ⚙️ Setup Instructions

1. **Create a Conda environment**  
   - ```conda create -n ENV_NAME python=3.10```  
   - ```conda activate ENV_NAME```
     
2. **Install Pytorch and CUDA support**
   - ```pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118```
2. **Install dependencies**
   - ```pip install -r requirements.txt```  
3. **Launch Jupyter Notebook**
    - ```jupyter notebook```  

Then open and run all cells in
LLM_Uncertainty_Quantification.ipynb

---

## 🧠 Model & Environment

- **Model:** `microsoft/Phi-3-mini-4k-instruct` (4-bit quantized)  
- **Hardware:** Tested locally on NVIDIA RTX 2070 (8 GB VRAM)  
- **Environment:** Python 3.10, Conda-based setup  
- Runs **fully offline** - no API keys or external services required  

---

## 📊 Key Findings

- **Reasoning:**  
  - Uncertainty strongly correlates with reasoning complexity. 
  - Abstention strategies (risk–coverage) effectively improve overall performance

- **Factual QA:**  
  - Confidence signals rank correctness well (AUROC ≈ 0.83) but calibration is poor
  - The model can be *highly confident yet wrong* due to missing facts (**confident ignorance**)

- **Cross-task insight:**  
  - **Reasoning tasks** benefit more from selective prediction.  
  - **Factual QA tasks** benefit more from calibration improvements.  
  - **Confidence** remains the simplest and most robust estimator overall, while **entropy** serves as a consistent secondary signal across both tasks.

---

