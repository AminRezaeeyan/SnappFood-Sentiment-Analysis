# Snapfood-Sentiment-Analysis

## Overview
This project implements a comprehensive sentiment analysis system for Snapfood restaurant reviews using various machine learning approaches, from traditional algorithms to advanced deep learning models. The system is designed to accurately classify customer sentiments from Persian text reviews.

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/AminRezaeeyan/SnappFood-Sentiment-Analysis.git
cd SnappFood-Sentiment-Analysis
```

### Installation Options

#### Option 1: Using Docker (Recommended)
1. Make sure you have Docker and Docker Compose installed on your system
2. Build and run the container:
```bash
docker-compose up --build
```
The application will be available at `http://localhost:8501`

#### Option 2: Manual Installation
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app/app.py
```

## Technical Details

### Traditional Machine Learning Approaches
The project initially explored several traditional machine learning algorithms:
- **Logistic Regression**: Used as a baseline model for binary classification
- **Random Forest**: Implemented with optimized hyperparameters for better feature importance analysis
- **XGBoost**: Utilized for its superior performance in handling imbalanced datasets

### Deep Learning Models
The project implemented various neural network architectures:
- **RNN (Recurrent Neural Network)**: Basic implementation for sequence modeling
- **LSTM (Long Short-Term Memory)**: Used to capture long-term dependencies in review texts
- **GRU (Gated Recurrent Unit)**: Implemented as a more efficient alternative to LSTM

### Transformer-Based Approach
The final and most successful implementation uses:
- **ParsBERT**: A pre-trained Persian BERT model
  - Fine-tuned on the Snapfood review dataset
  - Achieved state-of-the-art results in sentiment classification
  - Optimized for Persian language understanding

## Demo
<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/AminRezaeeyan/SnappFood-Sentiment-Analysis/raw/main/images/1.png" alt="Demo 1" width="400"/>
      </td>
      <td align="center">
        <img src="https://github.com/AminRezaeeyan/SnappFood-Sentiment-Analysis/raw/main/images/2.png" alt="Demo 2" width="400"/>
      </td>
    </tr>
  </table>
</div>

## Results
The fine-tuned ParsBERT model achieved the best performance in sentiment classification, demonstrating superior accuracy in understanding Persian language nuances and context in restaurant reviews.