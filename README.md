# Fake News Detection System using NLP

📌 Project Title

Fake News Detection System using Natural Language Processing (NLP), 
Machine Learning, and Deep Learning

Member 03 - CIT-24-01-0279 

---
<img width="1046" height="378" alt="image" src="https://github.com/user-attachments/assets/d46c0055-5de0-4b30-b8a8-e33d6e3c3d01" />

## Problem Statement

The rapid spread of fake news through websites and social media has become a significant challenge, making it difficult for users to distinguish between credible and misleading information. This project aims to develop an NLP-based fake news detection system that automatically classifies news articles as **Fake** or **Real** using machine learning and deep learning techniques. The application also supports website URL analysis by extracting article content, preprocessing the text, and predicting its authenticity. Additionally, if the input article is in Sinhala, it is translated into English before preprocessing due to the limited availability of Sinhala fake news datasets.

---

## Dataset Information

The project uses multiple publicly available datasets:

| Dataset | Purpose |
|----------|----------|
| ISOT Fake News Dataset | Main fake and real news articles |
| BuzzFeed Dataset | Additional fake and real news |
| PolitiFact Dataset | Verified fact-checking news |
| LIAR Dataset | Short political statements |

### Dataset Features

- Approximately 50,000+ news articles
- Binary classification
  - **0 → Fake**
  - **1 → Real**
- English news articles with support for translated Sinhala input

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <https://github.com/kinkini-cmd/NLP_Group_05>
cd NLP_Group_05
git checkout feature/cit-24-01-0279-LSTM
