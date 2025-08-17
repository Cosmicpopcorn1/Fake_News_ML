# 📰 Fake News Detection System
## 📌 Project Overview
This project focuses on detecting fake news articles using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system is built to classify news articles as fake or real based on their textual content.

By leveraging TF-IDF (Term Frequency–Inverse Document Frequency) for feature extraction and ML models such as Logistic Regression and Random Forest, the system achieved over 99.6% accuracy with strong F1-scores during robust cross-validation.

This project demonstrates skills in:

Text preprocessing & NLP techniques

Feature engineering using TF-IDF

Model training & evaluation with ML algorithms

Handling real-world binary classification problems

# ⚙️ Tech Stack
Programming Language: Python 3.x

Libraries:

pandas, numpy – data manipulation

scikit-learn – ML models, TF-IDF, evaluation metrics

nltk – tokenization, stopwords removal, text preprocessing

matplotlib, seaborn – visualization

# 📊 Dataset
The project uses a large labeled news dataset consisting of real and fake news articles.

Data columns: title, text, label (where label=1 for fake, label=0 for real)

Data size: Thousands of articles, balanced across both classes

Split:

Training set: 80%

Test set: 20%

# 🔎 Methodology
1. Text Preprocessing
To prepare raw text for model training, following preprocessing steps were applied:

Lowercasing

Removal of punctuation, numbers, and special characters

Stopword removal (using NLTK stopwords corpus)

Tokenization

Lemmatization

This ensures the text is cleaned and standardized before feature extraction.

2. Feature Engineering (TF-IDF)
Used TF-IDF Vectorizer from scikit-learn

Extracted numerical vector representations of text

Configured with:

ngram_range=(1,2) → Unigrams & bigrams

max_features=50,000+ → Large feature space for better representation

Final feature matrix used as input to ML models

3. Machine Learning Models
Two models were trained and evaluated:

🔹 Logistic Regression
A linear classifier suitable for high-dimensional text classification.

Performed exceptionally well due to the sparse TF-IDF features.

Fast training & inference.

🔹 Random Forest
Ensemble of decision trees using bagging.

Captures non-linear relationships.

Provided robust results but slightly more computationally expensive than Logistic Regression.

4. Model Evaluation
Metrics used:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Cross-validation (Stratified K-Fold)

# ✅ Results
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	99.6%	High	High	High
Random Forest	98.9%	High	High	High
Logistic Regression outperformed Random Forest in accuracy and generalization.

The system achieved 99.6% accuracy with F1-scores > 0.99, showing excellent balance between precision and recall.

Confusion matrix revealed very low misclassification rates.

📈 Visualization Examples
Distribution of labels (Fake vs Real)

Most frequent words in Fake vs Real news

Confusion matrix for both models

ROC-AUC curves

(Plots can be added in your repo under results/)

# 🚀 How to Run
1. Clone repo
bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
2. Install requirements
bash
pip install -r requirements.txt
3. Run training script
bash
python train.py
4. Run evaluation
bash
python evaluate.py
5. Predict on new input
python
from predict import predict_news
print(predict_news("Breaking: Scientists discover chocolate cures flu"))
🌟 Key Learnings
Effectiveness of TF-IDF vectors in representing text data.

Strength of Logistic Regression in large, sparse, high-dimensional feature spaces.

Importance of cross-validation and F1-score beyond only accuracy for reliable evaluation.

Practical application of NLP + ML in solving misinformation detection problems.

# 🔮 Future Improvements
Incorporate deep learning models (LSTMs, BERT, Transformers).

Use topic modeling (LDA) to understand themes in fake vs real news.

Deploy as a web app / API for real-time fake news detection.

Experiment with feature selection and dimensionality reduction for faster inference.

👨💻 Author
Developed by Sanidhya Mathur
📧 Contact: sanidhya.mathur5013@gmail.com
