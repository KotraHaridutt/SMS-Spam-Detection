# SMS Spam Detection 🚀  
This project aims to build an SMS spam detection system using **Natural Language Processing (NLP)** and **machine learning** techniques. The goal is to classify SMS messages as either **spam** or **ham (not spam)** using different classifiers, including **Naïve Bayes algorithms**.

## 📂 Dataset  
The dataset used in this project is **sms-spam.csv**, which contains two columns:
- `v1`: This column represents the label, either **spam** or **ham**.
- `v2`: This column contains the actual SMS message content.  

The dataset is loaded from Google Drive into a Pandas DataFrame for preprocessing and analysis.

## ⚙️ **Technologies Used**  
The project is implemented using **Python** and various data science libraries.  
- **Pandas** and **NumPy** are used for handling and processing the dataset.  
- **Matplotlib** and **Seaborn** help in visualizing data distributions.  
- **Scikit-learn** is used for feature extraction, model training, and evaluation.

## 📊 **Project Workflow**  
### 1️⃣ Data Preprocessing  
Before training the model, the dataset is cleaned and preprocessed. The labels (`spam` and `ham`) are converted into numerical values to facilitate training. The SMS text is then processed by converting all words to **lowercase**, removing special characters, **tokenizing** the text, eliminating **stopwords**, and applying **lemmatization**.  

### 2️⃣ Feature Extraction  
To convert the text into numerical format, the project uses **TF-IDF Vectorization** and **CountVectorizer**. These techniques help transform SMS messages into a format that machine learning models can understand.  

### 3️⃣ Model Training  
The project implements multiple **Naïve Bayes classifiers** to determine the best model for spam detection. The models used include:  
- **Multinomial Naïve Bayes**, which is particularly effective for text classification.  
- **Gaussian Naïve Bayes**, which assumes a normal distribution of data.  
- **Bernoulli Naïve Bayes**, which is useful when dealing with binary feature vectors.  

The dataset is split into a **training set (80%)** and a **test set (20%)** to evaluate the models effectively.

### 4️⃣ Model Evaluation  
To assess the performance of the models, several evaluation metrics are used, including:  
- **Accuracy Score**, which measures the overall correctness of predictions.  
- **Precision Score**, which evaluates how many of the messages predicted as spam were actually spam.  
- **Confusion Matrix**, which visually represents true positives, false positives, true negatives, and false negatives.  

The **Multinomial Naïve Bayes model** produced the best results, demonstrating high accuracy and precision in identifying spam messages.

## 📈 Results  
The results show that **Naïve Bayes classifiers are highly effective** for SMS spam detection. Among the three models tested, **Multinomial Naïve Bayes** achieved the best accuracy and precision. The confusion matrix and other evaluation metrics confirmed its strong performance in distinguishing between spam and ham messages.

## 🌍 Web Application for Spam Detection  
To make the spam detection model accessible, a **web application** has been developed where users can enter a message and get a prediction.  

### How It Works:
- The web app takes user input (an SMS message).  
- The input text is preprocessed using the same NLP techniques as the training data.  
- The trained **Multinomial Naïve Bayes model** makes a prediction.  
- The app displays whether the message is **Spam** or **Not Spam**.

## 🔥 How to Run the Project  
To run this project on your local system or Google Colab, follow these steps:  

1. **Clone the repository** using the following command:  
   ```sh
   git clone https://github.com/yourusername/SMS-Spam-Detection.git

👨‍💻 Author: KOTRA HARIDUTT
📧 Contact: kotraharidutt@gmail.com
