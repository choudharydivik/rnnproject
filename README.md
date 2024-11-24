## End to End RNN Project Using RNN
### IMDB Movie Review Sentiment Analysis

This project aims to perform sentiment analysis on IMDB movie reviews using a Recurrent Neural Network (RNN). The goal is to predict whether a given movie review is **positive** or **negative** based on the movie reviews.

The project follows an end-to-end workflow from data preparation, model training, to deployment of the trained model as a web application.


### Steps Involved

#### 1. **Data Preparation**
- Loaded the **IMDB movie reviews dataset** available in Keras.
- Performed text preprocessing and tokenization to convert raw text data into numerical representations.
- Applied padding to ensure uniform input size for the RNN model.

#### 2. **Model Building**
- Built an **RNN model** using **Keras** and **TensorFlow**.
- Incorporated an **embedding layer** for learning word representations directly from the data.
- Compiled the model with suitable loss functions and optimizers for binary classification.

#### 3. **Model Training**
- Trained the RNN model on the IMDB dataset with a focus on achieving high classification accuracy.
- Used evaluation metrics to monitor model performance during training and validation.

#### 4. **Making Predictions**
- The trained RNN model is used to predict whether a review is positive or negative.
- Evaluated the model on unseen data to ensure generalization and performance.

#### 5. **Web Application Integration**
- Integrated the trained RNN model into a **Streamlit** web application.
- The app allows users to input movie reviews and get real-time predictions on whether the review is positive or negative.

#### 6. **Deployment**
- The Streamlit web application is deployed on **Streamlit Cloud** for easy access and usage.

---

### Key Highlights

- **End-to-End Workflow**: From data preparation to model deployment, all stages are covered.
- **RNN for Text Data**: Using RNNs to capture the sequential nature of text data for sentiment analysis.
- **Word Embeddings**: Leveraging word embeddings to represent words in a vector space and enhance model accuracy.
- **Real-Time Predictions**: Integrated the trained model with a user-friendly Streamlit app for quick sentiment analysis.
- **Scalable Deployment**: Deployed the app on **Streamlit Cloud** for public access and easy usage.

---

### Try the Application

You can try the application live here: https://imdb-movie-review-sentiment-analysis-rnn.streamlit.app/

---

### Technologies Used
- **Keras**: For building the RNN model.
- **TensorFlow**: For training the model.
- **Streamlit**: For building the web app.
- **Python**: Programming language used for the entire project.

---

This project demonstrates the application of **Recurrent Neural Networks (RNNs)** and **word embeddings** for sentiment analysis tasks. The integrated **Streamlit web app** provides an intuitive interface for users to interact with the model and make predictions.
