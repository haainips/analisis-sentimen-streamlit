import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocessing import preprocess_text
from utils import load_slang_dictionary
from model import train_model, evaluate_model

from sklearn.feature_extraction.text import TfidfVectorizer

# Load Data
data = pd.read_csv('data/Hasil_Labelling.csv', sep=';', on_bad_lines='skip')
slang_dict = load_slang_dictionary('data/slang.txt')

# TF-IDF
tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True, stop_words='english')
X = tfidf.fit_transform(data['cleaned'])
y = data['Sentiment']

# Train Model
model, X_test, y_test = train_model(X, y)

# Evaluate
accuracy, precision, recall, f1_score, report, cm = evaluate_model(model, X_test, y_test)

# Streamlit App
st.title("Analisis Sentimen Pengguna MyBlueBird 🚖")
st.write(data[['cleaned','Score','Sentiment']].head(15))

# WordCloud
st.subheader("Wordcloud")
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(data['cleaned']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Sebelum SMOTE
fig_plotly = px.histogram(data, x='Sentiment', color='Sentiment', text_auto=True,
    title="Jumlah Label Positif dan Negatif", color_discrete_sequence=px.colors.qualitative.Pastel)
fig_plotly.update_layout(title_font_size=20)
st.plotly_chart(fig_plotly)


st.subheader("Hasil Evaluasi")
st.write(f"**Akurasi:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1-Score:** {f1_score:.4f}")

# Prediction Section
st.subheader("Prediksi Sentimen User")

text_input = st.text_area("Masukkan ulasan anda:")

if st.button("Prediksi"):
    preprocessed_text = preprocess_text(text_input, slang_dict)
    st.write('Kata setelah preprocessing :', preprocessed_text)
    input_tfidf = tfidf.transform([preprocessed_text])
    prediction = model.predict(input_tfidf)
    st.write(f"**Hasil Prediksi:** {prediction[0]}")
