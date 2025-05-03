import streamlit as st
import pandas as pd
from utils import load_slang_dictionary
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from model import train_model, evaluate_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data = pd.read_csv('data/Hasil_Labelling.csv', sep=';', on_bad_lines='skip')
slang_dict = load_slang_dictionary('data/slang.txt')

X = data['cleaned']
y = data['Sentiment']

model, X_test_tfidf, y_test, tfidf = train_model(X, y)
accuracy, precision, recall, f1_score, report, cm = evaluate_model(model, X_test_tfidf, y_test)

def show():
    
    tab1, tab2 = st.tabs(["Prediksi Sentimen", "Hasil Kinerja Model"])
    
    with tab1:
        st.title("🔮 Prediksi Sentimen")
        
        st.markdown("""
        Masukkan teks di bawah ini untuk memprediksi sentimen.
        Model kami akan menganalisis dan memberikan prediksi apakah teks tersebut
        termasuk dalam kategori **Positif**, **Negatif**, atau **Netral**.
        """)
        
        text_input = st.text_area("Masukkan teks yang ingin dianalisis:")
        
        if st.button("Prediksi"):
            if not text_input.strip():
                st.warning("Masukkan teks terlebih dahulu")
            else:
                with st.spinner('Menganalisis sentimen...'):
                    try:
                        preprocessed_text = preprocess_text(text_input, slang_dict)
                        st.write('Kata setelah preprocessing:', preprocessed_text)
                        
                        input_tfidf = tfidf.transform([preprocessed_text])
                        prediction = str(model.predict(input_tfidf)[0])
                        
                        result_container = st.container()

                        if prediction == "Positif":
                            result_container.success(f"**Hasil Prediksi :** **{prediction}** 😊")
                        elif prediction == "Negatif":
                            result_container.error(f"**Hasil Prediksi :** **{prediction}** 😢")
                        else:
                            result_container.warning(f"**Hasil Prediksi :** **{prediction}** 😐")
                    except Exception as e:
                        st.error(f"Prediksi gagal: {str(e)}")
                        
    with tab2:
        st.title("Hasil Kinerja Model")
        st.write(f"**Akurasi:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1-Score:** {f1_score:.4f}")
                    
show()