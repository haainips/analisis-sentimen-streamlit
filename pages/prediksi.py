import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_slang_dictionary
from preprocessing import preprocess_text
from model import train_model, evaluate_model

data = pd.read_csv('data/Hasil_Labelling3.csv', sep=';', on_bad_lines='skip')
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
        st.title("📊 Hasil Kinerja Model")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Akurasi", value=f"{accuracy:.2%}")
        with col2:
            st.metric(label="Precision", value=f"{precision:.2%}")
        with col3:
            st.metric(label="Recall", value=f"{recall:.2%}")
        with col4:
            st.metric(label="F1-Score", value=f"{f1_score:.2%}")
        
        st.markdown("---")
        
        # Confusion Matrix Section
        st.subheader("Confusion Matrix")
        
        # Option 1: Plotly (Interactive)
        st.checkbox("Tampilkan Confusion Matrix Interaktif", value=True)
        cm_labels = ['Negatif', 'Netral', 'Positif']
        fig = px.imshow(
            cm,
            x=cm_labels,
            y=cm_labels,
            color_continuous_scale='Blues',
            text_auto=True,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
        )
        st.plotly_chart(fig, use_container_width=True)

show()