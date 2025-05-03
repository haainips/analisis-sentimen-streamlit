import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import time

data = pd.read_csv('data/Hasil_Labelling.csv', sep=';', on_bad_lines='skip')

st.title("📊 Dashboard Analisis Sentimen")
st.markdown("""
Selamat datang di Dashboard Analisis Sentimen! Dashboard ini dirancang untuk membantu Anda:
- Menganalisis sentimen dari data teks
- Memvisualisasikan distribusi sentimen
- Melihat kata-kata yang sering muncul
- Memprediksi sentimen teks baru
""")

# Ambil metrik
total_data = len(data)
positive_count = len(data[data['Sentiment'] == 'Positif'])
negative_count = len(data[data['Sentiment'] == 'Negatif'])
neutral_count = len(data[data['Sentiment'] == 'Netral'])

positive_percentage = (positive_count / total_data) * 100
negative_percentage = (negative_count / total_data) * 100
neutral_percentage = (neutral_count / total_data) * 100

# Tampilkan metrik
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Data", f"{total_data:,}")

with col2:
    st.metric("Sentimen Positif", 
            f"{positive_count:,}", 
            f"{positive_percentage:.1f}%")

with col3:
    st.metric("Sentimen Negatif", 
            f"{negative_count:,}", 
            f"{negative_percentage:.1f}%")
    
with col4:
    st.metric("Sentimen Negatif", 
            f"{neutral_count:,}", 
            f"{neutral_percentage:.1f}%")

st.info("Dataset")

st.markdown("---")
    
    # Visualisasi pie chart
st.subheader("Distribusi Sentimen")
sample_data = pd.DataFrame({
    'Sentimen': ['Positif', 'Negatif', 'Netral'],
    'Jumlah': [positive_percentage, negative_percentage, neutral_percentage]
})
    
fig = px.pie(sample_data, values='Jumlah', names='Sentimen', 
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.3)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)
