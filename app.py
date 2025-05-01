import streamlit as st
import pandas as pd

beranda = st.Page("pages/beranda.py", title="🏠 Beranda")
visualiasi = st.Page("pages/visualisasi.py", title="📈 Visualisasi")
prediksi = st.Page("pages/prediksi.py", title="🔮 Prediksi")

pg = st.navigation([beranda, visualiasi, prediksi])
st.set_page_config(page_title="📊 Analisis Sentimen")
pg.run()