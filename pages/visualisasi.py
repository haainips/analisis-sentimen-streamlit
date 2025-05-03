import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

data = pd.read_csv('data/Hasil_Labelling.csv', sep=';', on_bad_lines='skip')
def visualization():
    st.title("📈 Visualisasi Sentimen")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi", "Trend Waktu", "Word Cloud", "Kata Umum"])
    
    with tab1:
        st.subheader("Distribusi Sentimen")
        fig1 = px.histogram(data, 
                x='Sentiment',
                color='Sentiment',
                color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        st.subheader("Trend Sentimen Over Time")
        
        try:
            # Pastikan kolom 'at' ada dan bisa dikonversi ke datetime
            if 'at' not in data.columns:
                st.warning("Kolom 'at' tidak ditemukan dalam data")
            else:
                # Coba konversi ke datetime
                data['at'] = pd.to_datetime(data['at'], errors='coerce')
                
                # Cek apakah konversi berhasil
                if data['at'].isnull().all():
                    st.error("Gagal mengkonversi kolom 'at' ke format datetime. Format tanggal tidak dikenali.")
                    st.write("Contoh data kolom 'at':", data['at'].head())
                else:
                    # Filter out rows with NaT (jika ada konversi gagal)
                    valid_data = data.dropna(subset=['at'])
                    
                    # 1. Aggregasi data harian dan bulanan
                    daily_sentiment = valid_data.resample('D', on='at')['Score'].mean().reset_index()
                    monthly_sentiment = valid_data.resample('M', on='at')['Score'].mean().reset_index()
                    
                    # Ubah nama kolom untuk konsistensi
                    daily_sentiment.columns = ['Tanggal', 'Skor']
                    monthly_sentiment.columns = ['Tanggal', 'Skor']
                    
                    # 2. Buat visualisasi trend bulanan
                    fig2 = px.line(monthly_sentiment, 
                                x='Tanggal', 
                                y='Skor',
                                title="Trend Sentimen Bulanan",
                                labels={'Skor': 'Intensitas Sentimen'},
                                markers=True)
                    
                    # 3. Format tampilan
                    fig2.update_xaxes(
                        dtick="M1",  # Tampilkan tiap bulan
                        tickformat="%b\n%Y",  # Format label "Apr\n2024"
                        tickangle=0
                    )
                    
                    fig2.update_layout(
                        xaxis_title="Bulan",
                        yaxis_title="Skor Sentimen",
                        hovermode="x unified",
                        height=500
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 4. Tampilkan Statistik Tambahan
                    if not daily_sentiment.empty:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            max_date = daily_sentiment.loc[daily_sentiment['Skor'].idxmax(), 'Tanggal']
                            st.metric("Hari Paling Positif", 
                                    max_date.strftime('%d %b %Y'), 
                                    f"Score: {daily_sentiment['Skor'].max():.2f}")
                        with col2:
                            min_date = daily_sentiment.loc[daily_sentiment['Skor'].idxmin(), 'Tanggal']
                            st.metric("Hari Paling Negatif", 
                                    min_date.strftime('%d %b %Y'), 
                                    f"Score: {daily_sentiment['Skor'].min():.2f}")
                        with col3:
                            st.metric("Rata-rata Harian", 
                                    f"{daily_sentiment['Skor'].mean():.2f}")
                    else:
                        st.warning("Tidak ada data valid untuk ditampilkan")
        
        except Exception as e:
            st.error(f"Terjadi error saat memproses data: {str(e)}")
            st.write("Detail error:", e)
    
    with tab3:
        st.subheader("Word Cloud Sentimen")
        
        st.markdown("**Positif**")
        text_pos = " ".join(data[data['Sentiment'] == 'Positif']['cleaned'])
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(text_pos)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        st.markdown("---")

        st.markdown("**Negatif**")
        text_neg = " ".join(data[data['Sentiment'] == 'Negatif']['cleaned'])
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white', max_words=500).generate(text_neg)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        
        st.markdown("---")

        st.markdown("**Netral**")
        text_neg = " ".join(data[data['Sentiment'] == 'Netral']['cleaned'])
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white', max_words=500).generate(text_neg)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    with tab4:
        st.subheader("Kata-Kata Umum Berdasarkan Sentimen")
    
        # Pastikan dataset Anda sudah memiliki kolom 'cleaned' dan 'Sentiment'
        if 'cleaned' in data.columns and 'Sentiment' in data.columns:
            # Ekstrak kata-kata dari teks yang sudah dibersihkan
            from collections import Counter
            import re
            
            # Fungsi untuk ekstrak kata
            def extract_words(text):
                if pd.isna(text):
                    return []
                return re.findall(r'\w+', text.lower())
            
            # Pisahkan data positif dan negatif
            positive_texts = data[data['Sentiment'] == 'Positif']['cleaned'].tolist()
            negative_texts = data[data['Sentiment'] == 'Negatif']['cleaned'].tolist()
            
            # Gabungkan semua teks untuk masing-masing sentimen
            all_positive_words = []
            for text in positive_texts:
                all_positive_words.extend(extract_words(text))
                
            all_negative_words = []
            for text in negative_texts:
                all_negative_words.extend(extract_words(text))
            
            # Hitung frekuensi kata
            positive_word_counts = Counter(all_positive_words)
            negative_word_counts = Counter(all_negative_words)
            
            # Ambil 10 kata paling umum untuk masing-masing sentimen
            top_positive = positive_word_counts.most_common(10)
            top_negative = negative_word_counts.most_common(10)
            
            # Buat DataFrame untuk visualisasi
            kata_data = pd.DataFrame({
                'Kata': [word for word, count in top_positive] + [word for word, count in top_negative],
                'Frekuensi': [count for word, count in top_positive] + [count for word, count in top_negative],
                'Sentimen': ['Positif']*len(top_positive) + ['Negatif']*len(top_negative)
            })
            
            # Buat visualisasi
            fig4 = px.bar(kata_data, x='Kata', y='Frekuensi', color='Sentimen',
                        barmode='group', height=500,
                        title='Kata-Kata Paling Umum Berdasarkan Sentimen',
                        color_discrete_map={'Positif': 'blue', 'Negatif': 'red'})
            
            fig4.update_layout(xaxis_title='Kata',
                            yaxis_title='Frekuensi',
                            showlegend=True)
            
            st.plotly_chart(fig4, use_container_width=True)
            
            # Tampilkan tabel data juga
            st.dataframe(kata_data.sort_values('Frekuensi', ascending=False))
        else:
            st.warning("Kolom 'cleaned' atau 'Sentiment' tidak ditemukan dalam dataset")
            
visualization()