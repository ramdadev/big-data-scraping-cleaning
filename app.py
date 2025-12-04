import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download lexicon VADER (hanya pertama kali)
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Fungsi analisis sentimen
def analisis_sentimen(teks):
    skor = sia.polarity_scores(teks)
    if skor['compound'] > 0.05:
        return "positif"
    elif skor['compound'] < -0.05:
        return "negatif"
    else:
        return "netral"

# --- UI ---
st.title("Analisis Sentimen - BIG DATA")
st.write("Upload file CSV berisi komentar yang sudah dicleaning")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Asumsi kolom komentar bernama 'clean_comment'
    if "clean_comment" not in df.columns:
        st.error("CSV harus punya kolom 'clean_comment'")
    else:
        st.success("File berhasil dibaca!")

        st.subheader("Data Komentar")
        st.dataframe(df)

        # Analisis sentimen
        df["sentimen"] = df["clean_comment"].astype(str).apply(analisis_sentimen)

        st.subheader("Hasil Analisis")
        st.dataframe(df)

        # Download hasil
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Hasil Analisis (CSV)",
            data=csv_download,
            file_name="hasil_sentimen.csv",
            mime="text/csv"
        )

        # --- INSIGHT & ANALISIS ---
        st.subheader("Insight")

        total = len(df)
        positif = (df["sentimen"] == "positif").sum()
        negatif = (df["sentimen"] == "negatif").sum()
        netral = (df["sentimen"] == "netral").sum()

        insight_text = f"""
        ### Ringkasan Insight Berdasarkan Komentar:
        - **Total komentar:** {total}
        - **Komentar positif:** {positif} ({positif/total*100:.1f}%)
        - **Komentar negatif:** {negatif} ({negatif/total*100:.1f}%)
        - **Komentar netral:** {netral} ({netral/total*100:.1f}%)

        ### Analisis:
        - Sentimen **dominan**: **{df['sentimen'].mode()[0]}**
        - Secara umum, komentar bersifat **{'lebih positif' if positif > negatif else 'lebih negatif' if negatif > positif else 'seimbang'}**.
        """

        st.markdown(insight_text)

        # --- Contoh Komentar Berdasarkan Sentimen ---
        st.subheader("Contoh Komentar Berdasarkan Sentimen")

        st.markdown("### Komentar Positif")
        st.write(df[df["sentimen"] == "positif"]["clean_comment"].head(5).tolist())

        st.markdown("### Komentar Negatif")
        st.write(df[df["sentimen"] == "negatif"]["clean_comment"].head(5).tolist())

        st.markdown("### Komentar Netral")
        st.write(df[df["sentimen"] == "netral"]["clean_comment"].head(5).tolist())


        # --- Visualisasi ---
        st.subheader("Visualisasi Sentimen")

        # Hitung jumlah sentimen
        count = df["sentimen"].value_counts()

        # Bar chart
        st.write("### Bar Chart Sentimen")
        fig1, ax1 = plt.subplots()
        ax1.bar(count.index, count.values)
        ax1.set_xlabel("Kategori Sentimen")
        ax1.set_ylabel("Jumlah")
        ax1.set_title("Distribusi Sentimen")
        st.pyplot(fig1)

        # Pie chart
        st.write("### Pie Chart Sentimen")
        fig2, ax2 = plt.subplots()
        ax2.pie(count.values, labels=count.index, autopct='%1.1f%%')
        ax2.set_title("Persentase Sentimen")
        st.pyplot(fig2)
