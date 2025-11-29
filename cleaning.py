import pandas as pd
import re
import string
from tqdm import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Buka file CSV
df = pd.read_csv('output/comments-youtube.csv')

# Jika file CSV kosong, tampilkan pesan kesalahan
if df.empty:
    print("File CSV tidak ditemukan atau kosong.")
    exit()

# Buat daftar stopwords dari Sastrawi
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

# Fungsi untuk membersihkan teks
def clean_text(text):
    if not isinstance(text, str):
        return ''

    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Hapus emoji dan simbol non-alfanumerik
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Lowercase
    text = text.lower()

    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus stopwords bahasa Indonesia
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Hapus spasi ekstra
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Konfigurasi progress bar
tqdm.pandas(
    desc="Membersihkan komentar",
    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} | ETA: {remaining} | {rate_fmt}"
)

# Terapkan cleaning dengan progress bar
df['clean_comment'] = df['comment'].progress_apply(clean_text)

# Simpan hasil ke file CSV
df.to_csv('output/comments-youtube-cleaned.csv', index=False)

# Tampilkan pesan sukses
print("Proses membersihkan teks selesai. Hasil tersimpan di output/comments-youtube-cleaned.csv")
print("Jumlah data:", len(df))