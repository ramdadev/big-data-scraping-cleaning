from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
from tqdm import tqdm
import os

# Inisialisasi YoutubeCommentDownloader
downloader = YoutubeCommentDownloader()

# Link Video
video_url = 'https://www.youtube.com/watch?v=4E3i_L5rGv8'

# Tampung komentar ke dalam list array
comments = []

# Ambil komentar
for comment in tqdm(downloader.get_comments_from_url(video_url, sort_by=0), desc="Memproses permintaan", unit=" komentar", total=None):
    author = comment.get('author', '')
    text = comment.get('text', '')
    time = comment.get('time', '')
    likes = comment.get('likes', 0)

    comments.append({
        'author': author,
        'comment': text,
        'time': time,
        'likes': likes
    })

# Pastikan folder "output" ada
os.makedirs("output", exist_ok=True)

# Simpan ke CSV
df = pd.DataFrame(comments)
df.to_csv("output/comments-youtube.csv", index=False)

print("Permintaan berhasil diproses. Hasil disimpan di output/comments-youtube.csv")
