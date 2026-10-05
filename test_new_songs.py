import os
import glob

import numpy as np
import pandas as pd
import joblib


# ==================================================
# 0. SETTINGS
# ==================================================

MODEL_FILE = "music_model.pkl"        # created by music_recommender_simple.py
NEW_SONGS_FOLDER = "new_songs"        # put your 15 new songs here
OUTPUT_FOLDER = "results"
K_RECOMMEND = 5
NEW_SONG_OFFSET = 30                  # skip the first 30 seconds (intro) of long songs
AUDIO_TYPES = ("wav", "mp3", "flac", "ogg", "m4a")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==================================================
# 1. LOAD THE TRAINED MODEL
# ==================================================

if not os.path.exists(MODEL_FILE):
    print("music_model.pkl not found.")
    print("First run:  ..\\venv\\Scripts\\python.exe music_recommender_simple.py")
    raise SystemExit

model = joblib.load(MODEL_FILE)

scaler = model["scaler"]
pca = model["pca"]
knn = model["knn"]
kmeans = model["kmeans"]
songs = model["songs"]
feature_names = model["feature_names"]

known_genres = set(songs["genre"].unique())

print("Model loaded successfully!")
print("Songs in the collection:", len(songs))


# ==================================================
# 2. FIND THE NEW SONGS
# ==================================================

new_files = []
for audio_type in AUDIO_TYPES:
    new_files += glob.glob(os.path.join(NEW_SONGS_FOLDER, "**", "*." + audio_type), recursive=True)
new_files = sorted(new_files)

print("\nNew songs found:", len(new_files))

if len(new_files) == 0:
    print("No songs found. Create a folder named", NEW_SONGS_FOLDER,
          "and put your songs inside it.")
    raise SystemExit

if len(new_files) != 15:
    print("Note: your teacher asked for 15 songs, and", len(new_files), "were found.")


# ==================================================
# 3. EXTRACT FEATURES FROM THE NEW SONGS
# ==================================================

def extract_features(path, sr=22050, duration=30):
    """Turn one audio file into a dictionary of 70 numbers (features)."""
    import librosa

    try:
        # long songs: skip the intro and take 30 seconds from inside the song
        offset = 0
        if librosa.get_duration(path=path) >= NEW_SONG_OFFSET + duration:
            offset = NEW_SONG_OFFSET

        y, sr = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)

        if len(y) < sr:
            return None

        feats = {}

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            feats[f"mfcc{i}_mean"] = mfcc[i].mean()
            feats[f"mfcc{i}_std"] = mfcc[i].std()

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            feats[f"chroma{i}_mean"] = chroma[i].mean()

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        feats["centroid_mean"] = centroid.mean()
        feats["centroid_std"] = centroid.std()
        feats["bandwidth_mean"] = bandwidth.mean()
        feats["bandwidth_std"] = bandwidth.std()
        feats["rolloff_mean"] = rolloff.mean()
        feats["rolloff_std"] = rolloff.std()

        for i in range(contrast.shape[0]):
            feats[f"contrast{i}_mean"] = contrast[i].mean()

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        feats["zcr_mean"] = zcr.mean()
        feats["zcr_std"] = zcr.std()
        feats["rms_mean"] = rms.mean()
        feats["rms_std"] = rms.std()
        feats["tempo"] = float(np.atleast_1d(tempo)[0])

        return feats

    except Exception as error:
        print("Skipped", path, "->", error)
        return None


song_names = []
song_genres = []
song_features = []

for number, file_path in enumerate(new_files, start=1):
    print("Reading song", number, "of", len(new_files), ":", os.path.basename(file_path))

    features = extract_features(file_path)
    if features is None:
        continue

    # genre = name of the folder inside new_songs (if you used genre folders)
    parent = os.path.basename(os.path.dirname(file_path))
    genre = parent if parent in known_genres else "unknown"

    song_names.append(os.path.basename(file_path))
    song_genres.append(genre)
    song_features.append(features)

print("\nSongs ready:", len(song_features))

if len(song_features) == 0:
    raise SystemExit


# ==================================================
# 4. RECOMMEND SIMILAR SONGS FOR EACH NEW SONG
# ==================================================

new_table = pd.DataFrame(song_features)[feature_names]
new_points = pca.transform(scaler.transform(new_table))

distances, neighbours = knn.kneighbors(new_points, n_neighbors=K_RECOMMEND)
new_clusters = kmeans.predict(new_points)

detail_rows = []
summary_rows = []

for i in range(len(song_names)):
    recommended = songs.iloc[neighbours[i]]
    recommended_genres = list(recommended["genre"])

    for rank in range(K_RECOMMEND):
        detail_rows.append({
            "New song": song_names[i],
            "Actual genre": song_genres[i],
            "Rank": rank + 1,
            "Recommended song": os.path.basename(recommended.iloc[rank]["path"]),
            "Recommended genre": recommended_genres[rank],
            "Similarity": round(1 - distances[i][rank], 3),
        })

    if song_genres[i] == "unknown":
        same_genre = "-"
    else:
        same_genre = str(sum(g == song_genres[i] for g in recommended_genres)) + " / " + str(K_RECOMMEND)

    summary_rows.append({
        "New song": song_names[i],
        "Actual genre": song_genres[i],
        "Cluster": int(new_clusters[i]),
        "Top-5 genres": ", ".join(recommended_genres),
        "Same genre": same_genre,
    })

detail = pd.DataFrame(detail_rows)
summary = pd.DataFrame(summary_rows)

print("\n================================")
print("NEW SONG RECOMMENDATIONS")
print("================================")

for i in range(len(song_names)):
    print("\nNew song:", song_names[i], "| actual genre:", song_genres[i],
          "| cluster:", int(new_clusters[i]))
    print(detail[detail["New song"] == song_names[i]][
        ["Rank", "Recommended song", "Recommended genre", "Similarity"]].to_string(index=False))


# ==================================================
# 5. EVALUATION ON THE NEW SONGS
# ==================================================

print("\n================================")
print("SUMMARY TABLE")
print("================================")
print(summary.to_string(index=False))

labelled = detail[detail["Actual genre"] != "unknown"]

if len(labelled) > 0:
    new_precision = (labelled["Actual genre"] == labelled["Recommended genre"]).mean()
    print("\nPrecision@5 on the new songs with a known genre:", round(new_precision, 3))
    print("Songs with a known genre:", labelled["New song"].nunique())
else:
    new_precision = None
    print("\nNo genre folders were used, so Precision@5 cannot be calculated.")


# ==================================================
# 6. SAVE RESULTS
# ==================================================

detail.to_csv(os.path.join(OUTPUT_FOLDER, "new_songs_recommendations.csv"), index=False)
summary.to_csv(os.path.join(OUTPUT_FOLDER, "new_songs_summary.csv"), index=False)

print("\nResults saved in the folder:", OUTPUT_FOLDER)


# ==================================================
# 7. FINAL SUMMARY
# ==================================================

print("\n================================")
print("FINAL TEST ON NEW SONGS")
print("================================")

print("New songs tested :", len(song_names))
cluster_counts = pd.Series(new_clusters).value_counts().sort_index()
print("Songs per cluster:", {int(k): int(v) for k, v in cluster_counts.items()})

if new_precision is not None:
    print("Precision@5      :", round(new_precision, 3))

print("================================")