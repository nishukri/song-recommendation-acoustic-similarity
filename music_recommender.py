"""
Song Recommendation based on Acoustic Similarity
Pipeline: audio -> features (librosa) -> scale + PCA -> cosine kNN / KMeans -> recommendations
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==================================================
# PSEUDO CODE
# ==================================================
#
#  INPUT : audio files, a query song, K (number of recommendations)
#  OUTPUT: K similar songs, song clusters, Precision@K
#
#   1. FOR each audio file
#   2.     load the first 30 seconds of the song
#   3.     compute MFCC, chroma, spectral features, ZCR, RMS and tempo
#   4.     summarise them (mean, std)  ->  one vector of 70 numbers
#   5. Save all vectors in a table (one row per song)
#   6. Standardise every column (mean 0, standard deviation 1)
#   7. Apply PCA and keep 99% of the variance
#   8. Fit a nearest-neighbour model on the PCA data (cosine distance)
#   9. FOR k = 2 to 15
#  10.     run K-Means and compute the silhouette score
#  11. Choose the k with the highest silhouette score and make the clusters
#  12. RECOMMEND(song):
#  13.     get the PCA vector of the song
#  14.         (a new song: same scaler and same PCA as the 999 songs)
#  15.     compute the cosine similarity to every song in the collection
#  16.     return the K most similar songs (the song itself is not returned)
#  17. EVALUATE:
#  18.     FOR each song: take its K recommendations, count how many have the same genre
#  19.     Precision@K = average of (same-genre count / K)
#  20. TEST ON NEW SONGS: repeat RECOMMEND for each new song and draw it as a star


# ==================================================
# 0. SETTINGS (change only these if needed)
# ==================================================

AUDIO_FOLDER = "genres_original"      # the 999 songs (used only if features.csv is missing)
FEATURES_FILE = "features.csv"        # table of song features
NEW_SONGS_FOLDER = "new_songs"        # new songs, inside genre folders (jazz, metal, pop ...)
OUTPUT_FOLDER = "results"             # pictures and tables are saved here
MODEL_FILE = "music_model.pkl"        # saved model
NEW_CACHE_FILE = os.path.join(OUTPUT_FOLDER, "new_songs_features.csv")

PCA_VARIANCE = 0.99                   # final PCA setting (best result)
K_RECOMMEND = 5                       # number of recommended songs
NEW_SONG_OFFSET = 30                  # new songs: skip the first 30 seconds (intro)
AUDIO_TYPES = ("wav", "mp3", "flac", "ogg", "m4a")

SONG_TO_TEST = "blues.00001"          # example song from the 999 songs
NEW_SONG_TO_TEST = 1                  # example new song (star number 1 to 15)

QUICK_MODE = False                    # False = run everything (3-6 minutes)
                                      # True  = skip the slow steps, use saved results (about 20 seconds)
PLAY_SONGS = False                    # True = play the example songs
SHOW_PLOTS = False                    # True = open picture windows (close each window to continue)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def finish_figure(filename):
    """Save the current picture in the results folder, then show or close it."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    try:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved picture:", path)
    except OSError as error:
        print("Could not save", filename, "->", error)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def play_file(path):
    """Play a song or open a picture with the default Windows program."""
    if not os.path.exists(path):
        print("File not found:", path)
    elif hasattr(os, "startfile"):
        os.startfile(path)
    else:
        print("Open this file yourself:", path)


def extract_features(path, sr=22050, duration=30, skip_intro=False):
    """Turn one audio file into a dictionary of 70 numbers (features)."""
    import librosa

    try:
        offset = 0
        if skip_intro and librosa.get_duration(path=path) >= NEW_SONG_OFFSET + duration:
            offset = NEW_SONG_OFFSET

        y, sr = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)

        if len(y) < sr:
            return None

        feats = {}

        # Timbre (sound texture): MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            feats[f"mfcc{i}_mean"] = mfcc[i].mean()
            feats[f"mfcc{i}_std"] = mfcc[i].std()

        # Harmony (notes): chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            feats[f"chroma{i}_mean"] = chroma[i].mean()

        # Brightness: spectral features
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

        # Noisiness, loudness, speed
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


def find_best_k(points, k_min=2, k_max=15):
    """Try k = 2..15 and return the k with the best silhouette score."""
    best_k, best_score = k_min, -1
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(points)
        score = silhouette_score(points, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def song_matches(points, genre_labels, k=5):
    """For every song, mark which of its top-k recommendations have the same genre."""
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(points)
    _, neighbours = nn.kneighbors(points, n_neighbors=k + 1)
    neighbours = neighbours[:, 1:]                      # remove the song itself
    labels = np.array(genre_labels)
    return labels[neighbours] == labels[:, None]


# ==================================================
# 1. LOAD DATASET (999 songs)
# ==================================================

if not os.path.exists(FEATURES_FILE):
    print("features.csv not found. Extracting features (15-30 minutes)...")

    audio_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "**", "*.wav"), recursive=True))
    print("Audio files found:", len(audio_files))

    rows = []
    for number, file_path in enumerate(audio_files, start=1):
        result = extract_features(file_path)
        if result is not None:
            result["path"] = file_path
            result["genre"] = os.path.basename(os.path.dirname(file_path))
            rows.append(result)
        if number % 50 == 0:
            print("Processed", number, "of", len(audio_files))

    pd.DataFrame(rows).to_csv(FEATURES_FILE, index=False)
    print("Features saved to", FEATURES_FILE)

data = pd.read_csv(FEATURES_FILE)

print("Dataset loaded successfully!")
print("Songs   :", data.shape[0])
print("Columns :", data.shape[1])
print("\nSongs per genre:")
print(data["genre"].value_counts().sort_index())


# ==================================================
# 2. SEPARATE FEATURES AND LABELS
# ==================================================

# X = the 70 numbers of each song
# genres = the folder name (used ONLY to check the result, not to build the model)

X = data.drop(["path", "genre"], axis=1)
genres = data["genre"]

print("\nNumber of features:", X.shape[1])


# ==================================================
# 3. CHECK MISSING VALUES
# ==================================================

missing = int(X.isnull().sum().sum())
print("Missing values:", missing)

if missing > 0:
    X = X.fillna(X.median())
    print("Missing values filled with the median.")


# ==================================================
# 4. SCALE FEATURES
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled successfully!")


# ==================================================
# 5. REDUCE FEATURES WITH PCA
# ==================================================

pca = PCA(n_components=PCA_VARIANCE, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("PCA:", X_scaled.shape[1], "features ->", X_pca.shape[1],
      "components (" + str(int(PCA_VARIANCE * 100)) + "% variance kept)")


# ==================================================
# 6. BUILD SIMILARITY MODEL (cosine nearest neighbours)
# ==================================================

knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(X_pca)

print("Similarity model built successfully!")


# ==================================================
# 7. FIND BEST NUMBER OF CLUSTERS AND RUN K-MEANS
# ==================================================

saved_k = None
if QUICK_MODE and os.path.exists(MODEL_FILE):
    old_model = joblib.load(MODEL_FILE)
    if "best_k" in old_model:
        saved_k = old_model["best_k"]
        saved_silhouette = old_model["best_silhouette"]

if saved_k is not None:
    best_k, best_silhouette = saved_k, saved_silhouette
    print("\nQUICK MODE: using the saved number of clusters.")
else:
    best_k, best_silhouette = find_best_k(X_pca)

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
data["cluster"] = kmeans.fit_predict(X_pca)

print("Best number of clusters:", best_k)
print("Silhouette score       :", round(best_silhouette, 3))


# ==================================================
# 8. MODEL EVALUATION (Precision@5)
# ==================================================

matches = song_matches(X_pca, genres, K_RECOMMEND)
precision_at_5 = matches.mean()

genre_counts = genres.value_counts()
total_songs = len(genres)
random_baseline = (genre_counts * (genre_counts - 1)).sum() / (total_songs * (total_songs - 1))

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("Precision@5     :", round(precision_at_5, 3))
print("Random baseline :", round(random_baseline, 3))
print("Times better    :", round(precision_at_5 / random_baseline, 1))

print("\nPrecision@5 for each genre:")
precision_by_genre = pd.Series(matches.mean(axis=1), index=genres.values).groupby(level=0).mean()
print(precision_by_genre.round(3).sort_values(ascending=False))


# ==================================================
# 9. COMPARE DIFFERENT PCA SETTINGS
# ==================================================

print("\n================================")
print("PCA COMPARISON")
print("================================")

COMPARISON_FILE = os.path.join(OUTPUT_FOLDER, "pca_comparison.csv")

if QUICK_MODE and os.path.exists(COMPARISON_FILE):
    comparison = pd.read_csv(COMPARISON_FILE)
    comparison["Variance kept"] = comparison["Variance kept"].astype(str)
    print("QUICK MODE: using the saved comparison.")
else:
    comparison_rows = []

    for variance in [0.80, 0.95, 0.99]:
        points = PCA(n_components=variance, random_state=42).fit_transform(X_scaled)

        score = song_matches(points, genres, K_RECOMMEND).mean()
        k_test, silhouette_test = find_best_k(points)

        comparison_rows.append({
            "Variance kept": str(int(variance * 100)) + "%",
            "Components": points.shape[1],
            "Best k": k_test,
            "Silhouette": round(silhouette_test, 3),
            "Precision@5": round(score, 3),
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_FILE, index=False)

print(comparison.to_string(index=False))


# ==================================================
# 10. CLUSTER TABLE (genre vs cluster)
# ==================================================

print("\n================================")
print("GENRES INSIDE EACH CLUSTER")
print("================================")

print(pd.crosstab(data["genre"], data["cluster"]))


# ==================================================
# 11. CLUSTER PICTURE 1: THE 999 SONGS
# ==================================================

colors = plt.cm.tab10.colors
pca_2d = PCA(n_components=2, random_state=42).fit(X_pca)
points_2d = pca_2d.transform(X_pca)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for cluster_number in sorted(data["cluster"].unique()):
    selected = data["cluster"] == cluster_number
    plt.scatter(points_2d[selected, 0], points_2d[selected, 1], s=15,
                color=colors[cluster_number % 10], label="Cluster " + str(cluster_number))
plt.title("KMeans clusters (k=" + str(best_k) + ")")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.subplot(1, 2, 2)
for number, genre_name in enumerate(sorted(genres.unique())):
    selected = (genres == genre_name).values
    plt.scatter(points_2d[selected, 0], points_2d[selected, 1], s=15,
                color=colors[number % 10], label=genre_name)
plt.title("True genre labels (folder names)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=8)

plt.tight_layout()
finish_figure("clusters.png")


# ==================================================
# 12. PCA COMPARISON CHART
# ==================================================

plt.figure(figsize=(6, 5))

plt.bar(comparison["Variance kept"], comparison["Precision@5"])
plt.axhline(random_baseline, linestyle="--", color="red", label="Random baseline")

for position, value in enumerate(comparison["Precision@5"]):
    plt.text(position, value + 0.01, str(value), ha="center")

plt.title("Precision@5 for Different PCA Settings")
plt.xlabel("PCA variance kept")
plt.ylabel("Precision@5")
plt.ylim(0, 0.8)
plt.legend()

finish_figure("pca_comparison.png")


# ==================================================
# 13. PRECISION BY GENRE CHART
# ==================================================

plt.figure(figsize=(8, 5))

sorted_precision = precision_by_genre.sort_values()
plt.barh(sorted_precision.index, sorted_precision.values)
plt.axvline(random_baseline, linestyle="--", color="red", label="Random baseline")

plt.title("Precision@5 for Each Genre")
plt.xlabel("Precision@5")
plt.legend()

finish_figure("precision_by_genre.png")


# ==================================================
# 14. SAVE TRAINED MODEL
# ==================================================

joblib.dump(
    {
        "scaler": scaler,
        "pca": pca,
        "knn": knn,
        "kmeans": kmeans,
        "songs": data[["path", "genre", "cluster"]],
        "feature_names": list(X.columns),
        "best_k": best_k,
        "best_silhouette": best_silhouette,
    },
    MODEL_FILE
)

print("\nModel saved successfully! (" + MODEL_FILE + ")")


# ==================================================
# 15. RECOMMEND SONGS FOR ONE EXAMPLE SONG (999 songs)
# ==================================================

print("\n================================")
print("SONG RECOMMENDATION")
print("================================")

found = data.index[data["path"].str.contains(SONG_TO_TEST, regex=False)]

if len(found) == 0:
    print("Song not found:", SONG_TO_TEST)
else:
    query_index = found[0]

    distances, neighbours = knn.kneighbors(X_pca[query_index:query_index + 1],
                                           n_neighbors=K_RECOMMEND + 1)
    neighbours = neighbours[0][1:]
    distances = distances[0][1:]

    recommendations = pd.DataFrame({
        "Song": [os.path.basename(p) for p in data.loc[neighbours, "path"]],
        "Genre": data.loc[neighbours, "genre"].values,
        "Cluster": data.loc[neighbours, "cluster"].values,
        "Similarity": (1 - distances).round(3),
    })

    print("Chosen song :", os.path.basename(data.loc[query_index, "path"]),
          "(genre: " + data.loc[query_index, "genre"] + ")")
    print("\nTop", K_RECOMMEND, "similar songs:")
    print(recommendations.to_string(index=False))

    same_genre = (recommendations["Genre"] == data.loc[query_index, "genre"]).sum()
    print("\nSame genre:", same_genre, "out of", K_RECOMMEND)

    if PLAY_SONGS:
        play_file(data.loc[query_index, "path"])


# ==================================================
# 16. LOAD THE NEW SONGS
# ==================================================

print("\n================================")
print("TEST ON NEW SONGS")
print("================================")

new_files = []
for audio_type in AUDIO_TYPES:
    new_files += glob.glob(os.path.join(NEW_SONGS_FOLDER, "**", "*." + audio_type), recursive=True)
new_files = sorted(new_files)

print("New songs found:", len(new_files))

if len(new_files) == 0:
    print("No new songs found in the folder:", NEW_SONGS_FOLDER)
    print("Put your songs in genre folders inside it (example: new_songs\\jazz) and run again.")
else:
    new_cache = None
    if os.path.exists(NEW_CACHE_FILE):
        new_cache = pd.read_csv(NEW_CACHE_FILE)
        if set(new_cache["path"]) != set(new_files):
            new_cache = None

    if new_cache is None:
        print("Reading the new songs (only the first time, a few minutes)...")
        rows = []
        for number, file_path in enumerate(new_files, start=1):
            print("Reading song", number, "of", len(new_files), ":", os.path.basename(file_path))
            features = extract_features(file_path, skip_intro=True)
            if features is None:
                continue
            parent = os.path.basename(os.path.dirname(file_path))
            features["path"] = file_path
            features["name"] = os.path.basename(file_path)
            features["genre"] = parent if parent in set(genres) else "unknown"
            rows.append(features)
        new_cache = pd.DataFrame(rows)
        new_cache.to_csv(NEW_CACHE_FILE, index=False)
    else:
        print("New songs loaded from the saved file.")

    new_names = list(new_cache["name"])
    new_genres = list(new_cache["genre"])
    new_paths = list(new_cache["path"])

    # same scaler and same PCA as the 999 songs
    new_points = pca.transform(scaler.transform(new_cache[list(X.columns)]))
    new_clusters = kmeans.predict(new_points)

    print("New songs ready:", len(new_names))


    # ==================================================
    # 17. RECOMMEND SONGS FOR EACH NEW SONG
    # ==================================================

    new_distances, new_neighbours = knn.kneighbors(new_points, n_neighbors=K_RECOMMEND)

    detail_rows = []
    summary_rows = []

    for i in range(len(new_names)):
        rec_songs = data.iloc[new_neighbours[i]]
        rec_genres = list(rec_songs["genre"])

        for rank in range(K_RECOMMEND):
            detail_rows.append({
                "Star": i + 1,
                "New song": new_names[i],
                "Actual genre": new_genres[i],
                "Rank": rank + 1,
                "Recommended song": os.path.basename(rec_songs.iloc[rank]["path"]),
                "Recommended genre": rec_genres[rank],
                "Similarity": round(1 - new_distances[i][rank], 3),
            })

        if new_genres[i] == "unknown":
            same = "-"
        else:
            same = str(sum(g == new_genres[i] for g in rec_genres)) + " / " + str(K_RECOMMEND)

        summary_rows.append({
            "Star": i + 1,
            "New song": new_names[i],
            "Actual genre": new_genres[i],
            "Cluster": int(new_clusters[i]),
            "Top-5 genres": ", ".join(rec_genres),
            "Same genre": same,
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    print("\n================================")
    print("NEW SONG SUMMARY")
    print("================================")
    print(summary.to_string(index=False))

    labelled = detail[detail["Actual genre"] != "unknown"]

    if len(labelled) > 0:
        new_precision = (labelled["Actual genre"] == labelled["Recommended genre"]).mean()
        print("\nPrecision@5 on the new songs:", round(new_precision, 3))
    else:
        new_precision = None
        print("\nNo genre folders were used, so Precision@5 cannot be calculated.")

    detail.to_csv(os.path.join(OUTPUT_FOLDER, "new_songs_recommendations.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_FOLDER, "new_songs_summary.csv"), index=False)
    print("Tables saved in the folder:", OUTPUT_FOLDER)


    # ==================================================
    # 18. CLUSTER PICTURE 2: THE 999 SONGS + NEW SONGS AS STARS
    # ==================================================

    new_2d = pca_2d.transform(new_points)
    genre_color = {g: colors[n % 10] for n, g in enumerate(sorted(genres.unique()))}

    star_legend = Line2D([0], [0], marker="*", color="w", markerfacecolor="lightgray",
                         markeredgecolor="black", markersize=16, label="New songs (stars)")

    plt.figure(figsize=(15, 7))

    # Left picture: clusters (new songs = stars)
    plt.subplot(1, 2, 1)
    for cluster_number in sorted(data["cluster"].unique()):
        selected = (data["cluster"] == cluster_number).values
        plt.scatter(points_2d[selected, 0], points_2d[selected, 1], s=14, alpha=0.5,
                    color=colors[cluster_number % 10], label="Cluster " + str(cluster_number))
    for i in range(len(new_names)):
        plt.scatter(new_2d[i, 0], new_2d[i, 1], marker="*", s=350, linewidths=1.2,
                    color=colors[new_clusters[i] % 10], edgecolor="black", zorder=5)
        plt.annotate(str(i + 1), (new_2d[i, 0], new_2d[i, 1]), textcoords="offset points",
                     xytext=(7, 7), fontsize=10, fontweight="bold", zorder=6)
    plt.title("KMeans clusters (k=" + str(best_k) + ") with the new songs (stars)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles + [star_legend], labels + ["New songs (stars)"])

    # Right picture: true genres (new songs = stars)
    plt.subplot(1, 2, 2)
    for genre_name in sorted(genres.unique()):
        selected = (genres == genre_name).values
        plt.scatter(points_2d[selected, 0], points_2d[selected, 1], s=14, alpha=0.5,
                    color=genre_color[genre_name], label=genre_name)
    for i in range(len(new_names)):
        plt.scatter(new_2d[i, 0], new_2d[i, 1], marker="*", s=350, linewidths=1.2,
                    color=genre_color.get(new_genres[i], "black"), edgecolor="black", zorder=5)
        plt.annotate(str(i + 1), (new_2d[i, 0], new_2d[i, 1]), textcoords="offset points",
                     xytext=(7, 7), fontsize=10, fontweight="bold", zorder=6)
    plt.title("True genres with the new songs (stars)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles + [star_legend], labels + ["New songs (stars)"], fontsize=8)

    plt.tight_layout()
    finish_figure("clusters_with_new_songs.png")

    print("\nStar numbers in the picture:")
    print(summary[["Star", "New song", "Actual genre", "Cluster"]].to_string(index=False))


    # ==================================================
    # 19. RECOMMEND FOR ONE EXAMPLE NEW SONG
    # ==================================================

    print("\n================================")
    print("NEW SONG RECOMMENDATION")
    print("================================")

    if 1 <= NEW_SONG_TO_TEST <= len(new_names):
        i = NEW_SONG_TO_TEST - 1
        print("Chosen new song:", new_names[i], "| genre:", new_genres[i],
              "| cluster:", int(new_clusters[i]))
        print("\nTop", K_RECOMMEND, "similar songs from the 999:")
        print(detail[detail["Star"] == NEW_SONG_TO_TEST][
            ["Rank", "Recommended song", "Recommended genre", "Similarity"]].to_string(index=False))

        if PLAY_SONGS:
            play_file(new_paths[i])
    else:
        print("NEW_SONG_TO_TEST must be a number from 1 to", len(new_names))


# ==================================================
# 20. FINAL PROJECT SUMMARY
# ==================================================

print("\n================================")
print("FINAL MODEL PERFORMANCE")
print("================================")

print("Songs used        :", data.shape[0])
print("Features per song :", X.shape[1])
print("PCA components    :", X_pca.shape[1])
print("Clusters          :", best_k)
print("Silhouette score  :", round(best_silhouette, 3))
print("Precision@5       :", round(precision_at_5, 3))
print("Random baseline   :", round(random_baseline, 3))

if len(new_files) > 0:
    print("New songs tested  :", len(new_names))
    if new_precision is not None:
        print("Precision@5 (new) :", round(new_precision, 3))

print("================================")
