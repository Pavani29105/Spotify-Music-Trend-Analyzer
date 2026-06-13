import pandas as pd

df = pd.read_csv("spotify_tracks.csv")

df = df.drop_duplicates()
df = df.dropna()

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

target = "popularity"

processed = df[features + [target]]

processed.to_csv("processed_tracks.csv", index=False)

print(processed.shape)
print(processed.head())