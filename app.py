import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------
# Load Data
# --------------------------

df = pd.read_csv("spotify_tracks.csv")
model = joblib.load("popularity_model.pkl")

CLIENT_ID = st.secrets["95b23e8d94354c7e931ec06971c76072"]
CLIENT_SECRET = st.secrets["bb914b0ae41f4cfbb5629239ffbc493a"]

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

st.set_page_config(
    page_title="Spotify Music Trend Analyzer",
    layout="wide"
)

st.title("🎵 Spotify Music Trend Analyzer & Popularity Predictor")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Analysis",
    "🎯 Song Analyzer",
    "📈 Music Insights",
    "🔍 Spotify Search"
])

# =====================================================
# TAB 1 : DATASET ANALYSIS
# =====================================================

with tab1:

    st.header("Dataset Overview")

    st.write("Total Songs:", len(df))

    st.dataframe(df.head())

    st.subheader("🎼 Top 15 Genres")

    genre_counts = (
        df["track_genre"]
        .value_counts()
        .head(15)
    )

    st.bar_chart(genre_counts)

    st.subheader("🎵 Danceability vs Energy")

    sample_df = df.sample(
        min(5000, len(df))
    )

    fig = px.scatter(
        sample_df,
        x="danceability",
        y="energy",
        color="popularity",
        hover_data=["track_name"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📈 Popularity Distribution")

    fig2 = px.histogram(
        df,
        x="popularity",
        nbins=30
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("🎯 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": [
            "Acousticness",
            "Danceability",
            "Tempo",
            "Valence",
            "Speechiness",
            "Loudness",
            "Energy",
            "Liveness",
            "Instrumentalness"
        ],
        "Importance": [
            0.126860,
            0.118597,
            0.117383,
            0.117024,
            0.112829,
            0.111802,
            0.106312,
            0.100881,
            0.088313
        ]
    })

    fig3 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =====================================================
# TAB 2 : SONG ANALYZER
# =====================================================

with tab2:

    st.header("🔍 Search Song")

    song_name = st.text_input(
        "Enter Song Name"
    )

    if song_name:

        matches = df[
            df["track_name"]
            .str.contains(
                song_name,
                case=False,
                na=False
            )
        ]

        if len(matches) == 0:

            st.error("Song Not Found")

        else:

            matches = matches.sort_values(
                by="popularity",
                ascending=False
            )

            options = []

            for _, row in matches.iterrows():

                options.append(
                    f"{row['track_name']} - {row['artists']}"
                )

            selected = st.selectbox(
                "Select Song",
                options
            )

            selected_row = matches.iloc[
                options.index(selected)
            ]

            feature_df = pd.DataFrame([{
                "danceability":
                    selected_row["danceability"],
                "energy":
                    selected_row["energy"],
                "loudness":
                    selected_row["loudness"],
                "speechiness":
                    selected_row["speechiness"],
                "acousticness":
                    selected_row["acousticness"],
                "instrumentalness":
                    selected_row["instrumentalness"],
                "liveness":
                    selected_row["liveness"],
                "valence":
                    selected_row["valence"],
                "tempo":
                    selected_row["tempo"]
            }])

            prediction = model.predict(
                feature_df
            )[0]

            actual = int(
                selected_row["popularity"]
            )

            predicted = round(
                prediction
            )

            difference = abs(
                actual - predicted
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Actual Popularity",
                    actual
                )

            with col2:
                st.metric(
                    "Predicted Popularity",
                    predicted
                )

            with col3:
                st.metric(
                    "Difference",
                    difference
                )

            st.subheader("Song Details")

            st.write(
                "Artist:",
                selected_row["artists"]
            )

            st.write(
                "Genre:",
                selected_row["track_genre"]
            )

            st.subheader("Audio Features")

            feature_display = pd.DataFrame({
                "Feature": [
                    "Danceability",
                    "Energy",
                    "Valence",
                    "Acousticness",
                    "Speechiness",
                    "Liveness"
                ],
                "Value": [
                    selected_row["danceability"],
                    selected_row["energy"],
                    selected_row["valence"],
                    selected_row["acousticness"],
                    selected_row["speechiness"],
                    selected_row["liveness"]
                ]
            })

            fig4 = px.bar(
                feature_display,
                x="Feature",
                y="Value"
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

# =====================================================
# TAB 3 : LIVE SONGS
# =====================================================

with tab3:

    st.header("📈 Music Insights")

    # Top Songs
    st.subheader("🔥 Top 20 Most Popular Songs")

    top_songs = df.sort_values(
        by="popularity",
        ascending=False
    ).head(20)

    st.dataframe(
        top_songs[
            [
                "track_name",
                "artists",
                "track_genre",
                "popularity"
            ]
        ],
        use_container_width=True
    )

    # Top Artists
    st.subheader("🎤 Top Artists by Average Popularity")

    artist_popularity = (
        df.groupby("artists")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig_artist = px.bar(
        artist_popularity,
        x="artists",
        y="popularity",
        title="Top Artists"
    )

    st.plotly_chart(
        fig_artist,
        use_container_width=True
    )

    # Genre Popularity
    st.subheader("🎼 Most Popular Genres")

    genre_popularity = (
        df.groupby("track_genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig_genre = px.bar(
        genre_popularity,
        x="track_genre",
        y="popularity",
        title="Genre Popularity"
    )

    st.plotly_chart(
        fig_genre,
        use_container_width=True
    )

    # Correlation Heatmap
    st.subheader("📊 Audio Feature Correlation")

    numeric_cols = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "popularity"
    ]

    corr = df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )
# =====================================================
# TAB 4 : SPOTIFY SEARCH
# =====================================================

# =====================================================
# TAB 4 : SPOTIFY SEARCH
# =====================================================

with tab4:

    st.header("🔍 Spotify Song Search")

    search_query = st.text_input(
        "Search Spotify Song",
        key="spotify_search"
    )

    if search_query:

        try:

            results = sp.search(
                q=search_query,
                type="track",
                limit=10
            )

            tracks = results["tracks"]["items"]

            if len(tracks) == 0:

                st.warning("No songs found.")

            else:

                songs = []

                for track in tracks:

                    songs.append({
                        "Song": track["name"],
                        "Artist": track["artists"][0]["name"],
                        "Album": track["album"]["name"],
                        "Release Date": track["album"]["release_date"],
                        "Spotify Link": track["external_urls"]["spotify"]
                    })

                spotify_df = pd.DataFrame(songs)

                st.data_editor(
                    spotify_df,
                    column_config={
                        "Spotify Link":
                        st.column_config.LinkColumn(
                            "Spotify Link"
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Spotify API Error: {e}")