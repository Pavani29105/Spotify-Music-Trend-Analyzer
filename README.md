# 🎵 Spotify Music Trend Analyzer & Popularity Predictor

## 📌 Project Overview

Spotify Music Trend Analyzer is a Machine Learning and Data Analytics project that predicts song popularity using audio features from Spotify tracks. The project also provides interactive visualizations, music insights, and live Spotify song search using the Spotify Web API.

Built using:

- Python
- Streamlit
- Scikit-Learn
- Pandas
- Plotly
- Spotify API

---

## 🚀 Features

### 📊 Dataset Analysis
- Explore 114,000+ Spotify tracks
- Genre distribution analysis
- Popularity distribution visualization
- Danceability vs Energy analysis
- Feature importance visualization

### 🎯 Song Popularity Predictor
- Search songs from the dataset
- Predict popularity using Machine Learning
- Compare actual vs predicted popularity
- Analyze song audio features

### 📈 Music Insights
- Top 20 most popular songs
- Most popular artists
- Most popular genres
- Audio feature correlation analysis

### 🔍 Spotify Search
- Search songs using Spotify API
- View artist, album, and release date
- Open songs directly in Spotify

---

## 📂 Dataset

Dataset Source:
Spotify Tracks Dataset (Kaggle)

Dataset Size:
- 114,000+ Songs
- 21 Features

Features Used:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

Target Variable:

- Popularity

---

## 🤖 Machine Learning Model

Algorithm:

- Random Forest Regressor

Evaluation Metrics:

- Mean Absolute Error (MAE)
- R² Score

Model predicts song popularity based on audio characteristics.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| Scikit-Learn | Machine Learning |
| Plotly | Data Visualization |
| Streamlit | Web Application |
| Spotify API | Live Music Search |

---

## 📁 Project Structure

```text
Spotify-Music-Trend-Analyzer/
│
├── app.py
├── train_model.py
├── preprocess.py
├── spotify_tracks.csv
├── popularity_model.pkl
├── requirements.txt
├── README.md
└── fetch_data.py
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone <repository-url>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📷 Dashboard Features

### Dataset Analysis
- Genre Distribution
- Popularity Distribution
- Feature Importance

### Song Analyzer
- Search Song
- Actual Popularity
- Predicted Popularity
- Audio Feature Analysis

### Music Insights
- Top Songs
- Top Artists
- Top Genres
- Correlation Matrix

### Spotify Search
- Live Spotify Song Search
- Clickable Spotify Links

---

## 🎯 Future Improvements

- Song Recommendation System
- Trend Forecasting
- Artist Popularity Prediction
- Advanced Music Analytics
- Real-Time Spotify Data Integration

---

## 👨‍💻 Author

Pavani

---

## ⭐ Project Highlights

- Analyzed 114,000+ Spotify songs
- Built Machine Learning popularity prediction model
- Created interactive Streamlit dashboard
- Integrated Spotify Web API
- Visualized music trends and insights
