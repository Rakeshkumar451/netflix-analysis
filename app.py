import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Setup ───
st.set_page_config(
    page_title="Netflix Analysis",
    page_icon="🎬",
    layout="wide"
)

sns.set_style("whitegrid")

# ─── Load Data (cached for speed) ───
@st.cache_data
def load_data():
    df = pd.read_csv('data/netflix_titles.csv')
    
    # Cleaning
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Not Rated')
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    
    return df

df = load_data()

# ─── Header ───
st.title("🎬 Netflix Content Analysis")
st.markdown("An interactive dashboard exploring Netflix's movies and TV shows catalog.")
st.markdown("---")

# ─── Key Metrics Row ───
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", f"{len(df):,}")
col2.metric("Movies", f"{(df['type'] == 'Movie').sum():,}")
col3.metric("TV Shows", f"{(df['type'] == 'TV Show').sum():,}")
col4.metric("Countries", f"{df['country'].nunique():,}")

st.markdown("---")

# ─── Q1: Movies vs TV Shows ───
st.header("1️⃣ Movies vs TV Shows")
fig, ax = plt.subplots(figsize=(8, 5))
df['type'].value_counts().plot(kind='bar', color=['crimson', 'steelblue'], ax=ax)
ax.set_title('Distribution of Content Type')
ax.set_xlabel('Type')
ax.set_ylabel('Count')
plt.xticks(rotation=0)
st.pyplot(fig)

st.info(f"💡 **Insight:** Netflix's library is split **{(df['type'] == 'Movie').sum() / len(df) * 100:.1f}% Movies** and **{(df['type'] == 'TV Show').sum() / len(df) * 100:.1f}% TV Shows**.")

st.markdown("---")

# ─── Q2: Content Growth ───
st.header("2️⃣ Content Growth Over the Years")
yearly = df['year_added'].value_counts().sort_index()
yearly = yearly[yearly.index >= 2008]  # Filter outliers

fig, ax = plt.subplots(figsize=(12, 5))
yearly.plot(kind='line', marker='o', color='crimson', ax=ax)
ax.set_title('Content Added to Netflix Over the Years')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Titles')
ax.grid(True)
st.pyplot(fig)

st.info("💡 **Insight:** Netflix's content additions **peaked around 2019** and slowed down after 2020.")

st.markdown("---")

# ─── Q3: Top Countries ───
st.header("3️⃣ Top Content-Producing Countries")
countries = df['country'].str.split(', ').explode()
top_countries = countries.value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
top_countries.plot(kind='barh', color='teal', ax=ax)
ax.set_title('Top 10 Content-Producing Countries')
ax.set_xlabel('Number of Titles')
ax.invert_yaxis()
st.pyplot(fig)

st.info("💡 **Insight:** The **USA, India, and UK** dominate Netflix's content production.")

st.markdown("---")

# ─── Q4: Top Genres ───
st.header("4️⃣ Most Popular Genres")
genres = df['listed_in'].str.split(', ').explode()
top_genres = genres.value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
top_genres.plot(kind='barh', color='purple', ax=ax)
ax.set_title('Top 10 Genres on Netflix')
ax.set_xlabel('Number of Titles')
ax.invert_yaxis()
st.pyplot(fig)

st.info("💡 **Insight:** **Dramas, Comedies, and International Movies** are the most common genres.")

st.markdown("---")

# ─── Q5: Ratings ───
st.header("5️⃣ Content Ratings Distribution")
top_ratings = df['rating'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
top_ratings.plot(kind='bar', color='orange', ax=ax)
ax.set_title('Content Ratings Distribution')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig)

st.info("💡 **Insight:** Netflix focuses heavily on **mature audiences** (TV-MA, TV-14).")

# ─── Footer ───
st.markdown("---")
st.markdown("### 📊 Built with Python, Pandas, Matplotlib & Streamlit")
st.markdown("*Data source: [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)*")