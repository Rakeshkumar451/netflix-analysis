import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# MINIMALIST FIGMA-STYLE STYLING & ANIMATIONS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Keyframe Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Apply animations to sections */
    .main .block-container { animation: fadeIn 0.6s ease-out; }
    [data-testid="stMetric"] { animation: fadeInUp 0.5s ease-out forwards; }
    .stTabs { animation: fadeInUp 0.7s ease-out forwards; }
    
    /* Minimalist Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border-color: #E50914;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1A1A1A;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #666666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #EAEAEA;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 600;
        padding: 12px 0px;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E50914;
    }
    .stTabs [aria-selected="true"] {
        color: #E50914 !important;
        border-bottom: 3px solid #E50914 !important;
        background-color: transparent !important;
    }
    
    /* Card containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #EAEAEA;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        padding: 20px;
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }
    
    /* Insight Boxes */
    .insight-box {
        background-color: #F8F9FA;
        border-left: 4px solid #E50914;
        padding: 15px 20px;
        border-radius: 6px;
        margin-top: 15px;
        font-size: 0.95rem;
        color: #444;
        animation: fadeIn 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Set a clean minimalist theme for charts
sns.set_theme(style="white")
plt.rcParams['figure.facecolor'] = '#FFFFFF'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.edgecolor'] = '#EAEAEA'
plt.rcParams['axes.labelcolor'] = '#666666'
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['text.color'] = '#1A1A1A'

# ─────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    # 1. BASE DATA
    try:
        df = pd.read_csv('data/netflix_titles.csv')
    except FileNotFoundError:
        st.error("Error: 'data/netflix_titles.csv' not found.")
        st.stop()
    except UnicodeDecodeError:
        df = pd.read_csv('data/netflix_titles.csv', encoding='latin-1')
    
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Not Rated')
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    movies_mask = df['type'] == 'Movie'
    df.loc[movies_mask, 'duration_int'] = (
        df.loc[movies_mask, 'duration']
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
    )
    df['title_clean'] = df['title'].str.lower().str.strip()

    # 2. ENRICHED DATA
    try:
        enriched = pd.read_csv('data/netflix_large_dataset_cleaned.csv', encoding='latin-1')
        enriched.columns = enriched.columns.str.lower().str.strip()
        title_col = next((c for c in enriched.columns if 'title' in c), None)
        if title_col:
            enriched['title_clean'] = enriched[title_col].str.lower().str.strip()
            cols_to_keep = ['title_clean']
            for c in enriched.columns:
                if any(k in c for k in ['imdb', 'score', 'rating', 'popularity', 'vote']):
                    cols_to_keep.append(c)
            df = pd.merge(df, enriched[cols_to_keep], on='title_clean', how='left')
    except FileNotFoundError:
        pass

    # 3. ROTTEN TOMATOES
    try:
        rt = pd.read_csv('data/netflix-rotten-tomatoes-metacritic-imdb.csv', encoding='latin-1')
        rt.columns = rt.columns.str.lower().str.strip().str.replace(' ', '_')
        title_col = next((c for c in rt.columns if 'title' in c or 'name' in c), None)
        if title_col:
            rt['title_clean'] = rt[title_col].str.lower().str.strip()
            cols_to_keep = ['title_clean']
            for c in rt.columns:
                if 'rotten' in c or 'metacritic' in c:
                    cols_to_keep.append(c)
            if len(cols_to_keep) > 1:
                df = pd.merge(df, rt[cols_to_keep], on='title_clean', how='left')
    except FileNotFoundError:
        pass

    # 4. VIEWERSHIP
    try:
        global_views = pd.read_csv('data/all-weeks-global.csv', encoding='latin-1')
        global_views.columns = global_views.columns.str.lower().str.strip()
        title_col = next((c for c in global_views.columns if 'title' in c or 'show' in c), None)
        if title_col:
            global_views = global_views.rename(columns={title_col: 'title_clean'})
            global_views['title_clean'] = global_views['title_clean'].str.lower().str.strip()
            hours_col = next((c for c in global_views.columns if 'hours' in c or 'view' in c), None)
            if hours_col:
                views_grouped = global_views.groupby('title_clean')[hours_col].sum().reset_index()
                views_grouped = views_grouped.rename(columns={hours_col: 'total_hours_viewed'})
                df = pd.merge(df, views_grouped, on='title_clean', how='left')
    except FileNotFoundError:
        pass

    return df

with st.spinner("🎬 Loading and merging all datasets..."):
    df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: #1A1A1A; font-size: 1.2rem;'>Filters</h2>", unsafe_allow_html=True)
    
    types = st.multiselect(
        "Content Type",
        options=sorted(df['type'].unique()),
        default=sorted(df['type'].unique())
    )

    valid_years = df['release_year'].dropna().astype(int)
    year_min, year_max = int(valid_years.min()), int(valid_years.max())
    year_range = st.slider(
        "Release Year",
        min_value=year_min,
        max_value=year_max,
        value=(2000, year_max)
    )

    ratings = st.multiselect(
        "Age Rating",
        options=sorted(df['rating'].unique()),
        default=sorted(df['rating'].unique())[:5] # Default to top 5 to prevent empty view
    )

    if st.button("Reset Filters"):
        st.rerun()

# Apply filters
filtered = df[df['type'].isin(types)]
filtered = filtered[
    (filtered['release_year'].isna()) |
    ((filtered['release_year'] >= year_range[0]) & (filtered['release_year'] <= year_range[1]))
]
if ratings:
    filtered = filtered[filtered['rating'].isin(ratings)]

# ─────────────────────────────────────────────
# HEADER & METRICS
# ─────────────────────────────────────────────
st.title("Netflix Content Analytics")
st.markdown(f"Analyzing **{len(df):,}** titles enriched with ratings, viewership, and critic scores.")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Titles", f"{len(filtered):,}")
col2.metric("Movies", f"{(filtered['type'] == 'Movie').sum():,}")
col3.metric("TV Shows", f"{(filtered['type'] == 'TV Show').sum():,}")
col4.metric("Countries", f"{filtered['country'].str.split(', ').explode().nunique():,}")

# Dynamically find the IMDb column
imdb_col = next((c for c in filtered.columns if 'imdb' in c and ('score' in c or 'rating' in c)), None)
if imdb_col:
    avg_imdb = pd.to_numeric(filtered[imdb_col], errors='coerce').mean()
    col5.metric("Avg IMDb Score", f"{avg_imdb:.1f}" if pd.notna(avg_imdb) else "N/A")
else:
    col5.metric("Avg IMDb Score", "N/A")

st.markdown("<br>", unsafe_allow_html=True)

if len(filtered) == 0:
    st.warning("No titles match your filters. Try broadening them in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# TABS SETUP
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Trends", "Geography", "Genres", "Scores & Ratings", "Viewership", "Explore"
])

# ─── TAB 1: OVERVIEW ───
with tab1:
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Movies vs TV Shows")
            type_counts = filtered['type'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   colors=['#E50914', '#DDDDDD'], startangle=90, 
                   wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
                   textprops=dict(color='#1A1A1A', fontsize=11, fontweight='600'))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.subheader("Overview Insight")
            st.markdown("<br>", unsafe_allow_html=True)
            movies_pct = (filtered['type'] == 'Movie').sum() / len(filtered) * 100
            st.markdown(f"""
            <div class='insight-box'>
                Netflix's catalog in view is <b>{movies_pct:.1f}% Movies</b> and <b>{100-movies_pct:.1f}% TV Shows</b>. 
                Movies have historically dominated the platform, but TV content has been growing faster in recent years.
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Top Ratings by Count")
        ratings_data = filtered['rating'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(ratings_data.index, ratings_data.values, color='#E50914', width=0.5)
        for i, v in enumerate(ratings_data.values):
            ax.text(i, v + max(ratings_data.values)*0.01, f'{v:,}', ha='center', fontsize=10, color='#666666')
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class='insight-box'>
            Netflix focuses heavily on <b>mature audiences</b> — TV-MA and TV-14 dominate the catalog. 
            Family-friendly ratings like G and TV-Y form a small fraction.
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: TRENDS ───
with tab2:
    with st.container(border=True):
        st.subheader("Content Added Over the Years")
        yearly = filtered.dropna(subset=['year_added'])
        yearly = yearly[yearly['year_added'] >= 2008]['year_added'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(yearly.index, yearly.values, color='#E50914', linewidth=3)
        ax.fill_between(yearly.index, yearly.values, color='#E50914', alpha=0.1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Titles")
        ax.grid(axis='y', linestyle='--', alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class='insight-box'>
            Netflix's content additions <b>peaked around 2019</b>, then slowed down after 2020 — 
            likely a strategic shift toward quality over quantity, plus pandemic-related production delays.
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Growth Comparison: Movies vs TV Shows")
        fig, ax = plt.subplots(figsize=(12, 4))
        for content_type, color in [('Movie', '#E50914'), ('TV Show', '#999999')]:
            sub = filtered[(filtered['type'] == content_type) & (filtered['year_added'] >= 2008)]
            trend = sub['year_added'].value_counts().sort_index()
            if not trend.empty:
                ax.plot(trend.index, trend.values, color=color, linewidth=2, label=content_type)
        ax.legend(frameon=False)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Titles")
        ax.grid(axis='y', linestyle='--', alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ─── TAB 3: GEOGRAPHY ───
with tab3:
    with st.container(border=True):
        st.subheader("Top 10 Content-Producing Countries")
        countries = filtered['country'].str.split(', ').explode()
        countries = countries[countries != 'Unknown'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(countries.index[::-1], countries.values[::-1], color='#E50914', height=0.5)
        for i, v in enumerate(countries.values[::-1]):
            ax.text(v + max(countries.values)*0.01, i, f'{v:,}', va='center', fontsize=10, color='#666666')
        ax.set_xlabel("Number of Titles")
        ax.grid(axis='x', linestyle='--', alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class='insight-box'>
            The <b>USA, India, and UK</b> dominate Netflix's content library. 
            India's massive presence reflects Netflix's aggressive investment in Bollywood and regional content.
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 4: GENRES ───
with tab4:
    with st.container(border=True):
        st.subheader("Top 10 Genres")
        genres = filtered['listed_in'].str.split(', ').explode().value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(genres.index[::-1], genres.values[::-1], color='#E50914', height=0.5)
        for i, v in enumerate(genres.values[::-1]):
            ax.text(v + max(genres.values)*0.01, i, f'{v:,}', va='center', fontsize=10, color='#666666')
        ax.set_xlabel("Number of Titles")
        ax.grid(axis='x', linestyle='--', alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class='insight-box'>
            <b>Dramas, Comedies, and International Movies</b> are the most common genres. 
            Netflix invests heavily in drama — it drives the most watch time globally.
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Top 10 Longest Movies")
        movies_only = filtered[(filtered['type'] == 'Movie')].dropna(subset=['duration_int'])
        if not movies_only.empty:
            longest = movies_only.nlargest(10, 'duration_int')[['title', 'duration_int']]
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(longest['title'][::-1], longest['duration_int'][::-1], color='#E50914', height=0.5)
            for i, v in enumerate(longest['duration_int'][::-1]):
                ax.text(v + 2, i, f'{int(v)} min', va='center', fontsize=9, color='#666666')
            ax.set_xlabel("Duration (minutes)")
            ax.grid(axis='x', linestyle='--', alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ─── TAB 5: SCORES & RATINGS ───
with tab5:
    with st.container(border=True):
        st.subheader("Critical Reception Analysis")
        
        if imdb_col:
            c1, c2 = st.columns(2)
            pop_col = next((c for c in filtered.columns if 'popularity' in c or 'vote' in c), None)
            
            with c1:
                st.markdown("**IMDb Score vs. Popularity**")
                if pop_col:
                    plot_df = filtered.dropna(subset=[imdb_col, pop_col]).copy()
                    plot_df[imdb_col] = pd.to_numeric(plot_df[imdb_col], errors='coerce')
                    plot_df[pop_col] = pd.to_numeric(plot_df[pop_col], errors='coerce')
                    plot_df = plot_df.dropna(subset=[imdb_col, pop_col])
                    
                    if not plot_df.empty:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.scatterplot(data=plot_df, x=imdb_col, y=pop_col,
                                        hue='type', alpha=0.7, ax=ax,
                                        palette={'Movie': '#E50914', 'TV Show': '#999999'})
                        ax.set_xlabel("IMDb Score")
                        ax.set_ylabel("Popularity")
                        ax.grid(linestyle='--', alpha=0.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.info("Not enough overlapping data for this chart.")
                else:
                    st.info("Popularity column missing.")

            with c2:
                st.markdown("**Top 10 Highest Rated on IMDb**")
                top_rated = filtered[filtered['type'] == 'Movie'].copy()
                top_rated[imdb_col] = pd.to_numeric(top_rated[imdb_col], errors='coerce')
                top_rated = top_rated.dropna(subset=[imdb_col]).nlargest(10, imdb_col)
                
                if not top_rated.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.barh(top_rated['title'][::-1], top_rated[imdb_col][::-1], color='#E50914', height=0.5)
                    for i, v in enumerate(top_rated[imdb_col][::-1]):
                        ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9, color='#666666')
                    ax.set_xlabel("IMDb Score")
                    ax.grid(axis='x', linestyle='--', alpha=0.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
        else:
            st.warning("⚠️ IMDb score data not found. Please check your enriched dataset.")

# ─── TAB 6: VIEWERSHIP ───
with tab6:
    with st.container(border=True):
        st.subheader("Official Viewership Analytics")
        
        if 'total_hours_viewed' in filtered.columns:
            top_viewed = filtered.dropna(subset=['total_hours_viewed']).nlargest(15, 'total_hours_viewed')
            
            if not top_viewed.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(top_viewed['title'][::-1], top_viewed['total_hours_viewed'][::-1], color='#E50914', height=0.5)
                for i, v in enumerate(top_viewed['total_hours_viewed'][::-1]):
                    label = f'{v/1000000:.1f}M' if v > 1000000 else f'{v/1000:.0f}K'
                    ax.text(v + (top_viewed['total_hours_viewed'].max() * 0.01), i, label, va='center', fontsize=9, color='#666666')
                ax.set_xlabel("Total Hours Viewed")
                ax.set_title("Top 15 Most-Watched Titles on Netflix")
                ax.grid(axis='x', linestyle='--', alpha=0.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                st.markdown("""
                <div class='insight-box'>
                    The most-watched titles are often not the highest-rated. 
                    This shows the difference between <b>popularity</b> and <b>critical acclaim</b>.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No viewership data available after merging.")
        else:
            st.warning("⚠️ Viewership dataset missing or column names mismatch.")

# ─── TAB 7: EXPLORE ───
with tab7:
    with st.container(border=True):
        st.subheader("Explore the Catalog")
        
        search = st.text_input("Search by title", placeholder="e.g. Stranger Things, Breaking Bad...")
        
        view = filtered.copy()
        if search:
            view = view[view['title'].str.contains(search, case=False, na=False)]
        
        st.markdown(f"Showing **{len(view):,}** titles")
        
        display_cols = ['title', 'type', 'country', 'release_year', 'rating', 'duration']
        if imdb_col:
            display_cols.append(imdb_col)
        if 'total_hours_viewed' in view.columns:
            display_cols.append('total_hours_viewed')
            
        st.dataframe(
            view[display_cols].rename(columns={
                'title': 'Title', 'type': 'Type', 'country': 'Country',
                'release_year': 'Year', 'rating': 'Rating', 'duration': 'Duration',
                imdb_col: 'IMDb', 'total_hours_viewed': 'Hours Viewed'
            }),
            width='stretch',
            height=400
        )
        
        csv = view[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Filtered Data as CSV",
            data=csv,
            file_name='netflix_filtered.csv',
            mime='text/csv'
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999999; font-size: 0.8rem; border-top: 1px solid #EAEAEA; padding-top: 20px;'>
    Built with Python, Pandas, Matplotlib & Streamlit<br>
    Data: Kaggle Netflix Movies and TV Shows Dataset + Enriched Sources
</div>
""", unsafe_allow_html=True)