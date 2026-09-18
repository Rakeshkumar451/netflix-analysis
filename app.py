import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
# MINIMALIST STYLING (Your Exact Version)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean up padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Minimalist Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        padding-bottom: 10px;
    }
    
    /* Insight Box Styling */
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #E50914;
        padding: 15px 20px;
        border-radius: 4px;
        margin-top: 15px;
        font-size: 0.95rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Set a clean minimalist theme for charts
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ─────────────────────────────────────────────
# SMART DATA LOADING HELPER (The Fix)
# ─────────────────────────────────────────────
def safe_read_csv(filepath):
    """
    Reads a CSV safely:
    - Skips if file doesn't exist or is empty (0 bytes) to prevent EmptyDataError
    - Tries multiple encodings (utf-8, latin-1, cp1252) to prevent UnicodeDecodeError
    - Standardizes column names (lowercase, strip, replace spaces with underscores)
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None
    
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            # Standardize columns immediately!
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None

# ─────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    # 1. BASE DATA
    df = safe_read_csv('data/netflix_titles.csv')
    if df is None:
        st.error("Error: 'data/netflix_titles.csv' not found or empty.")
        st.stop()
    
    # Clean base data
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    if 'rating' not in df.columns:
        df['rating'] = 'Not Rated'
    else:
        df['rating'] = df['rating'].fillna('Not Rated')
    # Force to plain strings so sorted()/unique() never chokes on a
    # mixed-type column (e.g. stray numeric ratings in the source CSV)
    df['rating'] = df['rating'].astype(str)
    if 'listed_in' in df.columns:
        df['listed_in'] = df['listed_in'].fillna('Unknown')
        
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    # Parse duration (movies only)
    movies_mask = df['type'] == 'Movie'
    df.loc[movies_mask, 'duration_int'] = (
        df.loc[movies_mask, 'duration']
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
    )
    
    # Create a clean key for merging
    df['title_clean'] = df['title'].str.lower().str.strip()

    # ── Diagnostics: track exactly what happened to each optional source ──
    load_report = []

    def _report(name, status, detail=""):
        load_report.append((name, status, detail))

    # 2. ENRICHED DATA (IMDb & TMDb)
    enriched_path = 'data/netflix_large_dataset_cleaned.csv'
    enriched = safe_read_csv(enriched_path)
    if enriched is None:
        _report(enriched_path, "missing/unreadable",
                "file not found, empty, or unreadable in every tried encoding")
    else:
        title_col = next((c for c in enriched.columns if 'title' in c), None)
        if not title_col:
            _report(enriched_path, "loaded but no title column",
                    f"columns seen: {list(enriched.columns)}")
        else:
            enriched['title_clean'] = enriched[title_col].str.lower().str.strip()
            cols_to_keep = ['title_clean']
            for c in enriched.columns:
                if any(k in c for k in ['imdb', 'score', 'vote', 'popularity']) and c not in df.columns:
                    cols_to_keep.append(c)
            if len(cols_to_keep) == 1:
                _report(enriched_path, "loaded but no score/imdb/vote/popularity columns found",
                        f"columns seen: {list(enriched.columns)}")
            else:
                before = df['title_clean'].isin(enriched['title_clean']).sum()
                df = pd.merge(df, enriched[cols_to_keep], on='title_clean', how='left')
                _report(enriched_path, "merged", f"{before:,} titles matched out of {len(df):,}")

    # 3. ROTTEN TOMATOES & METACRITIC
    rt_path = 'data/netflix-rotten-tomatoes-metacritic-imdb.csv'
    rt = safe_read_csv(rt_path)
    if rt is None:
        _report(rt_path, "missing/unreadable", "")
    else:
        title_col = next((c for c in rt.columns if 'title' in c or 'name' in c), None)
        if not title_col:
            _report(rt_path, "loaded but no title/name column", f"columns seen: {list(rt.columns)}")
        else:
            rt['title_clean'] = rt[title_col].str.lower().str.strip()
            cols_to_keep = ['title_clean']
            for c in rt.columns:
                if ('rotten' in c or 'metacritic' in c) and c not in df.columns:
                    cols_to_keep.append(c)
            if len(cols_to_keep) == 1:
                _report(rt_path, "loaded but no rotten/metacritic columns found",
                        f"columns seen: {list(rt.columns)}")
            else:
                before = df['title_clean'].isin(rt['title_clean']).sum()
                df = pd.merge(df, rt[cols_to_keep], on='title_clean', how='left')
                _report(rt_path, "merged", f"{before:,} titles matched out of {len(df):,}")

    # 4. OFFICIAL VIEWERSHIP (Global)
    gv_path = 'data/all-weeks-global.csv'
    global_views = safe_read_csv(gv_path)
    if global_views is None:
        _report(gv_path, "missing/unreadable", "")
    else:
        title_col = next((c for c in global_views.columns if 'title' in c or 'show' in c), None)
        if not title_col:
            _report(gv_path, "loaded but no title/show column", f"columns seen: {list(global_views.columns)}")
        else:
            global_views = global_views.rename(columns={title_col: 'title_clean'})
            global_views['title_clean'] = global_views['title_clean'].str.lower().str.strip()
            hours_col = next((c for c in global_views.columns if 'hours' in c or 'view' in c), None)
            if not hours_col:
                _report(gv_path, "loaded but no hours/view column", f"columns seen: {list(global_views.columns)}")
            else:
                views_grouped = global_views.groupby('title_clean')[hours_col].sum().reset_index()
                views_grouped = views_grouped.rename(columns={hours_col: 'total_hours_viewed'})
                before = df['title_clean'].isin(views_grouped['title_clean']).sum()
                df = pd.merge(df, views_grouped, on='title_clean', how='left')
                _report(gv_path, "merged", f"{before:,} titles matched out of {len(df):,}")

    # 5. CONTENT INTELLIGENCE
    intel_path = 'data/netflix_content_intelligence_combined.csv'
    intel = safe_read_csv(intel_path)
    if intel is None:
        _report(intel_path, "missing/unreadable", "")
    else:
        title_col = next((c for c in intel.columns if 'title' in c), None)
        if not title_col:
            _report(intel_path, "loaded but no title column", f"columns seen: {list(intel.columns)}")
        else:
            intel['title_clean'] = intel[title_col].str.lower().str.strip()
            cols_to_keep = ['title_clean']
            for c in intel.columns:
                if any(k in c for k in ['sentiment', 'popularity', 'content_type']) and c not in df.columns:
                    cols_to_keep.append(c)
            if len(cols_to_keep) == 1:
                _report(intel_path, "loaded but no sentiment/popularity/content_type columns found",
                        f"columns seen: {list(intel.columns)}")
            else:
                before = df['title_clean'].isin(intel['title_clean']).sum()
                df = pd.merge(df, intel[cols_to_keep], on='title_clean', how='left')
                _report(intel_path, "merged", f"{before:,} titles matched out of {len(df):,}")

    df.attrs['load_report'] = load_report
    return df

with st.spinner("Loading and merging all datasets..."):
    df = load_data()

# Always-visible diagnostics for the optional enrichment sources, so a
# missing/mismatched file in the deployed repo is obvious at a glance
# instead of showing up only as a downstream "No score data found" warning.
with st.sidebar.expander("📦 Data source status", expanded=False):
    for name, status, detail in df.attrs.get('load_report', []):
        icon = "✅" if status == "merged" else "⚠️"
        st.markdown(f"{icon} **{name}**  \n{status}" + (f"  \n_{detail}_" if detail else ""))

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    
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
        default=[]
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
st.title("🎬 Netflix Content Analytics")
st.markdown(f"Analyzing **{len(df):,}** titles enriched with ratings, viewership, and critic scores.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Titles", f"{len(filtered):,}")
col2.metric("Movies", f"{(filtered['type'] == 'Movie').sum():,}")
col3.metric("TV Shows", f"{(filtered['type'] == 'TV Show').sum():,}")
col4.metric("Countries", f"{filtered['country'].str.split(', ').explode().nunique():,}")

# Dynamically find the imdb column to fix the "N/A" issue
imdb_col = next((c for c in filtered.columns if 'imdb' in c and ('score' in c or 'rating' in c)), None)
if imdb_col:
    avg_imdb = pd.to_numeric(filtered[imdb_col], errors='coerce').mean()
    col5.metric("Avg IMDb Score", f"{avg_imdb:.1f}" if pd.notna(avg_imdb) else "N/A")
else:
    col5.metric("Avg IMDb Score", "N/A")

st.divider()

# Guard: empty dataframe
if len(filtered) == 0:
    st.warning("No titles match your filters. Try broadening them in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# TABS SETUP
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", 
    "📈 Trends", 
    "🌍 Geography", 
    "🎭 Genres", 
    "⭐ Scores & Ratings", 
    "📺 Viewership",
    "🔍 Explore"
])

# ─── TAB 1: OVERVIEW ───
with tab1:
    st.subheader("Content Distribution")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        type_counts = filtered['type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               colors=['#E50914', '#4A4A4A'], startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title("Movies vs TV Shows")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        movies_pct = (filtered['type'] == 'Movie').sum() / len(filtered) * 100
        st.markdown(f"""
        <div class='insight-box'>
            Netflix's catalog in view is <b>{movies_pct:.1f}% Movies</b> and <b>{100-movies_pct:.1f}% TV Shows</b>. 
            Movies have historically dominated, but TV content has been growing faster in recent years.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Top Ratings by Count")
    ratings_data = filtered['rating'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Fixed FutureWarning by using hue
    sns.barplot(x=ratings_data.index, y=ratings_data.values, hue=ratings_data.index, palette="Reds_r", ax=ax, legend=False)
    for i, v in enumerate(ratings_data.values):
        ax.text(i, v + max(ratings_data.values)*0.01, f'{v:,}', ha='center', fontsize=9)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
    <div class='insight-box'>
        Netflix focuses heavily on <b>mature audiences</b> — TV-MA and TV-14 dominate the catalog. 
        Family-friendly ratings like G and TV-Y form a small fraction.
    </div>
    """, unsafe_allow_html=True)


# ─── TAB 2: TRENDS ───
with tab2:
    st.subheader("Content Added Over the Years")
    
    yearly = filtered.dropna(subset=['year_added'])
    yearly = yearly[yearly['year_added'] >= 2008]['year_added'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yearly.index, yearly.values, color='#E50914', linewidth=2.5, marker='o')
    ax.fill_between(yearly.index, yearly.values, color='#E50914', alpha=0.1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Titles")
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
    <div class='insight-box'>
        Netflix's content additions <b>peaked around 2019</b>, then slowed down after 2020 — 
        likely a strategic shift toward quality over quantity, plus pandemic-related production delays.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Growth Comparison: Movies vs TV Shows")
    fig, ax = plt.subplots(figsize=(12, 5))
    for content_type, color in [('Movie', '#E50914'), ('TV Show', '#4A4A4A')]:
        sub = filtered[(filtered['type'] == content_type) & (filtered['year_added'] >= 2008)]
        trend = sub['year_added'].value_counts().sort_index()
        if not trend.empty:
            ax.plot(trend.index, trend.values, color=color, linewidth=2, marker='o', label=content_type)
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Titles")
    st.pyplot(fig)
    plt.close(fig)


# ─── TAB 3: GEOGRAPHY ───
with tab3:
    st.subheader("Top 10 Content-Producing Countries")
    
    countries = filtered['country'].str.split(', ').explode()
    countries = countries[countries != 'Unknown'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=countries.values, y=countries.index, hue=countries.index, palette="Reds_r", ax=ax, legend=False)
    for i, v in enumerate(countries.values):
        ax.text(v + max(countries.values)*0.01, i, f'{v:,}', va='center', fontsize=10)
    ax.set_xlabel("Number of Titles")
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
    <div class='insight-box'>
        The <b>USA, India, and UK</b> dominate Netflix's content library. 
        India's massive presence reflects Netflix's aggressive investment in Bollywood and regional content.
    </div>
    """, unsafe_allow_html=True)


# ─── TAB 4: GENRES ───
with tab4:
    st.subheader("Top 10 Genres")
    
    genres = filtered['listed_in'].str.split(', ').explode().value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=genres.values, y=genres.index, hue=genres.index, palette="Reds_r", ax=ax, legend=False)
    for i, v in enumerate(genres.values):
        ax.text(v + max(genres.values)*0.01, i, f'{v:,}', va='center', fontsize=10)
    ax.set_xlabel("Number of Titles")
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("""
    <div class='insight-box'>
        <b>Dramas, Comedies, and International Movies</b> are the most common genres. 
        Netflix invests heavily in drama — it drives the most watch time globally.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Top 10 Longest Movies")
    movies_only = filtered[(filtered['type'] == 'Movie')].dropna(subset=['duration_int'])
    if not movies_only.empty:
        longest = movies_only.nlargest(10, 'duration_int')[['title', 'duration_int']]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=longest['duration_int'], y=longest['title'], hue=longest['title'], palette="Reds_r", ax=ax, legend=False)
        for i, v in enumerate(longest['duration_int']):
            ax.text(v + 2, i, f'{int(v)} min', va='center', fontsize=9)
        ax.set_xlabel("Duration (minutes)")
        st.pyplot(fig)
        plt.close(fig)


# ─── TAB 5: SCORES & RATINGS ───
with tab5:
    st.subheader("⭐ Critical Reception Analysis")
    
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
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(data=plot_df, x=imdb_col, y=pop_col,
                                    hue='type', alpha=0.6, ax=ax,
                                    palette={'Movie': '#E50914', 'TV Show': '#4A4A4A'})
                    ax.set_xlabel("IMDb Score")
                    ax.set_ylabel("Popularity")
                    st.pyplot(fig)
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
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=top_rated[imdb_col], y=top_rated['title'], hue=top_rated['title'], palette="Reds_r", ax=ax, legend=False)
                for i, v in enumerate(top_rated[imdb_col]):
                    ax.text(v + 0.1, i, f'{v:.1f}', va='center', fontsize=9)
                ax.set_xlabel("IMDb Score")
                st.pyplot(fig)
                plt.close(fig)
        
        st.divider()
        st.markdown("**Compare Critics: IMDb vs Rotten Tomatoes**")
        if 'rotten_tomatoes' in filtered.columns:
            compare_df = filtered.dropna(subset=[imdb_col, 'rotten_tomatoes']).copy()
            if not compare_df.empty:
                compare_df['rotten_tomatoes'] = compare_df['rotten_tomatoes'].astype(str).str.replace('%', '')
                compare_df['rotten_tomatoes'] = pd.to_numeric(compare_df['rotten_tomatoes'], errors='coerce')
                compare_df = compare_df.dropna(subset=['rotten_tomatoes'])

                if not compare_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.scatter(compare_df['rotten_tomatoes'], compare_df[imdb_col] * 10, 
                               alpha=0.5, color='#E50914')
                    ax.set_xlabel("Rotten Tomatoes Score (%)")
                    ax.set_ylabel("IMDb Score (scaled to 100)")
                    ax.set_title("IMDb vs Rotten Tomatoes Score Comparison")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Not enough valid data to compare scores.")
            else:
                st.info("No overlapping data for score comparison.")
    else:
        st.warning("⚠️ No score data found. Please ensure 'netflix_large_dataset_cleaned.csv' is loaded correctly.")
        with st.expander("🔧 Debug: See loaded columns"):
            st.write(list(df.columns))

# ─── TAB 6: VIEWERSHIP ───
with tab6:
    st.subheader("📺 Official Viewership Analytics")
    
    if 'total_hours_viewed' in filtered.columns:
        top_viewed = filtered.dropna(subset=['total_hours_viewed']).nlargest(15, 'total_hours_viewed')
        
        if not top_viewed.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_viewed['total_hours_viewed'], y=top_viewed['title'], hue=top_viewed['title'], palette="Reds_r", ax=ax, legend=False)
            for i, v in enumerate(top_viewed['total_hours_viewed']):
                label = f'{v/1000000:.1f}M' if v > 1000000 else f'{v/1000:.0f}K'
                ax.text(v + (top_viewed['total_hours_viewed'].max() * 0.01), i, label, va='center', fontsize=9)
            ax.set_xlabel("Total Hours Viewed")
            ax.set_title("Top 15 Most-Watched Titles on Netflix")
            st.pyplot(fig)
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
        st.warning("⚠️ Viewership dataset missing or column names mismatch. Please ensure 'all-weeks-global.csv' is loaded correctly.")

# ─── TAB 7: EXPLORE ───
with tab7:
    st.subheader("🔍 Explore the Catalog")
    
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
        height=500
    )
    
    csv = view[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name='netflix_filtered.csv',
        mime='text/csv'
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    Built with Python, Pandas, Matplotlib & Streamlit<br>
    Data: Kaggle Netflix Movies and TV Shows Dataset + Enriched Sources
</div>
""", unsafe_allow_html=True)