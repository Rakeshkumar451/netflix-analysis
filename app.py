import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
# CUSTOM CSS - THE MAGIC
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Cinematic dark background */
.stApp {
    background: radial-gradient(circle at 15% 0%, #1A0B0F 0%, #0B0B0F 45%, #050507 100%);
    color: #F5F5F7;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}

/* Reduce top padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0B0F 0%, #150810 100%);
    border-right: 1px solid rgba(229, 9, 20, 0.15);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #FF4B4B;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.5px;
}

/* Metric cards - glassmorphism */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 22px 20px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #E50914, #FF6B6B, #E50914);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
[data-testid="stMetric"]:hover {
    transform: translateY(-6px);
    border-color: rgba(229, 9, 20, 0.5);
    box-shadow: 0 16px 48px rgba(229, 9, 20, 0.20);
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800;
    font-size: 2.2rem !important;
    letter-spacing: -1px;
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stMetricLabel"] {
    color: #8B8B9A !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255, 255, 255, 0.03);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 22px;
    color: #8B8B9A;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.25s ease;
    border: none;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #FF6B6B;
    background: rgba(229, 9, 20, 0.08);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #E50914, #B20710) !important;
    color: white !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.4);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #E50914, #B20710);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 26px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(229, 9, 20, 0.45);
    filter: brightness(1.1);
}

/* Multiselect / select / slider accents */
[data-baseweb="tag"] {
    background: linear-gradient(135deg, #E50914, #B20710) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0B0B0F; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#E50914, #B20710);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #FF4B4B; }

/* Fade-in */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
.main .block-container { animation: fadeIn 0.7s ease-out; }

/* Custom header classes */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0;
    background: linear-gradient(135deg, #FFFFFF 0%, #FFFFFF 40%, #E50914 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #8B8B9A;
    font-size: 1.1rem;
    margin-top: 12px;
    font-weight: 400;
}
.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(229, 9, 20, 0.12);
    color: #FF6B6B;
    padding: 7px 16px;
    border-radius: 24px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(229, 9, 20, 0.25);
    letter-spacing: 0.3px;
}
.badge.muted {
    background: rgba(255, 255, 255, 0.05);
    color: #F5F5F7;
    border: 1px solid rgba(255, 255, 255, 0.10);
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin: 0 0 8px 0;
    color: #FFFFFF;
}
.insight-card {
    background: linear-gradient(135deg, rgba(229, 9, 20, 0.10), rgba(229, 9, 20, 0.02));
    border-left: 4px solid #E50914;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 18px 0 4px 0;
    display: flex;
    gap: 14px;
    align-items: flex-start;
    animation: fadeIn 0.6s ease-out;
}
.insight-icon { font-size: 1.4rem; line-height: 1; }
.insight-text {
    color: #E5E5EA;
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0;
}
.insight-text b { color: #FF6B6B; font-weight: 700; }

.chart-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 24px 24px 12px 24px;
    margin-top: 10px;
    transition: border-color 0.3s ease;
}
.chart-card:hover { border-color: rgba(229, 9, 20, 0.30); }

.footer {
    text-align: center;
    color: #555560;
    font-size: 0.85rem;
    margin-top: 60px;
    padding-top: 30px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('data/netflix_titles.csv')

    # Clean
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Not Rated')
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

    return df


with st.spinner("🎬 Loading Netflix library..."):
    df = load_data()


# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.markdown("<p style='color:#8B8B9A; font-size:0.85rem; margin-top:-10px;'>Refine the dataset</p>", unsafe_allow_html=True)

    # Type filter
    types = st.multiselect(
        "📺 Content Type",
        options=sorted(df['type'].unique()),
        default=sorted(df['type'].unique())
    )

    # Year range
    valid_years = df['release_year'].dropna().astype(int)
    year_min, year_max = int(valid_years.min()), int(valid_years.max())
    year_range = st.slider(
        "📅 Release Year",
        min_value=year_min,
        max_value=year_max,
        value=(2000, year_max)
    )

    # Rating filter
    ratings = st.multiselect(
        "🔞 Age Rating",
        options=sorted(df['rating'].unique()),
        default=[]
    )

    st.markdown("---")

    # Reset button (visual only - refresh required)
    if st.button("🔄 Reset All Filters"):
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='color:#555560; font-size:0.75rem; text-align:center;'>Data source: Kaggle Netflix Dataset</p>", unsafe_allow_html=True)


# Apply filters
filtered = df[df['type'].isin(types)]
filtered = filtered[
    (filtered['release_year'].isna()) |
    ((filtered['release_year'] >= year_range[0]) & (filtered['release_year'] <= year_range[1]))
]
if ratings:
    filtered = filtered[filtered['rating'].isin(ratings)]


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown(f"""
<h1 class='hero-title'>Netflix Analytics</h1>
<p class='hero-sub'>A data-driven deep dive into {len(df):,} titles across the Netflix catalog.</p>
<div class='badge-row'>
    <span class='badge'>📊 Live Dashboard</span>
    <span class='badge muted'>🎯 {len(filtered):,} titles in view</span>
    <span class='badge muted'>🌍 {df['country'].nunique()} countries</span>
    <span class='badge muted'>📅 {year_min} – {year_max}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Titles", f"{len(filtered):,}")
c2.metric("Movies", f"{(filtered['type'] == 'Movie').sum():,}")
c3.metric("TV Shows", f"{(filtered['type'] == 'TV Show').sum():,}")
c4.metric("Countries", f"{filtered['country'].str.split(', ').explode().nunique():,}")

st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHART HELPER
# ─────────────────────────────────────────────
def styled_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color((1, 1, 1, 0.15))
    ax.spines['left'].set_color((1, 1, 1, 0.15))
    ax.tick_params(colors='#8B8B9A', labelsize=10)
    ax.xaxis.label.set_color('#8B8B9A')
    ax.yaxis.label.set_color('#8B8B9A')
    ax.title.set_color('#FFFFFF')
    ax.grid(color=(1, 1, 1, 0.06), linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    return fig, ax


def insight(text):
    st.markdown(f"""
    <div class='insight-card'>
        <span class='insight-icon'>💡</span>
        <p class='insight-text'>{text}</p>
    </div>
    """, unsafe_allow_html=True)


# Guard: empty dataframe
if len(filtered) == 0:
    st.warning("😕 No titles match your filters. Try broadening them in the sidebar.")
    st.stop()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Trends", "🌍 Geography", "🎭 Genres", "🔍 Explore"
])

# ─── TAB 1: OVERVIEW ───
with tab1:
    st.markdown("<h2 class='section-title'>Movies vs TV Shows</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        type_counts = filtered['type'].value_counts()
        fig, ax = styled_fig(7, 6)
        wedges, texts, autotexts = ax.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#E50914', '#FF6B6B'],
            wedgeprops=dict(width=0.42, edgecolor='#0B0B0F', linewidth=3),
            textprops=dict(color='#F5F5F7', fontsize=12, fontweight='600')
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontweight('bold')
            at.set_fontsize(13)
        ax.text(0, 0, f"{len(filtered):,}\nTitles", ha='center', va='center',
                color='#FFFFFF', fontsize=16, fontweight='bold', family='sans-serif')
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        movies_pct = (filtered['type'] == 'Movie').sum() / len(filtered) * 100
        tv_pct = 100 - movies_pct
        insight(f"Netflix's catalog in view is <b>{movies_pct:.1f}% Movies</b> and <b>{tv_pct:.1f}% TV Shows</b>. Movies have historically dominated, but TV content has been growing faster in recent years.")

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Ratings Distribution</h2>", unsafe_allow_html=True)

    ratings_data = filtered['rating'].value_counts().head(10)
    fig, ax = styled_fig(12, 5)
    bars = ax.bar(ratings_data.index, ratings_data.values,
                  color=['#E50914' if i == 0 else '#8B1A1F' for i in range(len(ratings_data))],
                  edgecolor='none', width=0.65)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(ratings_data.values)*0.01,
                f'{int(h):,}', ha='center', color='#F5F5F7', fontsize=9, fontweight='600')
    ax.set_title('Top 10 Ratings by Count', fontsize=14, pad=15, loc='left')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    insight("Netflix focuses heavily on <b>mature audiences</b> — TV-MA and TV-14 dominate the catalog. Family-friendly ratings like G and TV-Y form a small fraction.")


# ─── TAB 2: TRENDS ───
with tab2:
    st.markdown("<h2 class='section-title'>Content Added Over the Years</h2>", unsafe_allow_html=True)

    yearly = filtered.dropna(subset=['year_added'])
    yearly = yearly[yearly['year_added'] >= 2008]['year_added'].value_counts().sort_index()

    fig, ax = styled_fig(12, 5.5)
    ax.plot(yearly.index, yearly.values, color='#E50914', linewidth=2.8,
            marker='o', markersize=6, markerfacecolor='#FF6B6B',
            markeredgecolor='#0B0B0F', markeredgewidth=1.5)
    ax.fill_between(yearly.index, yearly.values, color='#E50914', alpha=0.15)
    ax.set_title('Titles Added to Netflix Per Year', fontsize=14, pad=15, loc='left')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Titles')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    insight("Netflix's content additions <b>peaked around 2019</b>, then slowed down after 2020 — likely a strategic shift toward quality over quantity, plus pandemic-related production delays.")

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Movies vs TV Shows — Growth Comparison</h2>", unsafe_allow_html=True)

    fig, ax = styled_fig(12, 5.5)
    for content_type, color in [('Movie', '#E50914'), ('TV Show', '#FF6B6B')]:
        sub = filtered[(filtered['type'] == content_type) & (filtered['year_added'] >= 2008)]
        trend = sub['year_added'].value_counts().sort_index()
        if not trend.empty:
            ax.plot(trend.index, trend.values, color=color, linewidth=2.4,
                    marker='o', markersize=5, label=content_type)
    ax.legend(frameon=False, labelcolor='#F5F5F7', fontsize=11, loc='upper left')
    ax.set_title('Content Growth by Type', fontsize=14, pad=15, loc='left')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Titles')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─── TAB 3: GEOGRAPHY ───
with tab3:
    st.markdown("<h2 class='section-title'>Top 10 Content-Producing Countries</h2>", unsafe_allow_html=True)

    countries = filtered['country'].str.split(', ').explode()
    countries = countries[countries != 'Unknown'].value_counts().head(10)

    fig, ax = styled_fig(12, 6)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(countries)))[::-1]
    bars = ax.barh(countries.index[::-1], countries.values[::-1],
                   color=colors, edgecolor='none', height=0.7)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(countries.values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(w):,}', va='center', color='#F5F5F7', fontsize=10, fontweight='600')
    ax.set_title('Top 10 Countries by Number of Titles', fontsize=14, pad=15, loc='left')
    ax.set_xlabel('Number of Titles')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    insight("The <b>USA, India, and UK</b> dominate Netflix's content library. India's massive presence reflects Netflix's aggressive investment in Bollywood and regional content.")


# ─── TAB 4: GENRES ───
with tab4:
    st.markdown("<h2 class='section-title'>Top 10 Genres</h2>", unsafe_allow_html=True)

    genres = filtered['listed_in'].str.split(', ').explode().value_counts().head(10)

    fig, ax = styled_fig(12, 6)
    colors = plt.cm.copper(np.linspace(0.3, 0.9, len(genres)))[::-1]
    bars = ax.barh(genres.index[::-1], genres.values[::-1],
                   color=colors, edgecolor='none', height=0.7)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(genres.values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{int(w):,}', va='center', color='#F5F5F7', fontsize=10, fontweight='600')
    ax.set_title('Top 10 Genres on Netflix', fontsize=14, pad=15, loc='left')
    ax.set_xlabel('Number of Titles')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    insight("<b>Dramas, Comedies, and International Movies</b> are the most common genres. Netflix invests heavily in drama — it drives the most watch time globally.")

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Top 10 Longest Movies</h2>", unsafe_allow_html=True)

    movies_only = filtered[(filtered['type'] == 'Movie')].dropna(subset=['duration_int'])
    if not movies_only.empty:
        longest = movies_only.nlargest(10, 'duration_int')[['title', 'duration_int', 'release_year']]

        fig, ax = styled_fig(12, 6)
        bars = ax.barh(longest['title'][::-1], longest['duration_int'][::-1],
                       color='#E50914', edgecolor='none', height=0.7)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 3, bar.get_y() + bar.get_height()/2,
                    f'{int(w)} min', va='center', color='#F5F5F7', fontsize=10, fontweight='600')
        ax.set_title('Longest Movies (in minutes)', fontsize=14, pad=15, loc='left')
        ax.set_xlabel('Duration (minutes)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─── TAB 5: EXPLORE ───
with tab5:
    st.markdown("<h2 class='section-title'>🔍 Explore the Catalog</h2>", unsafe_allow_html=True)

    search = st.text_input("Search by title", placeholder="e.g. Stranger Things, Breaking Bad...")

    view = filtered.copy()
    if search:
        view = view[view['title'].str.contains(search, case=False, na=False)]

    st.markdown(f"<p style='color:#8B8B9A; font-size:0.9rem;'>Showing <b style='color:#FF6B6B;'>{len(view):,}</b> titles</p>", unsafe_allow_html=True)

    display_cols = ['title', 'type', 'country', 'release_year', 'rating', 'duration']
    st.dataframe(
        view[display_cols].rename(columns={
            'title': 'Title', 'type': 'Type', 'country': 'Country',
            'release_year': 'Year', 'rating': 'Rating', 'duration': 'Duration'
        }),
        use_container_width=True,
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
st.markdown("""
<div class='footer'>
    Built with ❤️ using <b style='color:#E50914;'>Streamlit</b>, Pandas, Matplotlib & Seaborn<br>
    <span style='font-size:0.75rem;'>Data: Kaggle Netflix Movies and TV Shows Dataset</span>
</div>
""", unsafe_allow_html=True)