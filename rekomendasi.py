import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Product Recommendation")

st.markdown("""
<style>
    /* 1. Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* 2. Custom Font Global */
    html, body, .stApp {
        font-family: 'Poppins', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, label, input, textarea, select, .stMarkdown {
        font-family: 'Poppins', sans-serif !important;
    }

    /* 3. Background */
    .stApp {
        background-color: #ffffff; 
    }

    /* 4. Styling Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1B5E20; 
        color: white !important;
    }
    
    /* Teks Sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #e8f5e9 !important; 
    }

    section[data-testid="stSidebar"] svg,
    section[data-testid="stSidebar"] svg path {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    
    button[kind="header"] {
        color: #ffffff !important;
    }

    /* 5. Container Style */
    div.block-container {
        padding-top: 2rem;
    }

    /* 6. Input Fields */
    .stTextInput > div > div, .stTextArea > div > div {
        background-color: #f8f9fa !important; 
        border-color: #e0e0e0 !important;
        color: #333 !important;
    }
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: #43A047 !important; 
        box-shadow: 0 0 0 2px rgba(67, 160, 71, 0.2) !important;
        background-color: #ffffff !important;
    }

    /* 7. Table Style */
    .custom-table {
        width: 100%;
        border-collapse: separate; 
        border-spacing: 0;
        margin: 15px 0;
        border-radius: 10px;
        overflow: hidden; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .custom-table thead tr {
        background-color: #1B5E20 !important; 
        color: #ffffff !important; 
        text-align: left !important;
    }
    .custom-table th, .custom-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #f0f0f0;
    }
    .custom-table tbody tr {
        border-bottom: 1px solid #dddddd;
        background-color: #ffffff;
        color: #333333; 
    }
    .custom-table tbody tr:nth-of-type(even) {
        background-color: #f9f9f9; 
    }
    .custom-table tbody tr:hover {
        background-color: #e8f5e9; 
        cursor: default;
    }
    .custom-table tbody th, .custom-table thead th:first-child, .custom-table tbody td:first-child {
        display: none;
    }

    /* 8. TAB STYLING */
    div[data-baseweb="tab-highlight"] { visibility: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #f1f3f4; 
        border-radius: 8px 8px 0 0;
        padding: 0 20px;
        border: 1px solid transparent;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #43A047; 
        border-bottom: 3px solid #ffffff; 
        color: #1B5E20;
        font-weight: 600;
        box-shadow: 0 -5px 10px rgba(0,0,0,0.02);
    }

    div.stButton > button, 
    div.stButton > button p {
        background-color: #1B5E20 !important;
        color: #ffffff !important; /* Paksa teks putih */
        border: none;
        border-radius: 8px;
        font-weight: 500;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Hover State */
    div.stButton > button:hover {
        background-color: #2E7D32 !important;
        color: #ffffff !important; 
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* 10. TYPOGRAPHY */
    h1 { color: #1B5E20; font-weight: 700; }
    h2 { color: #2E7D32; font-weight: 600; font-size: 1.5rem; }
    h3 { color: #388E3C; font-weight: 500; font-size: 1.2rem; }
    p, label, span, div { color: #333; }
    
    /* INFO BOX */
    .stAlert {
        background-color: #e8f5e9; 
        border: 1px solid #c8e6c9;
        color: #1b5e20;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI RENDER HTML TABLE  ---
def render_custom_table(df):
    df = df.reset_index(drop=True)
    html = df.to_html(classes='custom-table', index=True, border=0, escape=False)
    st.markdown(html, unsafe_allow_html=True)

#BACKEND

#1. Memuat dataset dan preprocessing
@st.cache_data
def load_data():
    try:
        dtype_spec = {
            'order_id': 'int32',
            'product_id': 'int32',
            'aisle_id': 'int16',
            'department_id': 'int8',
            'add_to_cart_order': 'int16',
            'reordered': 'int8',
            'user_id': 'int32',
            'order_number': 'int16',
            'order_dow': 'int8',
            'order_hour_of_day': 'int8'
        }
        
        # Load data 
        orders = pd.read_csv("orders_sampled.csv", dtype=dtype_spec)
        order_products_prior = pd.read_csv("order_products__prior_sampled.csv", dtype=dtype_spec)
        products = pd.read_csv("products.csv", dtype=dtype_spec)
        aisles = pd.read_csv("aisles.csv", dtype=dtype_spec)
        departments = pd.read_csv("departments.csv", dtype=dtype_spec)
        
        products['product_name'] = products['product_name'].astype('category')
        aisles['aisle'] = aisles['aisle'].astype('category')
        departments['department'] = departments['department'].astype('category')

        products = products.merge(aisles, on='aisle_id', how='left').merge(departments, on='department_id', how='left')
        order_products_prior_merged = order_products_prior.merge(products, on='product_id', how='left')
        
        return orders, order_products_prior_merged, products, departments

    except Exception as e:
        st.error(f"Error Loading Data: {e}")
        return None, None, None, None

orders, order_products_prior, products, departments = load_data()

#2. Fungsi Logika (Helper Functions)
def get_competition_level(order_count):
    if order_count > 5000: return "🔥 High"
    elif order_count > 1000: return "✨ Medium"
    else: return "🌱 Low"

def top_products_by_department(department_name, data, top_n=10):
    dept_products = data[data['department'] == department_name]
    top_products = dept_products['product_name'].value_counts().reset_index()
    top_products.columns = ['Nama Produk', 'Jumlah Order'] 
    top_products['Tingkat Persaingan'] = top_products['Jumlah Order'].apply(get_competition_level)
    return top_products.head(top_n)

def recommend_similar_products(product_name, products_df, top_n=10):
    products_df = products_df.copy()
    products_df['text_features'] = (
        products_df['product_name'].astype(str) + ' ' + 
        products_df['aisle'].astype(str) + ' ' + 
        products_df['department'].astype(str)
    ).str.lower()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products_df['text_features'].fillna(''))
    try:
        idx = products_df[products_df['product_name'].str.lower() == product_name.lower()].index[0]
    except IndexError:
        return None
        
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = cosine_sim.argsort()[::-1][1:top_n+1]
    
    result_df = products_df.iloc[sim_indices][['product_name', 'aisle', 'department']].copy()
    result_df.columns = ['Nama Produk', 'Lorong (Aisle)', 'Departemen']
    result_df['Similarity Score'] = cosine_sim[sim_indices]
    result_df['Similarity Score'] = result_df['Similarity Score'].apply(lambda x: f"{x:.2f}") 
    return result_df.reset_index(drop=True)

def recommend_for_diversification(seller_owned_products, all_orders, products_df, num_recommendations=10, top_n_per_dept=3):
    seller_owned_lower = {p.lower() for p in seller_owned_products}
    owned_product_details = products_df[products_df['product_name'].str.lower().isin(seller_owned_lower)]
    owned_department_ids = set(owned_product_details['department_id'].unique())
    
    product_counts = all_orders.loc[
        ~all_orders['department_id'].isin(owned_department_ids), 
        'product_name'
    ].value_counts().reset_index()

    if product_counts.empty:
        return pd.DataFrame()

    product_counts.columns = ['product_name', 'order_count']
    product_info = product_counts.merge(products_df[['product_name', 'department']], on='product_name', how='left').drop_duplicates(subset=['product_name'])
    
    diverse_recommendations = (product_info.sort_values(['department', 'order_count'], ascending=[True, False]).groupby('department').head(top_n_per_dept))
    final_recommendations = diverse_recommendations.sort_values('order_count', ascending=False).head(num_recommendations)
    final_recommendations['tingkat_persaingan'] = final_recommendations['order_count'].apply(get_competition_level)
    
    final_view = final_recommendations[['product_name', 'department', 'order_count', 'tingkat_persaingan']].copy()
    final_view.columns = ['Nama Produk', 'Departemen', 'Jumlah Order', 'Tingkat Persaingan']
    
    return final_view

def top_trending_products(data, top_n=10):
    top_products = data['product_name'].value_counts().reset_index()
    top_products.columns = ['product_name', 'order_count']
    return top_products.head(top_n)


#UI

if products is not None:
    st.sidebar.title("MENU UTAMA")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigasi:", ["Dashboard Summary", "Fitur Rekomendasi"])

    if page == "Dashboard Summary":
        st.title("Dashboard Seller")
        st.write("Overview pasar dan tren produk terkini.")
 
        with st.container():
            st.markdown("### 💡 Untuk Pemula")
            st.info("Berikut adalah 5 produk paling populer di marketplace saat ini.")
            
            df_trending_home = top_trending_products(order_products_prior, top_n=5)
            df_display = df_trending_home.copy()
            df_display.columns = ['Nama Produk', 'Jumlah Order']
            
            # BAGIAN 1: Tabel Custom HTML
            render_custom_table(df_display)
            
            st.markdown("### 📊 Visualisasi Tren")
            
            # BAGIAN 2: Grafik 
            fig = px.bar(
                df_trending_home, 
                x='order_count', 
                y='product_name', 
                orientation='h',
                text='order_count',
                color_discrete_sequence=['#1B5E20'])
            
            fig.update_layout(
                font=dict(family="Poppins, sans-serif", color="#333333"),
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, title='Jumlah Penjualan'),
                yaxis=dict(title='Nama Produk', categoryorder='total ascending'),
                margin=dict(l=0, r=0, t=30, b=0),
                height=350
            )
            fig.update_traces(textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Fitur Rekomendasi":
        st.title("Rekomendasi Produk untuk Seller")
        
        tab1, tab2, tab3 = st.tabs(["Top Kategori", "Produk Serupa", "Peluang Baru"])
        
        with tab1:
            st.header("Top 10 Produk per Departemen")
            st.caption("Lihat produk paling mendominasi di setiap kategori pasar.")
            
            dept_options = departments['department'].sort_values().unique()
            selected_dept = st.selectbox("Pilih Departemen:", dept_options)
            
            if selected_dept:
                df_top_dept = top_products_by_department(selected_dept, order_products_prior)
                render_custom_table(df_top_dept)
                
        with tab2:
            st.header("Cari Alternatif Produk")
            st.caption("Analisis produk lain yang mirip dengan jualan Anda.")
            
            col_input, col_btn = st.columns([3, 1])
            with col_input:
                input_product = st.text_input("Nama Produk:", "Banana", label_visibility="collapsed", placeholder="Contoh: Banana")
            with col_btn:
                cari_btn = st.button("🔍 Cari Produk Serupa")
            
            if cari_btn:
                if input_product:
                    df_similar = recommend_similar_products(input_product, products)
                    if df_similar is None:
                        st.warning(f'Produk "{input_product}" tidak ditemukan.')
                    else:
                        st.success(f"Ditemukan kemiripan untuk **{input_product}**:")
                        render_custom_table(df_similar)
                        
        with tab3:
            st.header("Ekspansi ke Kategori Baru")
            st.caption("Sistem akan mencari kategori yang belum Anda jual dan merekomendasikan produk terlarisnya.")
            
            seller_products_input = st.text_area(
                "List produk Anda saat ini (pisahkan koma):",
                "Banana, Bag of Organic Bananas, Organic Strawberries"
            )
            
            if st.button("🔍 Analisis Peluang"):
                if seller_products_input:
                    seller_list = [p.strip() for p in seller_products_input.split(',') if p.strip()]
                    results = recommend_for_diversification(seller_list, order_products_prior, products)
                    if results.empty:
                        st.warning("Data tidak cukup atau Anda sudah mendominasi semua kategori.")
                    else:
                        st.write("### Rekomendasi Produk Baru:")
                        render_custom_table(results)
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #43A047;">
        <h2>⚠️ Data Tidak Ditemukan</h2>
        <p>Pastikan file CSV (orders, products, aisles, departments) ada di folder yang sama.</p>
    </div>
    """, unsafe_allow_html=True)

