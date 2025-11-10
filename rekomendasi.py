import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

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

        print("Di dalam load_data: Mencoba membaca file CSV dengan tipe data optimal...")
        orders = pd.read_csv("orders_sampled.csv", dtype=dtype_spec)
        order_products_prior = pd.read_csv("order_products__prior_sampled.csv", dtype=dtype_spec)
        products = pd.read_csv("products.csv", dtype=dtype_spec)
        aisles = pd.read_csv("aisles.csv", dtype=dtype_spec)
        departments = pd.read_csv("departments.csv", dtype=dtype_spec)
        print("Di dalam load_data: Berhasil membaca semua file.")

        products['product_name'] = products['product_name'].astype('category')
        aisles['aisle'] = aisles['aisle'].astype('category')
        departments['department'] = departments['department'].astype('category')

    except Exception as e:
        print(f"!!! TERJADI ERROR DI DALAM load_data: {e}")
        st.error(f"Gagal memuat data: {e}. Periksa file CSV Anda atau path folder 'data'.")
        return None, None, None, None

    products = products.merge(aisles, on='aisle_id', how='left').merge(departments, on='department_id', how='left')
    order_products_prior_merged = order_products_prior.merge(products, on='product_id', how='left')
    print("Di dalam load_data: Berhasil melakukan merge data.")
    
    return orders, order_products_prior_merged, products, departments

orders, order_products_prior, products, departments = load_data()

#2. Fitur rekomendasi
def get_competition_level(order_count):
    if order_count > 5000: return "Sangat Populer (Persaingan Tinggi)"
    elif order_count > 1000: return "Populer (Persaingan Sedang)"
    else: return "Niche (Persaingan Rendah)"

def top_products_by_department(department_name, data, top_n=10):
    dept_products = data[data['department'] == department_name]
    top_products = dept_products['product_name'].value_counts().reset_index()
    top_products.columns = ['product_name', 'order_count']
    top_products['tingkat_persaingan'] = top_products['order_count'].apply(get_competition_level)
    return top_products.head(top_n)

def recommend_similar_products(product_name, products_df, top_n=10):
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
    result_df['similarity_score'] = cosine_sim[sim_indices]
    
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
    
    return final_recommendations

def top_trending_products(data, top_n=10):
    top_products = data['product_name'].value_counts().reset_index()
    top_products.columns = ['product_name', 'order_count']
    top_products_with_dept = top_products.merge(products[['product_name', 'department', 'aisle']], on='product_name', how='left').drop_duplicates(subset=['product_name'])
    top_products_with_dept['tingkat_persaingan'] = top_products_with_dept['order_count'].apply(get_competition_level)
    return top_products_with_dept.head(top_n)

#3. UI streamlit
if products is not None:
    st.sidebar.title("MENU NAVIGASI")
    page = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Fitur Rekomendasi"])

    if page == "Beranda":
        st.title("🏡 Beranda: Dasbor untuk Seller")
        st.write("Selamat datang! Halaman ini dirancang untuk memberikan Anda wawasan cepat untuk memulai.")
        st.markdown("---")
        st.subheader("Bagi Anda yang Baru Memulai:")
        st.info("Berikut adalah 5 produk paling populer di seluruh marketplace saat ini. Ini adalah titik awal yang bagus untuk menentukan produk pertama Anda.")
        df_trending_home = top_trending_products(order_products_prior, top_n=5)
        st.dataframe(df_trending_home[['product_name', 'department', 'order_count', 'tingkat_persaingan']])

        st.write("#### Visualisasi Peringkat Produk")
        fig, ax = plt.subplots(figsize=(10, 4))
        product_names = df_trending_home['product_name']
        order_counts = df_trending_home['order_count']
        ax.barh(product_names, order_counts, color='skyblue')
        ax.invert_yaxis()
        ax.set_xlabel('Jumlah Order')
        ax.set_title('Top 5 Produk Paling Populer')
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Bagi Anda yang Ingin Mengembangkan Bisnis:")
        st.write("Gunakan **Fitur Rekomendasi** di sidebar untuk mendapatkan analisis yang lebih mendalam.")

    elif page == "Fitur Rekomendasi":
        st.title("🛠️ Fitur Lengkap Sistem Rekomendasi")
        tab1, tab2, tab3 = st.tabs(["🔝 Top per Departemen", "🤝 Produk Mirip", "🚀 Rekomendasi Kategori Lain"])
        with tab1:
            st.header("🏆 Top 10 Produk Terlaris di Setiap Departemen")
            st.info("Gunakan fitur ini untuk melihat produk apa yang paling mendominasi di setiap kategori pasar.")
            dept_options = departments['department'].sort_values().unique()
            selected_dept = st.selectbox("Pilih Departemen:", dept_options, key="dept_select")
            if selected_dept:
                df_top_dept = top_products_by_department(selected_dept, order_products_prior)
                st.dataframe(df_top_dept)
        with tab2:
            st.header("🤝 Rekomendasi Produk yang Mirip")
            st.info("Fitur ini cocok jika Anda sudah memiliki beberapa produk dan ingin mencari variasi atau alternatif yang mirip.")
            input_product = st.text_input("Masukkan nama produk yang sudah Anda jual:", "Banana", key="similar_input")
            if input_product:
                df_similar = recommend_similar_products(input_product, products)
                if df_similar is None:
                    st.warning(f'Produk "{input_product}" tidak ditemukan.')
                else:
                    st.write(f"Produk yang mirip dengan **{input_product}**:")
                    st.dataframe(df_similar)
        with tab3:
            st.header("🚀 Rekomendasi Produk yang Belum Dijual")
            st.info("Fitur ini menganalisis produk Anda dan merekomendasikan produk terlaris dari kategori yang belum pernah Anda masuki.")
            seller_products_input = st.text_area(
                "Masukkan produk yang sudah Anda miliki (pisahkan dengan koma):",
                "Banana, Bag of Organic Bananas, Organic Strawberries",
                key="diversify_input"
            )
            if seller_products_input:
                seller_list = [p.strip() for p in seller_products_input.split(',') if p.strip()]
                results = recommend_for_diversification(seller_list, order_products_prior, products)
                if results.empty:
                    st.warning("Tidak ditemukan rekomendasi. Mungkin Anda sudah menjual produk di semua departemen?")
                else:
                    st.write("### Top Produk di Kategori Baru untuk Anda Coba:")
                    st.dataframe(results[['product_name', 'department', 'order_count', 'tingkat_persaingan']])
else:
    st.header("❌ Aplikasi Gagal Dimuat")
    st.warning("Pastikan file data Anda berada di dalam folder 'data' yang benar dan tidak ada error saat pemuatan. Silakan periksa terminal untuk detail error.")