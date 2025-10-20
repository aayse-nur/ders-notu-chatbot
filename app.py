# GEREKLİ KÜTÜPHANELERİN İÇE AKTARILMASI
import streamlit as st
import os

# LangChain bileşenleri
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ========================================================
# 1. ORTAM AYARLARI VE GÜVENLİK KONTROLÜ
# ========================================================

# Streamlit'in GÜVENLİ SECRETS alanından API anahtarını oku.
# BU YÖNTEM, API anahtarını koda yazmaz ve Cloud ortamında çalışır.
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("GÜVENLİK HATASI: OPENAI_API_KEY gizli anahtarı Streamlit Cloud Secrets bölümünde ayarlanmadı.")
    st.stop()

# Vektör veritabanının fiziksel olarak nerede kayıtlı olduğunu belirtir.
PERSIST_DIR = "chroma_db"

# Vektörleme için kullanılan modeli tanımlar.
EMBEDDING_MODEL = OpenAIEmbeddings(openai_api_key=openai_api_key)

# ========================================================
# 2. RAG BAĞLANTISI VE MODEL YÜKLEME
# ========================================================

# Vektör veritabanını yüklemeyi dener.
try:
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=EMBEDDING_MODEL
    )
    # RAG sorgusu için Retriever oluşturulur.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.warning("UYARI: ChromaDB veritabanı yüklenemedi. 'chroma_db' klasörünün var olduğundan emin olun.")
    st.error(f"ChromaDB yüklenirken hata oluştu: {e}")
    st.stop()

# LLM'i (GPT-4o mini) yükler.
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)

# RetrievalQA Chain'i oluşturulur.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# ========================================================
# 3. STREAMLIT ARAYÜZÜ VE KULLANICI ETKİLEŞİMİ
# ========================================================

# Sayfa ayarları ve başlıkların güncellenmesi
st.set_page_config(page_title="Elektrik ve Elektronik RAG Asistanı", page_icon="⚡")
st.title("⚡ Elektrik ve Elektronik Ders Notları Asistanı")
st.markdown("Yüklenen tüm Elektrik ve Elektronik ders notlarını kullanarak bilgiye dayalı (Grounded RAG) yapay zeka sistemi ile etkileşim kurun.")

# Kullanıcıdan gelen input için arayüz bileşeni
user_query = st.text_input(
    "Sorunuzu Buraya Yazın (Örn: Diyot nedir ve en yaygın kullanım alanı nedir?)",
    key="user_input"
)

# Kullanıcı bir soru yazdığında bu blok çalışır.
if user_query:
    # Cevap üretilirken kullanıcıya 'spinner' (dönen ikon) gösterilir.
    with st.spinner("Ders notları taranıyor ve cevap üretiliyor..."):
        try:
            # RAG zincirini çalıştırır ve cevabı alır.
            response = qa_chain.invoke({"query": user_query})
            st.success("Cevap Hazır!")

            # Üretilen cevabı ekranda gösterir.
            st.subheader("Cevap")
            st.markdown(response["result"])

        except Exception as e:
            st.error(f"Sorgulama sırasında bir hata oluştu: {e}")
            st.stop()
