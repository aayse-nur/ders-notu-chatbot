# ⚡ RAG TEMELLİ CHATBOT PROJESİ: ELEKTRİK VE ELEKTRONİK DERS NOTLARI ASİSTANI

## 🎯 Projenin Amacı (Gereksinim 1)

Bu proje, **Retrieval-Augmented Generation (RAG)** mimarisini kullanarak, harici bir **Elektrik ve Elektronik** ders notları veri setine dayalı, bilgiye kapalı (grounded) bir soru-cevap asistanı geliştirmeyi amaçlamaktadır. Projenin temel hedefi, yalnızca sağlanan ders materyallerinin içeriğiyle sınırlı, **tutarlı, doğru ve akademik** cevaplar üretebilen bir chatbot sunmaktır.

## 📚 Veri Seti Hakkında Bilgi (Gereksinim 2)

Bu RAG sisteminde kullanılan bilgi kaynağı, **Analog Elektronik** ders notlarından oluşan bir PDF dosyasıdır. (İleride kolayca diğer ders notları eklenebilecek şekilde genel bir mimariyle tasarlanmıştır.)

* **Veri Tipi:** Teknik ders notu (PDF).

* **İçerik:** Diyotlar, transistörler, temel devre analizi ve pasif elektronik bileşenler gibi Analog Elektronik temel konularını kapsamaktadır.

* **Hazırlık Metodolojisi:** Veri seti, **`PyPDFLoader`** kullanılarak okunmuş ve **`RecursiveCharacterTextSplitter`** ile anlamlı metin parçalarına (chunk) ayrılmıştır. Bu parçalar daha sonra vektörlere çevrilmiştir.

## 🛠️ Çözüm Mimarisi ve Kullanılan Teknolojiler (Gereksinim 4)

Proje, **LangChain** çatısı etrafında kurulmuş bir **RAG mimarisini** kullanır. Bu mimari, LLM'in genel bilgi yerine ders notlarına odaklanmasını sağlayarak "halüsinasyon" riskini ortadan kaldırır.

| Bileşen Adı | Kullanılan Teknoloji | Görev | 
 | ----- | ----- | ----- | 
| **Büyük Dil Modeli (LLM)** | **OpenAI GPT-4o-mini** | Çekilen kaynak metinleri yorumlayarak nihai cevabı üretir. | 
| **Vektörleştirme (Embedding)** | **OpenAI Embeddings** | Metin parçalarını ve kullanıcı sorgusunu sayısal vektörlere çevirir. | 
| **Vektör Veritabanı** | **ChromaDB** | Vektörleri depolar ve sorgu anında en alakalı $k=3$ metin parçasını çeker (Retrieval). | 
| **Sorgulama Zinciri (RAG Chain)** | **LangChain's RetrievalQA** | Sorgulama, çekme (Retrieval) ve cevap üretme (Generation) süreçlerini yöneten ana beynidir. | 
| **Web Arayüzü** | **Streamlit** | Kullanıcı etkileşimini sağlayan basit, hızlı ve temiz arayüzü sunar. | 

## ⚙️ KODUN ÇALIŞMA KILAVUZU (Gereksinim 3) - **Streamlit Cloud Yayınlama Kılavuzu**

Bu proje, `app.py` dosyasında yaptığımız değişiklik nedeniyle artık **Streamlit Cloud** ortamına göre ayarlanmıştır. API anahtarının yönetimi için güvenli **`st.secrets`** yapısını kullanır.

### 1. GitHub Dosya Hazırlığı

Uygulamanın Streamlit Cloud'da çalışması için, yeni deponuzda (`ders-notu-chatbot`) aşağıdaki **4 ana bileşen** bulunmalıdır:

* **`app.py`**: İçinde **`st.secrets`** ile API anahtarını okuyan en son kodunuz.

* **`requirements.txt`**: Gerekli tüm Python bağımlılıklarını içeren dosya.

* **`chroma_db`**: **Önceden oluşturulmuş** vektör veritabanını içeren klasör.

* **`README.md`**: Bu belgenin kendisi.

### 2. Kütüphane Kurulumu (requirements.txt İçeriği)

Streamlit Cloud, bu dosyayı okuyarak gerekli tüm kütüphaneleri otomatik olarak kurar.


streamlit
langchain-core
langchain-community
langchain-openai
openai
tiktoken
chromadb
pypdf
pypdfium2
python-dotenv


### 3. API Anahtarı Ayarı (**Streamlit Secrets** - KRİTİK ADIM)

Kod, API anahtarınızı güvenli bir şekilde Streamlit Secrets üzerinden okuyacaktır. Bu ayar, **yalnızca Streamlit Cloud arayüzünde** yapılmalıdır. Yerel `.env` dosyası KULLANILMAZ.

1. Streamlit Cloud'da uygulamanızı yayınlarken veya Ayarlar (Settings) bölümünden **"Manage app"** menüsüne gidin.

2. **"Settings"** -> **"Secrets"** bölümünü açın.

3. Aşağıdaki formatta bir gizli anahtar ekleyin:



secrets.toml dosyasına eklenmesi gereken içerik

OPENAI_API_KEY = "sk-SENİN_ANAHTARIN_BURAYA_GELMELİ"


* **Uyarı:** Anahtar adı (`OPENAI_API_KEY`), `app.py` dosyasındaki kod ile birebir eşleşmelidir.

### 4. Vektör Veritabanının Oluşturulması (Initial Setup)
1.  Kullanmak istediğiniz PDF'leri **`data`** klasörüne yerleştirin.
2.  Ana Python dosyasını (`app.py`) çalıştırarak **`chroma_db`** klasörünü oluşturun:
    ```bash
    python app.py
    ```
    *Bu adım, LLM modelini kullanır ve biraz zaman alabilir.*

### 5. Chatbot'un Çalıştırılması
1.  Veritabanı oluştuktan sonra, Streamlit arayüzünü başlatın:
    ```bash
    streamlit run app.py
    ```

---

### 6. Chatbot'un Yayınlanması (Deploy)

1. Streamlit Cloud'da, yeni GitHub deponuzu (`ders-notu-chatbot`) seçin.

2. Gerekli ayarları (Branch, Main file path) kontrol edin ve **"Deploy!"** butonuna tıklayın.

## 🌐 Web Arayüzü ve Product Kılavuzu (Gereksinim 5)

Uygulama, temiz ve odaklanmış bir Streamlit arayüzü ile sunulmaktadır. Sayfanın başlığı ve simgesi, projenin **Elektrik ve Elektronik** temasını yansıtır.

### Çalışma Akışı

1. Arayüz, tarayıcıda açılır. Kullanıcı, sayfanın altındaki metin kutusuna ders notlarıyla ilgili sorusunu yazar.

2. Sistem, anlık olarak:
a. Kullanıcı sorusuna en alakalı 3 metin parçasını veritabanından çeker.
b. Bu parçaları ve soruyu **GPT-4o mini** modeline gönderir.
c. LLM tarafından üretilen cevabı ekrana basar.

### Test Önerisi

* **Test Sorusu:** "Bir diyotun en yaygın kullanım alanı nedir ve neden?"

* **Beklenen Sonuç:** Cevap, internetteki genel bilgiler yerine, sadece yüklediğiniz ders notlarında geçen diyot tanımları ve kullanım senaryolarına bağlı kalacaktır.

### 🔗 Uygulama Linki (Deploy Linki Buraya Gelecek)

**Web Linki:** `[WEB UYGULAMASI LİNKİNİZ BURAYA GELECEK]`
