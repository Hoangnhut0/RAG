import streamlit as st
import fitz  # PyMuPDF để trích xuất văn bản từ PDF
import pytesseract
from pdf2image import convert_from_path
import qdrant_client
from qdrant_client.models import PointStruct, Distance, VectorParams
import os
import uuid
from sentence_transformers import SentenceTransformer
import cv2  # OpenCV để xử lý ảnh
from PIL import Image  # Pillow để xử lý ảnh
from fuzzywuzzy import fuzz  # So sánh chuỗi gần đúng
import nltk  # Thay thế spaCy để xử lý ngôn ngữ tự nhiên
import numpy as np
import requests  # Để gọi API Ollama

# Tải tài nguyên punkt_tab nếu chưa có
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# 🔹 Khởi tạo mô hình embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Kết nối đến Qdrant
qdrant = qdrant_client.QdrantClient("localhost", port=6333)
collection_name = "library_docs"

def ensure_collection_exists():
    collections = qdrant.get_collections()
    collection_names = [col.name for col in collections.collections]
    if collection_name not in collection_names:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        st.success(f"✅ Collection `{collection_name}` đã được tạo thành công!")

ensure_collection_exists()

# 📄 Tiền xử lý hình ảnh trước OCR
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return Image.fromarray(gray)

# 🔹 Hàm gọi API Ollama DeepSeek R1
def call_ollama(prompt, model="deepseek-r1"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Không thể tạo câu trả lời.")
    except Exception as e:
        print(f"❌ Lỗi khi gọi Ollama: {e}")
        return "Không thể tạo câu trả lời."

# Sửa lỗi chính tả bằng Ollama
def correct_spelling_with_ollama(text):
    prompt = f"""
    Văn bản sau có thể chứa lỗi chính tả tiếng Việt và tiếng Anh do trích xuất từ OCR.
    Hãy sửa lỗi chính tả và trả về văn bản đã được chỉnh sửa:
    "{text}"
    """
    return call_ollama(prompt)

# 📄 Trích xuất văn bản từ PDF thường và PDF scan
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                text_content.append(f"{text}\n[Trang {page_num}]")
        return "\n".join(text_content) if text_content else extract_text_with_ocr(pdf_path)
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc PDF: {e}")
        return "Không có nội dung."

# Sử dụng OCR với tiền xử lý nâng cao và sửa lỗi chính tả
def extract_text_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text_content = []
        for i, img in enumerate(images):
            processed_img = preprocess_image(img)
            custom_config = r'--oem 3 --psm 6 -l eng+vie'
            raw_text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            if raw_text:
                corrected_text = correct_spelling_with_ollama(raw_text)
                text_content.append(f"{corrected_text}\n[Trang {i+1}]")
                print(f"📜 Văn bản sau khi sửa lỗi chính tả (Trang {i+1}):\n{corrected_text}\n{'='*50}")
        return "\n".join(text_content) if text_content else "Không có nội dung."
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý OCR: {e}")
        return "Không có nội dung."

# 🔹 Chuyển văn bản thành vector embedding
def generate_embedding(text):
    return embedding_model.encode(text).tolist() if text.strip() else [0.0] * 384

# 🔹 Lưu tài liệu vào Qdrant
def store_document_in_qdrant(doc_id, text, metadata):
    vector = generate_embedding(text)
    metadata["text_content"] = text
    print(f"📦 Văn bản cuối cùng lưu vào Qdrant:\n{text}\n{'='*50}")
    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=doc_id, vector=vector, payload=metadata)]
    )

# 🔍 Tìm kiếm tài liệu trong Qdrant
def search_documents(query):
    query_vector = generate_embedding(query)
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=10,
        with_payload=True
    )
    filtered_results = []
    for r in results:
        text = r.payload.get("text_content", "").lower()
        if fuzz.partial_ratio(query.lower(), text) > 70 or r.score >= 0.4:
            filtered_results.append(r)
    return filtered_results

# 📌 Trích xuất đoạn văn bản với nltk
def extract_relevant_text(full_text, query, filename, context_size=300):
    query_lower = query.lower()
    sentences = nltk.sent_tokenize(full_text)
    for sent in sentences:
        if query_lower in sent.lower():
            start_idx = full_text.lower().find(sent.lower())
            if start_idx == -1:
                continue
            start = max(0, start_idx - context_size // 2)
            end = min(len(full_text), start_idx + len(sent) + context_size // 2)
            relevant_text = full_text[start:end]
            page_number = "Không rõ trang"
            text_before = full_text[:end]
            pages = [line for line in text_before.split("\n") if "[Trang " in line]
            if pages:
                last_page_line = pages[-1]
                page_number = last_page_line.split("[Trang ")[-1].replace("]", "").strip()
            return f"{relevant_text}...", page_number
    return None, "Không rõ trang"

# 🤖 Xử lý với Ollama DeepSeek R1 (Phần trả lời chính từ Ollama)
def process_with_ollama(query, filename=None, context=None):
    if context:
        prompt = f"""
        Người dùng đang tìm kiếm thông tin trong tài liệu: `{filename}`
        Truy vấn: "{query}"
        Nội dung liên quan từ tài liệu:
        {context}
        Hãy diễn giải lại thông tin một cách dễ hiểu và cung cấp nội dung tối ưu nhất.
        """
    else:
        prompt = f'Người dùng đang hỏi: "{query}" Không tìm thấy tài liệu phù hợp.'
    return call_ollama(prompt)  # Phần trả lời từ Ollama được trả về đây

# 🎯 Giao diện Streamlit
def main():
    st.title("📚 Thư viện số thông minh")
    
    uploaded_file = st.file_uploader("📤 Tải lên tài liệu PDF", type=["pdf"])
    if uploaded_file and st.button("📥 Lưu tài liệu"):
        pdf_path = f"uploads/{uploaded_file.name}"
        os.makedirs("uploads", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text = extract_text_from_pdf(pdf_path)
        metadata = {"filename": uploaded_file.name}
        doc_id = str(uuid.uuid4())
        store_document_in_qdrant(doc_id, text, metadata)
        st.success(f"✅ Tài liệu `{uploaded_file.name}` đã được lưu trữ thành công!")

    query = st.text_input("🔍 Nhập từ khóa tìm kiếm")
    if st.button("🔎 Tìm kiếm"):
        results = search_documents(query)
        
        if results:
            for result in results:
                filename = result.payload.get("filename", "Không có tên tài liệu")
                full_text = result.payload.get("text_content", "")
                similarity_score = result.score
                relevant_text, page_number = extract_relevant_text(full_text, query, filename)
                
                if relevant_text:
                    st.markdown(
                        f"📄 **Tài liệu:** {filename} - 📌 **Trang:** {page_number} - "
                        f"📊 **Điểm tương đồng:** {similarity_score:.4f}"
                    )
                    st.markdown(f"📌 **Nội dung trích xuất:**\n\n{relevant_text}")
                    # Lấy và hiển thị câu trả lời từ Ollama
                    refined_answer = process_with_ollama(query, filename, relevant_text)
                    st.markdown(f"🤖 **Trả lời từ Ollama:**\n\n{refined_answer}")  # Phần trả lời được hiển thị ở đây
        else:
            # Trường hợp không tìm thấy tài liệu, Ollama vẫn trả lời dựa trên truy vấn
            refined_answer = process_with_ollama(query)
            st.warning("⚠️ Không tìm thấy tài liệu phù hợp.")
            st.markdown(f"🤖 **Trả lời từ Ollama:**\n\n{refined_answer}")  # Hiển thị câu trả lời khi không có tài liệu

if __name__ == "__main__":
    main()