import streamlit as st
import fitz  # PyMuPDF để trích xuất văn bản từ PDF
import pytesseract
from pdf2image import convert_from_path
import qdrant_client
from qdrant_client.models import PointStruct, Distance, VectorParams
import os
import uuid
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import cv2  # OpenCV để xử lý ảnh
from PIL import Image  # Pillow để xử lý ảnh
from fuzzywuzzy import fuzz  # So sánh chuỗi gần đúng
import nltk  # Thay thế spaCy để xử lý ngôn ngữ tự nhiên
import numpy as np

# Tải tài nguyên punkt_tab nếu chưa có
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# 🔹 Cấu hình Gemini API
genai.configure(api_key="AIzaSyCc5UqNQ1IrAN5rwLQQc4WTckKV0KRfgKw")

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
    # Chuyển sang grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Tăng độ tương phản
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    # Lọc nhiễu
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return Image.fromarray(gray)

#  Sửa lỗi chính tả bằng Gemini AI
def correct_spelling_with_gemini(text):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Văn bản sau có thể chứa lỗi chính tả tiếng Việt và tiếng Anh do trích xuất từ OCR.
    Hãy sửa lỗi chính tả và trả về văn bản đã được chỉnh sửa:
    "{text}"
    """
    try:
        response = model.generate_content(prompt)
        corrected_text = response.text.strip() if response else text
        return corrected_text
    except Exception as e:
        print(f"❌ Lỗi khi sửa chính tả với Gemini: {e}")
        return text  

# 📄 Trích xuất văn bản từ PDF thường và PDF scan
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                text_content.append(f"{text}\n[Trang {page_num}]")
        
        if text_content:
            extracted_text = "\n".join(text_content)
            #print(f"📜 Nội dung trích xuất từ '{pdf_path}':\n{extracted_text}\n{'='*50}")
            return extracted_text
        
        # Nếu không có văn bản -> Xử lý bằng OCR
        return extract_text_with_ocr(pdf_path)
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc PDF: {e}")
        return "Không có nội dung."

#  Sử dụng OCR với tiền xử lý nâng cao và sửa lỗi chính tả
def extract_text_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text_content = []
        
        for i, img in enumerate(images):
            # Tiền xử lý hình ảnh
            processed_img = preprocess_image(img)
            # OCR với cấu hình tùy chỉnh
            custom_config = r'--oem 3 --psm 6 -l eng+vie'
            raw_text = pytesseract.image_to_string(processed_img, config=custom_config).strip()
            if raw_text:
                # Sửa lỗi chính tả
                corrected_text = correct_spelling_with_gemini(raw_text)
                text_content.append(f"{corrected_text}\n[Trang {i+1}]")
                # In văn bản thô và văn bản đã sửa ra terminal để kiểm tra
                #print(f"📜 Văn bản OCR thô (Trang {i+1}) từ '{pdf_path}':\n{raw_text}\n{'-'*50}")
                print(f"📜 Văn bản sau khi sửa lỗi chính tả (Trang {i+1}):\n{corrected_text}\n{'='*50}")
        
        extracted_text = "\n".join(text_content) if text_content else "Không có nội dung."
        return extracted_text
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
    print(f"📦 Văn bản cuối cùng lưu vào Qdrant:\n{text}\n{'='*50}")  # In văn bản trước khi lưu
    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=doc_id, vector=vector, payload=metadata)]
    )

# 🔍 Tìm kiếm tài liệu trong Qdrant với độ tương đồng cải thiện
def search_documents(query):
    query_vector = generate_embedding(query)
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=10,
        with_payload=True
    )
    # Lọc kết quả dựa trên độ tương đồng chuỗi
    filtered_results = []
    for r in results:
        text = r.payload.get("text_content", "").lower()
        if fuzz.partial_ratio(query.lower(), text) > 70 or r.score >= 0.4:
            filtered_results.append(r)
    return filtered_results

# 📌 Trích xuất đoạn văn bản với nltk và tìm số trang chính xác hơn
def extract_relevant_text(full_text, query, filename, context_size=300):
    query_lower = query.lower()
    sentences = nltk.sent_tokenize(full_text)  # Tách câu với nltk
    
    for sent in sentences:
        if query_lower in sent.lower():
            # Tìm vị trí của câu trong văn bản gốc
            start_idx = full_text.lower().find(sent.lower())
            if start_idx == -1:
                continue
            start = max(0, start_idx - context_size // 2)
            end = min(len(full_text), start_idx + len(sent) + context_size // 2)
            relevant_text = full_text[start:end]
            
            # Tìm số trang gần nhất trước hoặc sau đoạn văn bản
            page_number = "Không rõ trang"
            text_before = full_text[:end]  # Lấy toàn bộ văn bản từ đầu đến cuối đoạn trích xuất
            pages = [line for line in text_before.split("\n") if "[Trang " in line]
            if pages:
                # Lấy số trang từ dòng cuối cùng có chứa "[Trang ...]"
                last_page_line = pages[-1]
                page_number = last_page_line.split("[Trang ")[-1].replace("]", "").strip()
            
            return f"{relevant_text}...", page_number
    return None, "Không rõ trang"

# 🤖 Xử lý Gemini AI
def process_with_gemini(query, filename=None, context=None):
    model = genai.GenerativeModel("gemini-2.0-flash")
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
    response = model.generate_content(prompt)
    return response.text if response else "Không thể tạo câu trả lời."

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
                relevant_text, page_number = extract_relevant_text(full_text, query, filename)
                
                if relevant_text:
                    st.markdown(f"📄 **Tài liệu:** {filename} - 📌 **Trang:** {page_number}")
                    st.markdown(f"📌 **Nội dung trích xuất:**\n\n{relevant_text}")
                    refined_answer = process_with_gemini(query, filename, relevant_text)
                    st.markdown(f"🤖 **Trả lời:**\n\n{refined_answer}")

if __name__ == "__main__":
    main()  