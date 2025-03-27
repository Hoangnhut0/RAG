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
from PIL import Image  # để xử lý ảnh
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

# Xử lý hình ảnh trước OCR
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return Image.fromarray(gray)

# Sửa lỗi chính tả bằng Gemini AI
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
            return extracted_text
        
        return extract_text_with_ocr(pdf_path)
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
                corrected_text = correct_spelling_with_gemini(raw_text)
                text_content.append(f"{corrected_text}\n[Trang {i+1}]")
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
    print(f"📦 Văn bản cuối cùng lưu vào Qdrant:\n{text}\n{'='*50}")
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
    filtered_results = [r for r in results if r.score >= 0.5]
    return filtered_results

# 📌 Trích xuất đoạn văn bản với nltk và tìm số trang chính xác hơn
def extract_relevant_text(full_text, query, filename, context_size=300):
    sentences = nltk.sent_tokenize(full_text)
    query_embedding = generate_embedding(query)
    
    best_sentence = None
    best_similarity = -1
    best_sentence_idx = -1
    
    for idx, sent in enumerate(sentences):
        sent_embedding = generate_embedding(sent)
        similarity = np.dot(query_embedding, sent_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(sent_embedding))
        if similarity > best_similarity:
            best_similarity = similarity
            best_sentence = sent
            best_sentence_idx = idx
    
    if best_sentence and best_similarity > 0.7:
        paragraphs = full_text.split("\n\n")
        relevant_paragraph = None
        
        for para in paragraphs:
            if best_sentence in para:
                relevant_paragraph = para
                break
        
        if relevant_paragraph:
            para_sentences = nltk.sent_tokenize(relevant_paragraph)
            if len(para_sentences) < 3:
                para_idx = paragraphs.index(relevant_paragraph)
                start_para_idx = max(0, para_idx - 1)
                end_para_idx = min(len(paragraphs), para_idx + 2)
                relevant_text = "\n\n".join(paragraphs[start_para_idx:end_para_idx]).strip()
            else:
                relevant_text = relevant_paragraph.strip()
            
            page_number = "Không rõ trang"
            text_before = full_text[:full_text.find(relevant_text) + len(relevant_text)]
            pages = [line for line in text_before.split("\n") if "[Trang " in line]
            if pages:
                last_page_line = pages[-1]
                page_number = last_page_line.split("[Trang ")[-1].replace("]", "").strip()
            
            return f"{relevant_text}...", page_number
        
        relevant_text = determine_paragraph_boundaries_with_gemini(full_text, best_sentence)
        if relevant_text:
            page_number = "Không rõ trang"
            text_before = full_text[:full_text.find(relevant_text) + len(relevant_text)]
            pages = [line for line in text_before.split("\n") if "[Trang " in line]
            if pages:
                last_page_line = pages[-1]
                page_number = last_page_line.split("[Trang ")[-1].replace("]", "").strip()
            return f"{relevant_text}...", page_number
        
        start_idx = full_text.find(best_sentence)
        if start_idx == -1:
            return None, "Không rõ trang"
        
        start = max(0, full_text.rfind("\n\n", 0, start_idx))
        end = full_text.find("\n\n", start_idx + len(best_sentence))
        if end == -1:
            end = len(full_text)
        relevant_text = full_text[start:end].strip()
        
        page_number = "Không rõ trang"
        text_before = full_text[:end]
        pages = [line for line in text_before.split("\n") if "[Trang " in line]
        if pages:
            last_page_line = pages[-1]
            page_number = last_page_line.split("[Trang ")[-1].replace("]", "").strip()
        
        return f"{relevant_text}...", page_number
    
    return None, "Không rõ trang"

# Sử dụng Gemini AI để xác định ranh giới đoạn văn
def determine_paragraph_boundaries_with_gemini(full_text, best_sentence):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Dưới đây là một đoạn văn bản dài:
    "{full_text[:2000]}"
    
    Câu sau nằm trong văn bản: "{best_sentence}"
    Hãy xác định đoạn văn (paragraph) chứa câu này và trả về đoạn văn đó.
    """
    try:
        response = model.generate_content(prompt)
        relevant_paragraph = response.text.strip() if response else best_sentence
        return relevant_paragraph
    except Exception as e:
        print(f"❌ Lỗi khi xác định đoạn văn với Gemini: {e}")
        return best_sentence

# 🤖 Kiểm tra mức độ liên quan của nội dung trích xuất với câu hỏi
def check_relevance_with_gemini(query, relevant_text):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Câu hỏi: "{query}"
    Nội dung trích xuất từ tài liệu: "{relevant_text}"
    Hãy đánh giá xem nội dung trích xuất có thực sự liên quan đến câu hỏi không.
    Trả về "Có" nếu liên quan, "Không" nếu không liên quan.
    """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip() if response else "Không"
        return result == "Có"
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra mức độ liên quan với Gemini: {e}")
        return False

# 🤖 Xử lý Gemini AI với prompt cải tiến
def process_with_gemini(query, filename=None, context=None, full_text=None):
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    if context and full_text:
        # Prompt cải tiến để Gemini hiểu câu hỏi và tìm kiếm trong tài liệu
        prompt = f"""
        Bạn là một trợ lý thông minh giúp trả lời câu hỏi dựa trên nội dung tài liệu.
        
        **Câu hỏi của người dùng:** "{query}"
        **Tài liệu liên quan:** `{filename}`
        **Nội dung trích xuất ban đầu từ tài liệu (có thể không đầy đủ):** 
        {context}
        
        **Toàn bộ nội dung tài liệu (dùng để tìm kiếm thêm nếu cần):** 
        {full_text[:5000]}  # Giới hạn độ dài để tránh vượt quá giới hạn của Gemini
        
        **Nhiệm vụ:**
        1. Hiểu câu hỏi của người dùng và xác định thông tin cần thiết để trả lời.
        2. Tìm kiếm trong toàn bộ nội dung tài liệu để lấy đoạn văn bản liên quan nhất đến câu hỏi.
        3. Trả lời câu hỏi một cách chính xác, dễ hiểu, và dựa trên thông tin trong tài liệu.
        4. Nếu thông tin trong tài liệu không đủ, hãy giải thích ngắn gọn và đưa ra gợi ý hợp lý.
        
        **Định dạng trả lời:**
        - Nếu tìm thấy thông tin liên quan: Trả lời câu hỏi và trích dẫn đoạn văn bản liên quan từ tài liệu.
        - Nếu không tìm thấy: Giải thích rằng không có thông tin phù hợp và đưa ra gợi ý.
        """
    elif context:
        prompt = f"""
        Bạn là một trợ lý thông minh giúp trả lời câu hỏi dựa trên nội dung tài liệu.
        
        **Câu hỏi của người dùng:** "{query}"
        **Tài liệu liên quan:** `{filename}`
        **Nội dung trích xuất từ tài liệu:** 
        {context}
        
        **Nhiệm vụ:**
        1. Hiểu câu hỏi của người dùng và xác định thông tin cần thiết để trả lời.
        2. Dựa vào nội dung trích xuất để trả lời câu hỏi một cách chính xác và dễ hiểu.
        3. Nếu thông tin không đủ, hãy giải thích ngắn gọn và đưa ra gợi ý hợp lý.
        
        **Định dạng trả lời:**
        - Trả lời câu hỏi và trích dẫn đoạn văn bản liên quan từ nội dung trích xuất.
        - Nếu không đủ thông tin, giải thích và đưa ra gợi ý.
        """
    else:
        prompt = f"""
        Bạn là một trợ lý thông minh giúp trả lời câu hỏi.
        
        **Câu hỏi của người dùng:** "{query}"
        **Thông tin:** Không tìm thấy tài liệu phù hợp.
        
        **Nhiệm vụ:**
        1. Trả lời câu hỏi dựa trên kiến thức chung của bạn một cách ngắn gọn và dễ hiểu.
        2. Nếu không thể trả lời, hãy giải thích và đưa ra gợi ý hợp lý.
        """
    
    try:
        response = model.generate_content(prompt)
        return response.text if response else "Không thể tạo câu trả lời."
    except Exception as e:
        print(f"❌ Lỗi khi xử lý với Gemini: {e}")
        return "Không thể tạo câu trả lời."

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

    query = st.text_input("❓ Nhập câu hỏi của bạn")
    if st.button("🔎 Tra cứu"):
        if query:
            results = search_documents(query)
            relevant_found = False
            relevant_results = []
            
            if results:
                for result in results:
                    filename = result.payload.get("filename", "Không có tên tài liệu")
                    full_text = result.payload.get("text_content", "")
                    similarity_score = result.score
                    relevant_text, page_number = extract_relevant_text(full_text, query, filename)
                    
                    if relevant_text:
                        is_relevant = check_relevance_with_gemini(query, relevant_text)
                        if is_relevant:
                            relevant_found = True
                            relevant_results.append({
                                "filename": filename,
                                "full_text": full_text,
                                "similarity_score": similarity_score,
                                "relevant_text": relevant_text,
                                "page_number": page_number
                            })
            
            if relevant_found:
                for res in relevant_results:
                    filename = res["filename"]
                    full_text = res["full_text"]
                    similarity_score = res["similarity_score"]
                    relevant_text = res["relevant_text"]
                    page_number = res["page_number"]
                    
                    pdf_path = f"uploads/{filename}"
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as file:
                            pdf_data = file.read()
                        st.markdown(
                            f"📄 **Tài liệu:** {filename} - 📌 **Trang:** {page_number} - "
                            f"📊 **Điểm tương đồng:** {similarity_score:.4f}"
                        )
                        st.download_button(
                            label="📥 Tải xuống tài liệu",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf"
                        )
                    else:
                        st.markdown(
                            f"📄 **Tài liệu:** {filename} (File không khả dụng) - 📌 **Trang:** {page_number} - "
                            f"📊 **Điểm tương đồng:** {similarity_score:.4f}"
                        )
                    
                    st.markdown(f"📌 **Nội dung trích xuất ban đầu:**\n\n{relevant_text}")
                    # Truyền cả full_text để Gemini có thể tìm kiếm thêm nếu cần
                    refined_answer = process_with_gemini(query, filename, relevant_text, full_text)
                    st.markdown(f"🤖 **Trả lời từ Gemini:**\n\n{refined_answer}")
            else:
                refined_answer = process_with_gemini(query)
                st.markdown(f"🤖 **Trả lời từ Gemini:**\n\n{refined_answer}")
        else:
            st.error("❌ Vui lòng nhập câu hỏi!")

if __name__ == "__main__":
    main()