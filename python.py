# python.py

import streamlit as st
import pandas as pd
# Sử dụng google.generativeai để có các tính năng mới nhất
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính với Chatbot AI 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Sử dụng điều kiện để xử lý giá trị 0 thủ công cho mẫu số.
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Ban đầu ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét ban đầu."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = model.generate_content(prompt)
        return response.text

    except GoogleAPICallError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM MỚI: Gọi API Gemini cho tính năng Chat ---
def get_chat_response(api_key, chat_history, question, financial_context):
    """
    Gửi toàn bộ lịch sử chat và câu hỏi mới đến Gemini để có câu trả lời theo ngữ cảnh.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Xây dựng prompt với hướng dẫn, bối cảnh tài chính và lịch sử trò chuyện
        system_prompt = f"""
        Bạn là một chuyên gia phân tích tài chính. Hãy trả lời câu hỏi của người dùng dựa trên bối cảnh tài chính được cung cấp dưới đây.
        Hãy đưa ra câu trả lời sâu sắc, rõ ràng và trực tiếp vào vấn đề.

        --- BỐI CẢNH DỮ LIỆU TÀI CHÍNH ---
        {financial_context}
        --- KẾT THÚC BỐI CẢNH ---
        """

        # Bắt đầu phiên chat với hướng dẫn hệ thống và lịch sử
        # Streamlit chạy lại script mỗi lần tương tác, vì vậy chúng ta cần xây dựng lại lịch sử mỗi lần gọi
        history_for_api = []
        history_for_api.append({'role': 'user', 'parts': [{'text': system_prompt}]})
        history_for_api.append({'role': 'model', 'parts': [{'text': "Đã hiểu. Tôi đã sẵn sàng phân tích dữ liệu và trả lời câu hỏi của bạn."}]})

        for message in chat_history:
            role = "user" if message["role"] == "user" else "model"
            history_for_api.append({'role': role, 'parts': [{'text': message["content"]}]})
        
        # Thêm câu hỏi mới nhất của người dùng
        history_for_api.append({'role': 'user', 'parts': [{'text': question}]})

        response = model.generate_content(history_for_api)
        return response.text

    except GoogleAPICallError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            try:
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N else 0
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Chỉ số Thanh toán Hiện hành (Năm trước)", value=f"{thanh_toan_hien_hanh_N_1:.2f} lần")
                with col2:
                    st.metric(label="Chỉ số Thanh toán Hiện hành (Năm sau)", value=f"{thanh_toan_hien_hanh_N:.2f} lần", delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}")
            except (IndexError, ZeroDivisionError):
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chuẩn bị dữ liệu chung cho AI ---
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': ['Toàn bộ Bảng phân tích', 'Thanh toán hiện hành (N-1)', 'Thanh toán hiện hành (N)'],
                'Giá trị': [df_processed.to_markdown(index=False), f"{thanh_toan_hien_hanh_N_1}", f"{thanh_toan_hien_hanh_N}"]
            }).to_markdown(index=False)

            # --- Chức năng 5: Nhận xét AI ban đầu ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY")
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # --- TÍNH NĂNG MỚI: Chức năng 6: Khung Chat với Gemini AI ---
            st.divider()
            st.subheader("6. Trò chuyện sâu hơn với AI")

            api_key_chat = st.secrets.get("GEMINI_API_KEY")
            if not api_key_chat:
                st.error("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng tính năng chat.")
            else:
                # Khởi tạo lịch sử chat trong st.session_state.
                # Đây là bước quan trọng để lưu giữ tin nhắn giữa các lần người dùng tương tác.
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Hiển thị các tin nhắn đã có trong lịch sử
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Tạo ô nhập liệu chat ở cuối trang
                if prompt := st.chat_input("Đặt câu hỏi về báo cáo tài chính này..."):
                    # Thêm tin nhắn của người dùng vào lịch sử và hiển thị ngay lập tức
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Tạo tin nhắn của AI và hiển thị trạng thái "đang suy nghĩ"
                    with st.chat_message("assistant"):
                        with st.spinner("AI đang phân tích..."):
                            response = get_chat_response(
                                api_key=api_key_chat,
                                chat_history=st.session_state.messages[:-1], # Gửi lịch sử TRƯỚC câu hỏi hiện tại
                                question=prompt,
                                financial_context=data_for_ai
                            )
                            st.markdown(response)
                    
                    # Thêm phản hồi của AI vào lịch sử để duy trì cuộc trò chuyện
                    st.session_state.messages.append({"role": "assistant", "content": response})

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
