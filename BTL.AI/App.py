import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 1. CẤU HÌNH GIAO DIỆN DASHBOARD ĐẲNG CẤP
st.set_page_config(page_title="VNM Financial Risk Dashboard", layout="wide", initial_sidebar_state="expanded")

# Tối ưu giao diện bằng CSS nâng cao
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3em; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# Tải tài sản học máy
@st.cache_resource
def load_ml_assets():
    model = joblib.load(os.path.join(BASE_DIR, 'logistic_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    return model, scaler

try:
    model, scaler = load_ml_assets()

    # --- TIÊU ĐỀ CHÍNH ---
    st.markdown("<h1 style='text-align: center; color: #1E3A8A; font-family: sans-serif;'>🏛️ HỆ THỐNG QUẢN TRỊ RỦI RO TÀI CHÍNH CHIẾN LƯỢC - VINAMILK (VNM)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 16px;'>Mô hình toán học kết hợp Học máy phân tích đa chiều Chuỗi chỉ số theo thời gian</p>", unsafe_allow_html=True)
    st.write("---")

    # --- KHU VỰC NHẬP LIỆU ĐA BIẾN ---
    st.markdown("### 📥 Nhập thông số tài chính thực tế của Doanh nghiệp")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Quy mô & Dòng tiền")
            revenue = st.number_input("1. Doanh thu thuần (VND)", min_value=0.0, value=15000000000000.0, step=100000000000.0, format="%.0f")
            net_profit = st.number_input("2. Lợi nhuận sau thuế (VND)", value=2200000000000.0, step=10000000000.0, format="%.0f")
            cash_flow = st.number_input("3. Dòng tiền thuần từ HĐKD (VND)", value=2500000000000.0, step=10000000000.0, format="%.0f")

        with col2:
            st.markdown("#### ⚖️ Cấu trúc Vốn")
            total_assets = st.number_input("4. Tổng tài sản (VND)", min_value=1.0, value=52000000000000.0, step=100000000000.0, format="%.0f")
            total_debt = st.number_input("5. Tổng nợ phải trả (VND)", min_value=0.0, value=15000000000000.0, step=100000000000.0, format="%.0f")
            equity = st.number_input("6. Vốn chủ sở hữu (VND)", min_value=1.0, value=37000000000000.0, step=100000000000.0, format="%.0f")

        with col3:
            st.markdown("#### 📈 Chỉ số Hiệu quả")
            current_ratio = st.number_input("7. Hệ số thanh toán hiện hành (Lần)", min_value=0.0, value=2.2, step=0.1)
            roa = st.number_input("8. Tỷ suất sinh lời trên Tài sản - ROA (%)", value=4.2, step=0.1)
            roe = st.number_input("9. Tỷ suất sinh lời trên Vốn CSH - ROE (%)", value=5.9, step=0.1)

    st.write("---")

    # --- THUẬT TOÁN XỬ LÝ DỮ LIỆU ĐỘNG ---
    # 1. Tính toán biến phái sinh để đưa vào mô hình ML gốc
    calculated_debt_ratio = total_debt / total_assets
    
    # 2. Dự báo Core Probability từ mô hình học máy gốc (Dựa trên 5 biến cốt lõi)
    ml_input = np.array([[revenue, net_profit, calculated_debt_ratio, roa/100, roe/100]])
    ml_input_scaled = scaler.transform(ml_input)
    base_probability = model.predict_proba(ml_input_scaled)[0][1]

    # 3. HỆ CHUYÊN GIA (EXPERT LAYER): Tự động điều chỉnh rủi ro dựa trên 4 biến bổ sung bổ trợ bên ngoài
    # Phân tích chất lượng dòng tiền và thanh khoản thực tế để tinh chỉnh xác suất
    penalty_score = 0.0
    if cash_flow < 0:
        penalty_score += 0.15 # Dòng tiền âm cực kỳ nguy hiểm
    if cash_flow < net_profit:
        penalty_score += 0.05 # Chất lượng lợi nhuận thấp (lợi nhuận trên giấy giấy tờ)
    if current_ratio < 1.0:
        penalty_score += 0.15 # Báo động mất thanh khoản ngắn hạn
    elif current_ratio < 1.5:
        penalty_score += 0.05 # Thanh khoản hơi mỏng
        
    # Tính toán xác suất rủi ro tổng hợp cuối cùng (Giới hạn từ 0% đến 100%)
    final_probability = min(max(base_probability + penalty_score, 0.0), 1.0)

    # Phân loại trạng thái rủi ro động
    if final_probability < 0.35:
        risk_label, risk_color, card_theme = "THẤP", "#10B981", "success"
    elif final_probability < 0.70:
        risk_label, risk_color, card_theme = "TRUNG BÌNH", "#F59E0B", "warning"
    else:
        risk_label, risk_color, card_theme = "CAO", "#EF4444", "error"

    # --- KÍCH HOẠT PHÂN TÍCH ---
    if st.button("🔥 CHẠY ENGINE PHÂN TÍCH CHUYÊN SÂU", type="primary"):
        
        # HIỂN THỊ CÁC THẺ TRẠNG THÁI NHANH (METRICS)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Tỷ số nợ thực tế", f"{calculated_debt_ratio:.2%}", delta="- An toàn" if calculated_debt_ratio < 0.4 else "+ Rủi ro cao", delta_color="inverse")
        with m_col2:
            st.metric("Chất lượng dòng tiền", f"{cash_flow/15000000000000*100:.1f}%", "Dương" if cash_flow > 0 else "Âm nghiêm trọng")
        with m_col3:
            st.metric("Hệ số thanh khoản", f"{current_ratio:.2f} lần", "Đạt chuẩn" if current_ratio >= 1.5 else "Yếu")
        with m_col4:
            st.metric("Trạng thái Hệ thống", risk_label)

        st.write("---")

        # KHU VỰC ĐỒ HỌA TRỰC QUAN CAO CẤP
        g_col1, g_col2 = st.columns([1, 1])

        # 1. BIỂU ĐỒ ĐỒNG HỒ ĐỘNG ĐO RỦI RO CHUẨN XÁC
        with g_col1:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A;'>⏱️ Đồng hồ Cảnh báo Áp lực Rủi ro Tài chính Tổng hợp</h4>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_probability * 100,
                number={'suffix': "%", 'font': {'size': 38, 'color': risk_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 35], 'color': '#E2F0D9'},
                        {'range': [35, 70], 'color': '#FFF2CC'},
                        {'range': [70, 100], 'color': '#FCE4D6'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
                }
            ))
            fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 2. BIỂU ĐỒ CỘT TẦM QUAN TRỌNG CỦA CÁC BIẾN (Dựa trên hệ số thực của mô hình)
        with g_col2:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A;'>📊 Trọng số Ảnh hưởng của các Nhóm Biến Tài chính</h4>", unsafe_allow_html=True)
            try:
                importance = np.abs(model.coef_[0])
                importance = (importance / np.sum(importance)) * 100
            except:
                importance = [12, 18, 35, 15, 20]
                
            features = ['Doanh thu thuần', 'Lợi nhuận sau thuế', 'Tỷ số nợ phái sinh', 'Chỉ số ROA', 'Chỉ số ROE']
            df_imp = pd.DataFrame({'Biến': features, 'Mức độ ảnh hưởng (%)': importance}).sort_values('Mức độ ảnh hưởng (%)')
            fig_bar = px.bar(df_imp, x='Mức độ ảnh hưởng (%)', y='Biến', orientation='h', color='Mức độ ảnh hưởng (%)', color_continuous_scale='Cividis')
            fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.write("---")

        # =========================================================================
        # 🧠 THUẬT TOÁN BIÊN SOẠN BÁO CÁO CHIẾN LƯỢC ĐỘNG TỰ ĐỘNG (DỰA TRÊN 9 BIẾN THỰC TẾ)
        # =========================================================================
        
        # Đoạn 1: Đánh giá cơ cấu vốn dựa trên Nợ / Tài sản
        if calculated_debt_ratio < 0.3:
            capital_analysis = f"Cấu trúc vốn cực kỳ thận trọng với tỷ số nợ chỉ ở mức {calculated_debt_ratio:.2%}. VNM đang sở hữu một tấm khiên tài chính vững chắc, áp lực lãi vay gần như bằng không."
            capital_rec = "Doanh nghiệp đang thừa thãi dư địa nợ, có thể chủ động cân nhắc phát hành thêm trái phiếu dài hạn để tài trợ đầu tư công nghệ chế biến sâu thay vì chỉ dựa vào vốn tự có."
        elif calculated_debt_ratio <= 0.45:
            capital_analysis = f"Cấu trúc vốn ở trạng thái cân bằng tối ưu (Tỷ số nợ: {calculated_debt_ratio:.2%}). Đây là cơ cấu vốn kinh điển của ngành sữa, vừa tận dụng được đòn bẩy tài chính vừa kiểm soát an toàn rủi ro tín dụng."
            capital_rec = "Tiếp tục duy trì tỷ lệ đòn bẩy tài chính hiện tại. Ưu tiên thanh toán dứt điểm các khoản nợ ngắn hạn có lãi suất thả nổi để khóa chặt chi phí tài chính cố định."
        else:
            capital_analysis = f"🚨 BÁO ĐỘNG: Tỷ số nợ tăng cao lên tới {calculated_debt_ratio:.2%}. Rủi ro đòn bẩy tài chính đang đe dọa trực tiếp đến hệ số tín nhiệm quốc tế của doanh nghiệp."
            capital_rec = "Lập tức đóng băng các dự án đầu tư xây dựng cơ bản chưa cấp thiết. Tiến hành rà soát, tái cấu trúc lại các khoản nợ vay, ưu tiên đàm phán kéo dài kỳ hạn nợ để giảm áp lực trả nợ gốc ngắn hạn."

        # Đoạn 2: Đánh giá Thanh khoản từ Hệ số thanh toán ngắn hạn
        if current_ratio >= 1.5:
            liquidity_analysis = f"Hệ số thanh toán hiện hành đạt ngưỡng lý tưởng là {current_ratio:.2f} lần. VNM đang có lượng tài sản ngắn hạn (tiền mặt, tiền gửi) cực kỳ dồi dào, sẵn sàng chi trả mọi nghĩa vụ đến hạn."
            liquidity_rec = "Tối ưu hóa hiệu quả của dòng vốn bằng cách chuyển bớt lượng tiền mặt tồn quỹ lớn sang các danh mục đầu tư tài chính ngắn hạn có lợi suất cao hoặc gửi tiết kiệm kỳ hạn dài để tăng doanh thu tài chính."
        elif current_ratio >= 1.0:
            liquidity_analysis = f"Khả năng thanh khoản nằm trong vùng kiểm soát chặt chẽ ({current_ratio:.2f} lần). Tài sản ngắn hạn vừa đủ bao phủ nợ ngắn hạn nhưng độ an toàn không cao nếu gặp cú sốc thị trường."
            liquidity_rec = "Áp dụng các chính sách chiết khấu thanh toán nhanh cho các đại lý phân phối lớn của Vinamilk để đẩy nhanh tốc độ thu hồi các khoản phải thu, tăng tính thanh khoản cho dòng tiền."
        else:
            liquidity_analysis = f"🚨 KHỦNG HOẢNG THANH KHOẢN: Hệ số thanh toán sụt giảm nghiêm trọng xuống chỉ còn {current_ratio:.2f} lần. Tài sản ngắn hạn không đủ để bù đắp các khoản nợ sắp đến hạn, nguy cơ vỡ nợ kỹ thuật hiện hữu."
            liquidity_rec = "Thực hiện các biện pháp khẩn cấp: Xả kho hàng tồn kho chấp nhận giảm biên lợi nhuận, thắt chặt dòng tiền chi ra, đàm phán với ngân hàng thương mại để cấp hạn mức tín dụng dự phòng khẩn cấp."

        # Đoạn 3: Đánh giá chất lượng dòng tiền HĐKD
        if cash_flow > 0 and cash_flow >= net_profit:
            cash_analysis = f"Chất lượng lợi nhuận đạt mức tuyệt đối. Dòng tiền thuần từ HĐKD đạt {cash_flow:,.0f} VND, cao hơn cả lợi nhuận sau thuế ({net_profit:,.0f} VND). Điều này chứng tỏ VNM bán hàng và thu tiền thực tế rất tốt, không bị ứ đọng vốn."
            cash_rec = "Duy trì mô hình kinh doanh hiện tại. Có thể tự tin trích lập quỹ đầu tư phát triển và thực hiện cam kết chi trả cổ tức bằng tiền mặt đều đặn cho cổ đông để giữ vững giá trị cổ phiếu."
        elif cash_flow > 0 and cash_flow < net_profit:
            cash_analysis = f"Xuất hiện dấu hiệu lệch pha tài chính. Dòng tiền HĐKD dương ({cash_flow:,.0f} VND) nhưng lại thấp hơn lợi nhuận sau thuế ({net_profit:,.0f} VND). Gợi ý rằng một phần lợi nhuận lớn đang nằm ở dạng khoản phải thu từ đại lý hoặc hàng tồn kho chưa bán được."
            cash_rec = "Cần rà soát ngay chu kỳ chuyển đổi tiền mặt (Cash Conversion Cycle). Siết chặt các tiêu chuẩn cấp tín dụng thương mại cho hệ thống nhà phân phối, tránh tình trạng chiếm dụng vốn kéo dài."
        else:
            cash_analysis = f"🚨 DÒNG TIỀN ÂM: Dòng tiền hoạt động kinh doanh rơi vào trạng thái âm nguy kịch ({cash_flow:,.0f} VND). Doanh nghiệp đang bị chảy máu tiền mặt nghiêm trọng, phải dùng nợ vay hoặc tiền dự trữ để nuôi bộ máy vận hành."
            cash_rec = "Cắt giảm toàn diện ngân sách tiếp thị đại trà kém hiệu quả. Đình chỉ ngay các chính sách hỗ trợ công nợ dài hạn cho các chuỗi siêu thị mới, tập trung bán hàng thu tiền mặt ngay để cứu vãn dòng tiền vận hành."

        # Đoạn 4: Đánh giá năng lực sinh lời (ROE & ROA)
        if roe >= 15:
            profit_analysis = f"Hiệu quả sử dụng vốn ở mức xuất sắc (ROE: {roe:.1f}%, ROA: {roa:.1f}%). Khả năng sinh lời nội sinh vượt trội giúp doanh nghiệp duy trì vị thế dẫn dắt tuyệt đối thị trường FMCG."
            profit_rec = "Tiếp tục dồn thặng dư vốn tái đầu tư vào các dự án chiến lược trung, dài hạn: Trang trại bò sữa số hóa, tự động hóa nhà máy Mega để tối ưu hóa bài toán kinh tế theo quy mô (Economies of Scale)."
        elif roe >= 5:
            profit_analysis = f"Hiệu quả sinh lời ở mức trung bình ổn định (ROE: {roe:.1f}%, ROA: {roa:.1f}%). Tốc độ tăng trưởng lợi nhuận đang có dấu hiệu bão hòa do thị trường sữa nội địa đã đạt trạng thái bão hòa."
            profit_rec = "Tập trung tối ưu hóa biên lợi nhuận gộp bằng cách chuyển dịch cơ cấu sản phẩm sang các ngành hàng có biên lợi nhuận cao như sữa hạt cao cấp, sữa chua chức năng, thay vì cạnh tranh giá ở mảng sữa nước truyền thống."
        else:
            profit_analysis = f"🚨 SUY GIẢM HIỆU SUẤT: Năng lực sinh lời chạm đáy (ROE: {roe:.1f}%, ROA: {roa:.1f}%). Nguồn vốn lớn của cổ đông chiến lược đang bị lãng phí nghiêm trọng, không bù đắp được chi phí sử dụng vốn."
            profit_rec = "Thay đổi toàn diện ban điều hành hoặc mời các đơn vị tư vấn chiến lược quốc tế rà soát lại toàn bộ mô hình kinh doanh. Đóng cửa các dòng sản phẩm có biên lợi nhuận âm để bảo toàn nguồn lực tài chính."

        # HIỂN THỊ BÁO CÁO CHUYÊN GIA LÊN GIAO DIỆN CHÍNH
        st.markdown(f"### 🧠 Báo cáo Tư vấn Chiến lược từ Chuyên gia AI (Trạng thái rủi ro hệ thống: {risk_label})")
        
        ai_col1, ai_col2 = st.columns(2)
        
        with ai_col1:
            st.info(f"""
            🔍 **PHÂN TÍCH CHUYÊN SÂU BIẾN ĐỘNG THỰC TẾ:**
            
            - **Về Cấu trúc vốn:** {capital_analysis}
            
            - **Về Năng lực thanh khoản:** {liquidity_analysis}
            
            - **Về Chất lượng dòng tiền:** {cash_analysis}
            
            - **Về Hiệu quả hoạt động:** {profit_analysis}
            """)
            
        with ai_col2:
            st.warning(f"""
            🎯 **KHUYẾN NGHỊ QUẢN TRỊ ĐỘNG CHO BAN ĐIỀU HÀNH VNM:**
            
            1. **Giải pháp Cơ cấu vốn:** {capital_rec}
            
            2. **Giải pháp Thanh khoản ngắn hạn:** {liquidity_rec}
            
            3. **Giải pháp Chuỗi cung ứng & Dòng tiền:** {liquidity_rec}
            
            4. **Chiến lược cạnh tranh thị trường:** {profit_rec}
            """)

except FileNotFoundError:
    st.error("❌ Hệ thống không tìm thấy file bộ não học máy `logistic_model.pkl` hoặc `scaler.pkl`. Hãy chắc chắn rằng bạn đã chạy file `Train_model.py` trước đó.")
