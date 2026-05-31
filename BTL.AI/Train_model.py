import joblib
import streamlit as st


@st.cache_resource
def load_ml_assets():
    # Đi thẳng từ thư mục gốc của Repo vào thẳng thư mục BTL.AI để lấy file
    model_path = "BTL.AI/logistic_model.pkl"
    scaler_path = "BTL.AI/scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler
# 1. Đọc dữ liệu từ 3 doanh nghiệp 
df_vnm = pd.read_excel("VNM.xlsx")
df_mch = pd.read_excel("MCH.xlsx")
df_mcm = pd.read_excel("MCM.xlsx")

# 2. Gộp dữ liệu ngành sữa
df_industry = pd.concat([df_vnm, df_mch, df_mcm], ignore_index=True)

# 3. Chuyển đổi Quý sang định dạng ngày chuẩn MM/DD/YYYY
def convert_quarter_to_date(q_str):
    if pd.isna(q_str): return q_str
    q, year = str(q_str).split('/')
    month = {'Q1': '03/31', 'Q2': '06/30', 'Q3': '09/30', 'Q4': '12/31'}.get(q, '12/31')
    return f"{month}/{year}"

df_industry['Thời gian'] = df_industry['Thời gian'].apply(convert_quarter_to_date)

# 4. Tạo biến mục tiêu Y 
df_industry['Rui_ro'] = np.where((df_industry['Tỷ số nợ'] > 0.35) | (df_industry['ROE'] < 0.06), 1, 0)

# 5. Lựa chọn các biến độc lập (X)
features = ['Doanh thu thuần', 'Lợi nhuận sau thuế', 'Tỷ số nợ', 'ROA', 'ROE']
df_industry = df_industry.dropna(subset=features)
# 6. Xử lý dữ liệu khuyết thiếu

X = df_industry[features]
y = df_industry['Rui_ro']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4.3 Khởi tạo và Huấn luyện mô hình ---
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Lưu lại mô hình và scaler để dùng cho việc chạy App
joblib.dump(model, os.path.join(BASE_DIR, 'logistic_model.pkl'))
joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))

# --- 4.4 Đánh giá mô hình ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))

# --- 4.5 Trực quan hóa ---
# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Dự đoán (Predicted)')
plt.ylabel('Thực tế (Actual)')
plt.show()

# 2. Feature Importance (Hệ số của mô hình)
importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
importance = importance.sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=importance)
plt.title('Độ quan trọng của các biến tài chính (Feature Importance)')
plt.show()
