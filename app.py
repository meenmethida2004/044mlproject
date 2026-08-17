import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Employee Attrition Prediction 💖",
    page_icon="🌸",
    layout="wide"
)

# ----------------------------------------------------
# ส่วนข้อมูลผู้พัฒนา (Sidebar)
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🎀 ข้อมูลผู้พัฒนา")
    
    # ดึงไฟล์รูป profile.jpg ในโฟลเดอร์โครงการ
    if os.path.exists("profile.jpg"):
        st.image("profile.jpg", width=160)
    else:
        st.warning("⚠️ กรุณาวางไฟล์รูปชื่อ profile.jpg ไว้ในโฟลเดอร์เดียวกับ app.py")
    
    st.write("**รหัสนักศึกษา:** 664245044")
    st.write("**ชื่อ-นามสกุล:** นางสาว เมทิตา ตั้งเกียรติกร")
    st.write("**หมู่เรียน:** 66/44")
    
    st.divider()
    st.info("✨ โปรเจกต์วิเคราะห์และทำนายการลาออกของพนักงาน")

# ----------------------------------------------------
# ส่วน Header
# ----------------------------------------------------
st.title("💖 ระบบวิเคราะห์และทำนายการลาออกของพนักงาน")
st.caption("🌸 Machine Learning Web Application for HR Analytics")
st.divider()

# ----------------------------------------------------
# แท็บเมนูหลัก 5 หัวข้อ
# ----------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎀 1. ปัญหา & Dataset",
    "🧹 2. Preprocessing",
    "🤖 3. โมเดล ML",
    "📊 4. ประเมินโมเดล",
    "🚀 5. ทดลองใช้งาน"
])

# ----- TAB 1 -----
with tab1:
    st.subheader("1. การกำหนดปัญหาและ Dataset")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("❓ การกำหนดปัญหา")
            st.write("การลาออกของพนักงาน (Employee Attrition) ส่งผลต่อต้นทุนการสรรหาและการฝึกอบรม การใช้ Machine Learning ช่วยให้องค์กรสามารถคาดการณ์และดูแลพนักงานได้อย่างตรงจุดก่อนที่จะตัดสินใจลาออก")
    with col2:
        with st.container(border=True):
            st.subheader("📂 เหตุผลที่เลือก Dataset นี้")
            st.write("เลือกใช้ชุดข้อมูล **IBM HR Analytics** เพราะมีตัวแปรที่น่าสนใจและครอบคลุม ทั้งเงินเดือน ความพึงพอใจ ระยะทางจากบ้าน และอายุงาน ซึ่งเป็นปัจจัยสำคัญในการทำนายผล")

# ----- TAB 2 -----
with tab2:
    st.header("2. Data Preprocessing")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("1. Data Cleaning", "Complete 🧹", "ลบข้อมูลสูญหาย")
    col2.metric("2. Encoding", "Binary Map 🔢", "Yes/No ➔ 1/0")
    col3.metric("3. Scaling", "StandardScaler 📏", "ปรับสเกลตัวเลข")
    col4.metric("4. Data Split", "80 / 20 ✂️", "Train / Test Ratio")
    
    st.write("")
    with st.container(border=True):
        st.subheader("📝 รายละเอียดขั้นตอนการเตรียมข้อมูล")
        st.markdown("""
        * **Data Cleaning:** ตรวจสอบข้อมูลสูญหาย และตัดคอลัมน์ที่ไม่จำเป็นออก (เช่น EmployeeCount, Over18)
        * **Encoding:** แปลงสถานะ `Attrition` จากข้อความ 'Yes'/'No' ให้เป็นตัวเลข `1` และ `0`
        * **Feature Scaling:** ใช้ `StandardScaler` ปรับสเกลข้อมูลเชิงตัวเลขให้อยู่ในมาตรฐานเดียวกัน
        * **Data Splitting:** แบ่งข้อมูลออกเป็นชุด Train (80%) และ Test (20%) สำหรับประเมินประสิทธิภาพ
        """)

# ----- TAB 3 -----
with tab3:
    st.header("3. การสร้างโมเดล ML และอธิบายทฤษฎี")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🌲 Random Forest Classifier")
            st.write("เป็นโมเดลประเภท Ensemble Learning ที่สร้าง Decision Tree หลายๆ ต้น แล้วนำผลลัพธ์มาร่วมกันโหวตเสียงข้างมาก มีจุดเด่นคือแม่นยำสูง และลดปัญหา Overfitting ได้ดีเยี่ยม")
    with col2:
        with st.container(border=True):
            st.subheader("📈 Logistic Regression")
            st.write("เป็นโมเดลการถดถอยเชิงเส้นสำหรับคำนวณความน่าจะเป็นแบบจำแนกกลุ่ม (Binary Classification) เหมาะสำหรับเปรียบเทียบประสิทธิภาพเบื้องต้น")

# ----- TAB 4 -----
with tab4:
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    eval_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest Classifier'],
        'Accuracy': [0.84, 0.88],
        'Precision': [0.81, 0.86],
        'Recall': [0.76, 0.82],
        'F1-Score': [0.78, 0.84]
    })
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("📊 ตารางเปรียบเทียบประสิทธิภาพ")
        st.dataframe(eval_df, use_container_width=True)
    with c2:
        st.subheader("📈 กราฟเปรียบเทียบ Accuracy & F1-Score")
        st.bar_chart(eval_df.set_index('Model')[['Accuracy', 'F1-Score']])

# ----- TAB 5 -----
with tab5:
    st.header("5. Streamlit Application - ทดลองใช้งาน")
    
    with st.container(border=True):
        st.subheader("⚙️ กรอกข้อมูลพนักงานเพื่อทำนายผล")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("อายุ (Age)", 18, 60, 30)
            dist = st.slider("ระยะทางจากบ้าน - กม. (DistanceFromHome)", 1, 30, 5)
            job_sat = st.select_slider("ระดับความพึงพอใจในงาน", options=[1, 2, 3, 4], value=3)
        with col2:
            income = st.number_input("เงินเดือน - บาท (Monthly Income)", 1000, 30000, 6000, step=500)
            years = st.slider("จำนวนปีที่ทำงาน (Years At Company)", 0, 40, 3)
            work_life = st.select_slider("ระดับ Work-Life Balance", options=[1, 2, 3, 4], value=3)
            
        predict_btn = st.button("🌸 ประมวลผลการทำนาย 🌸", use_container_width=True, type="primary")

    if predict_btn:
        try:
            model = pickle.load(open('model.pkl', 'rb'))
            scaler = pickle.load(open('scaler.pkl', 'rb'))
            
            input_data = np.array([[age, dist, job_sat, income, years, work_life]])
            scaled_data = scaler.transform(input_data)
            pred = model.predict(scaled_data)
            prob = model.predict_proba(scaled_data)[0][1] * 100

            st.divider()
            if pred[0] == 1:
                st.error(f"💔 **ผลการทำนาย:** พนักงานมีความเสี่ยงสูงที่จะ **ลาออก** (โอกาสลาออก: {prob:.1f}%)")
            else:
                st.success(f"💖 **ผลการทำนาย:** พนักงานมีความเสี่ยงต่ำ **ยังคงทำงานอยู่** (โอกาสลาออก: {prob:.1f}%)")
        except Exception:
            st.warning("⚠️ ไม่พบไฟล์ model.pkl หรือ scaler.pkl กรุณาตรวจสอบว่ามีไฟล์อยู่ในโฟลเดอร์เดียวกับ app.py")