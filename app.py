import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 📊 Title
st.title("📚 AI Study Planner")

# 📂 Load dataset
df = pd.read_csv("dataset.csv")

# 🎯 Features & Target
X = df[['StudyHours', 'Attendance', 'Motivation', 
        'StressLevel', 'AssignmentCompletion', 'OnlineCourses']]

y = df['ExamScore']

# 🤖 Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

# 📊 Accuracy
st.write("Model Accuracy:", round(model.score(X_test, y_test), 2))

# 🧾 User Inputs
st.subheader("Enter Your Details")

study = st.slider("Study Hours", 0, 12, 4)
attend = st.slider("Attendance (%)", 0, 100, 70)
motive = st.slider("Motivation (1-10)", 1, 10, 5)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
assign = st.slider("Assignment Completion (%)", 0, 100, 60)
courses = st.slider("Online Courses Done", 0, 10, 1)

# 🚀 Predict Button
if st.button("Generate Study Plan"):
    pred = model.predict([[study, attend, motive, stress, assign, courses]])[0]

    st.subheader(f"📊 Predicted Score: {round(pred, 2)}")

    # Suggestions
    if pred < 50:
        st.warning("⚠️ Increase study hours and complete more assignments")
    elif pred < 75:
        st.info("👍 Moderate performance, improve consistency")
    else:
        st.success("🔥 Great performance! Maintain your strategy")

    if stress > 7:
        st.warning("🧘 Reduce stress and take breaks")

    if attend < 70:
        st.warning("📅 Improve attendance")

    if assign < 60:
        st.warning("📝 Focus more on assignments")