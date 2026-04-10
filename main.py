# main.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ------------------------------
# Title
# ------------------------------
st.title("🎓 AI Student Planner")
st.subheader("Predict your Exam Score and get suggestions to improve performance")

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")  # Make sure your dataset is here
    return df

df = load_data()
st.write("Dataset Preview:")
st.dataframe(df.head())

# ------------------------------
# Feature Selection
# ------------------------------
features = ['StudyHours', 'Attendance', 'Motivation', 'StressLevel', 
            'AssignmentCompletion', 'OnlineCourses']
target = 'FinalGrade'  # or 'ExamScore'

X = df[features]
y = df[target]

# ------------------------------
# Feature Engineering
# ------------------------------
# Polynomial features to capture interaction
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Scaling features
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# ------------------------------
# Train Model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_scaled, y, test_size=0.2, random_state=42
)

# Optimized Random Forest
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
st.write("✅ Model trained successfully!")
st.write("📊 Model Accuracy (R² score):", round(accuracy, 3))

# ------------------------------
# User Inputs
# ------------------------------
st.header("--- Enter Your Details ---")
study = st.number_input("Study Hours per day", min_value=0, max_value=24, value=5)
attend = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)
motive = st.slider("Motivation (1-10)", min_value=1, max_value=10, value=8)
stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
assign = st.slider("Assignment Completion (%)", min_value=0, max_value=100, value=70)
courses = st.number_input("Online Courses done", min_value=0, max_value=20, value=3)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Score"):
    input_data = pd.DataFrame([[study, attend, motive, stress, assign, courses]],
                              columns=features)
    
    # Apply same transformations as training
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)
    
    pred = model.predict(input_scaled)[0]
    st.write(f"📊 Predicted Score: {round(pred, 1)}")

    # --------------------------
    # Suggestions
    # --------------------------
    if pred >= 85:
        st.success("🌟 Excellent performance! Keep it up!")
    elif pred >= 60:
        st.info("👍 Moderate performance, improve consistency")
        st.info("🧘 Reduce stress and take breaks")
        st.info("📚 Increase study hours and complete assignments on time")
    else:
        st.warning("⚠ Low performance, focus on study plan and motivation")
        st.warning("📚 Increase study hours, reduce distractions")
        st.warning("🧘 Manage stress and participate in activities wisely")