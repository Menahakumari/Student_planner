# 🎓 AI Student Planner

## Overview

AI Student Planner is a Streamlit-based machine learning application that predicts a student's academic performance using study-related factors and provides suggestions to improve learning outcomes.

The project includes two implementations:

* `app.py` - Uses a Linear Regression model to predict exam scores.
* `main.py` - Uses a Random Forest Regressor with feature engineering techniques for prediction.

The application helps students understand how factors such as study hours, attendance, motivation, stress levels, assignment completion, and online learning activities may influence academic performance.

---

## Features

* Predict academic performance based on user inputs
* Interactive Streamlit user interface
* Machine learning-based score prediction
* Personalized study recommendations
* Analysis of important academic factors
* Real-time prediction results

---

## Files

### app.py

Implements:

* Dataset loading using Pandas
* Linear Regression model training
* Exam score prediction
* Personalized recommendations based on predicted performance

### main.py

Implements:

* Random Forest Regressor model
* Polynomial feature generation
* Feature scaling using StandardScaler
* Final grade prediction
* Performance improvement suggestions

### dataset.csv

Contains student-related data used for training the machine learning models.

Example attributes include:

* StudyHours
* Attendance
* Motivation
* StressLevel
* AssignmentCompletion
* OnlineCourses
* ExamScore
* FinalGrade

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn

---

## Machine Learning Models

### Linear Regression

Used in `app.py` to predict student exam scores based on selected study-related features.

### Random Forest Regressor

Used in `main.py` along with polynomial feature engineering to improve prediction performance.

---

## How to Run

### Run the Linear Regression Version

```bash
streamlit run app.py
```

### Run the Random Forest Version

```bash
streamlit run main.py
```

---

## Input Parameters

Users can provide:

* Study Hours
* Attendance Percentage
* Motivation Level
* Stress Level
* Assignment Completion Percentage
* Number of Online Courses Completed

The application uses these inputs to generate performance predictions and study recommendations.

---

## Prediction Feedback

Based on the predicted score or grade, the application provides recommendations such as:

* Increasing study hours
* Improving assignment completion
* Reducing stress levels
* Improving attendance
* Maintaining current study strategies

---

## Purpose

This project demonstrates the application of machine learning in education by helping students evaluate study habits and receive personalized suggestions for academic improvement.
