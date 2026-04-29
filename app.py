import streamlit as st
import joblib
import re
import os
import pandas as pd
from datetime import datetime

# ---------- LOAD ML MODEL & VECTORIZER ----------
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base, "model.pkl"))
    vectorizer = joblib.load(os.path.join(base, "vectorizer.pkl"))
    return model, vectorizer

model, vectorizer = load_model()

urgent_words = ["urgent", "immediately", "hacked", "fraud", "unauthorized"]

# ---------- FUNCTIONS ----------
def predict_fraud(text):
    vec = vectorizer.transform([text.lower()])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    return ("Fraudulent Activity" if pred == 1 else "Normal Activity", prob)

def detect_urgency(text):
    return min(sum([0.2 for w in urgent_words if w in text.lower()]), 1)

def extract_amount(text):
    # Match currency symbols + numbers, or plain numbers
    match = re.findall(r"[\$\£\€\₹]?\s?\d[\d,\.]*", text)
    return match[0].strip() if match else "Not Detected"

def calculate_risk(fraud_score, urgency_score, amount):
    amount_factor = 0.8 if amount != "Not Detected" else 0.3
    risk = (fraud_score * 0.5) + (urgency_score * 0.3) + (amount_factor * 0.2)
    return round(risk * 100, 2)

def risk_label(score):
    if score > 70: return "High"
    if score > 40: return "Medium"
    return "Low"

# ---------- STATE ----------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["time","text","label","confidence","urgency","amount","risk","risk_level"]
    )

# ---------- PAGE ----------
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center;'>💳 Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Real-time analysis of customer messages using NLP</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- INPUT BAR ----------
c1, c2 = st.columns([5,1])
with c1:
    user_input = st.text_area("Enter Message", height=120, placeholder="Paste customer message here...")
with c2:
    st.write("")
    st.write("")
    run = st.button("🔍 Analyze", use_container_width=True)

# ---------- PROCESS ----------
if run and user_input.strip():
    label, conf = predict_fraud(user_input)
    urg = detect_urgency(user_input)
    amt = extract_amount(user_input)
    risk = calculate_risk(conf, urg, amt)
    rlab = risk_label(risk)

    new_row = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "text": user_input,
        "label": label,
        "confidence": round(conf,2),
        "urgency": round(urg,2),
        "amount": amt,
        "risk": risk,
        "risk_level": rlab
    }
    st.session_state.history = pd.concat(
        [pd.DataFrame([new_row]), st.session_state.history],
        ignore_index=True
    )
elif run and not user_input.strip():
    st.warning("Please enter a message before analyzing.")

# ---------- KPIs ----------
df = st.session_state.history

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Messages", len(df))
k2.metric("Fraud Count", int((df["label"]=="Fraudulent Activity").sum()) if len(df)>0 else 0)
k3.metric("Avg Risk %", round(df["risk"].mean(),1) if len(df)>0 else 0)
k4.metric("High Risk Alerts", int((df["risk_level"]=="High").sum()) if len(df)>0 else 0)

st.markdown("---")

# ---------- MAIN DASHBOARD ----------
left, right = st.columns([2,1])

# LEFT: Charts + Table
with left:
    st.subheader("📊 Risk Distribution")
    if len(df) > 0:
        st.bar_chart(df["risk_level"].value_counts())
    else:
        st.info("No data yet — analyze a message to get started.")

    st.subheader("📈 Confidence Trend")
    if len(df) > 0:
        st.line_chart(df["confidence"])
    else:
        st.info("No data yet")

    st.subheader("📋 Recent Analyses")
    if len(df) > 0:
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("No data yet")

# RIGHT: Gauge + Filters
with right:
    st.subheader("⚠️ Latest Risk Gauge")
    if len(df) > 0:
        latest = df.iloc[0]
        st.progress(latest["risk"]/100)
        st.write(f"**{latest['risk']}% — {latest['risk_level']} Risk**")

        if latest["risk_level"] == "High":
            st.error("🚨 Immediate attention required")
        elif latest["risk_level"] == "Medium":
            st.warning("⚠️ Needs review")
        else:
            st.success("✅ Safe")
    else:
        st.info("Run analysis to see gauge")

    st.markdown("---")
    st.subheader("🔎 Filters")
    if len(df) > 0:
        level = st.selectbox("Risk Level", ["All","High","Medium","Low"])
        if level != "All":
            filtered = df[df["risk_level"]==level]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.write("No data to filter")

# ---------- SIDEBAR ----------
st.sidebar.title("📌 About")
st.sidebar.markdown("""
**Model**: TF-IDF + Logistic Regression  
**Accuracy**: ~88%  
**Features Analyzed**:
- 🔴 Fraud probability score
- ⏱ Urgency detection
- 💰 Amount extraction (Regex)
- 📉 Composite risk score

---
**Risk Levels**:
- 🔴 High: > 70%
- 🟡 Medium: 40–70%
- 🟢 Low: < 40%
""")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · NLP Fraud Detection Project")