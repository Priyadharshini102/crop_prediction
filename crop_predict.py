import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# Load your data
data = pd.read_csv("Cleaned_Crop_data.csv")
df_model=data

# Calculate Production in tonnes
data["Production"] = (data["Area harvested"] * data["Yield"]) / 1000
from sklearn.preprocessing import LabelEncoder

area_encoder = LabelEncoder()
item_encoder = LabelEncoder()

data = data.dropna(subset=["Production"])

data["Area_encoded"] = area_encoder.fit_transform(data["Area"])
data["Item_encoded"] = item_encoder.fit_transform(data["Item"])
X = data[["Area_encoded", "Item_encoded", "Year", "Area harvested", "Yield"]]
y = data["Production"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "crop_model.pkl")
joblib.dump(area_encoder, "area_encoder.pkl")
joblib.dump(item_encoder, "item_encoder.pkl")

# Load model and encoders
model = joblib.load("crop_model.pkl")
area_encoder = joblib.load("area_encoder.pkl")
item_encoder = joblib.load("item_encoder.pkl")

st.title("🌾 Crop Production Predictor")

# Dropdowns
area = st.selectbox("Select Area", sorted(data["Area"].unique()))
item = st.selectbox("Select Crop", sorted(data["Item"].unique()))
year = st.slider("Select Year", int(data["Year"].min()), int(data["Year"].max()))
area_harvested = st.number_input("Area harvested", min_value=0.0)
yield_val = st.number_input("Yield ", min_value=0.0)

# Prediction
if st.button("📊 Predict Production"):
    # Encode categorical variables
    area_encoded = area_encoder.transform([area])[0]
    item_encoded = item_encoder.transform([item])[0]

    # Prepare input vector
    input_vector = [[area_encoded, item_encoded, year, area_harvested, yield_val]]

    # Make prediction
    prediction = model.predict(input_vector)

    st.success(f"🌾 Predicted Production : {prediction[0]:,.2f} tonnes")

st.header("📊 Crop Insights Dashboard")
col1, col2 = st.columns(2)

with col1:
        st.subheader("Top Agricultural Regions")
        region_df = df_model.groupby('Area')['Production'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_area = px.bar(region_df, x='Production', y='Area', orientation='h', color='Area', title="Top Producing Regions", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_area, use_container_width=True)

with col2:
        st.subheader("Crop Distribution")
        crop_df = df_model['Item'].value_counts().head(10).reset_index()
        crop_df.columns = ['Crop', 'Count']
        fig_crop = px.pie(crop_df, names='Crop', values='Count', title="Top Cultivated Crops", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_crop, use_container_width=True)

st.subheader("📅 Yearly Trend Analysis")
yearly_avg = df_model.groupby('Year')[['Area harvested', 'Production']].mean().reset_index()
fig_yearly = px.line(yearly_avg, x='Year', y=['Area harvested', 'Production'], markers=True, title="Yearly Crop Trends", color_discrete_map={'Area harvested':'#f39c12', 'Production':'#2ecc71'})
st.plotly_chart(fig_yearly, use_container_width=True)
