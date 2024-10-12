import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px


# Load the trained model
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# Set page config
st.set_page_config(page_title="Sales Prediction App", layout="wide")

# Sidebar
st.sidebar.title("Input Parameters")


# Helper function to create input fields
def create_input_field(label, key, min_value=0, max_value=None, value=0):
    return st.sidebar.number_input(
        label, min_value=min_value, max_value=max_value, value=value, key=key
    )


# Main input fields
customer_number = create_input_field("Customer Number", "customer_number", min_value=1)
district = st.sidebar.text_input("District", key="district")
credit_limit = create_input_field("Credit Limit", "credit_limit")
invoice_number = create_input_field("Invoice Number", "invoice_number", min_value=1)
salesman_name = st.sidebar.text_input("Salesman Name", key="salesman_name")
discount = create_input_field(
    "Discount", "discount", min_value=0, max_value=100, value=0
)
subtotal_amount = create_input_field("Subtotal Amount", "subtotal_amount")
gst_amount = create_input_field("GST Amount", "gst_amount")
total_amount = create_input_field("Total Amount", "total_amount")
paid_amount = create_input_field("Paid Amount", "paid_amount")
line_number = create_input_field("Line Number", "line_number", min_value=1)
item_number = create_input_field("Item Number", "item_number", min_value=1)
product_price = create_input_field("Product Price", "product_price")
product_name = st.sidebar.text_input("Product Name", key="product_name")
base_quantity = create_input_field("Base Quantity", "base_quantity")

# Dropdown for categorical variables
province = st.sidebar.selectbox(
    "Province", ["NCR, SECOND DISTRICT", "RIZAL"], key="province"
)
salesman_territory = st.sidebar.selectbox(
    "Salesman Territory",
    [
        "002-PSS05",
        "002-PSS06",
        "002-PSS07",
        "002-PSS08",
        "002-PSS09",
        "002-PSS10",
        "002-PSS11",
        "002-PSS12",
        "002-PSS13",
    ],
    key="salesman_territory",
)
unit_of_measurement = st.sidebar.selectbox(
    "Unit of Measurement", ["C50", "PC", "PCK"], key="unit_of_measurement"
)

# Date inputs
invoice_date = st.sidebar.date_input("Invoice Date", key="invoice_date")
delivery_order_date = st.sidebar.date_input(
    "Delivery Order Date", key="delivery_order_date"
)
delivery_date = st.sidebar.date_input("Delivery Date", key="delivery_date")
due_date = st.sidebar.date_input("Due Date", key="due_date")

# Main content
st.title("Sales Prediction App")


# Create a dataframe from user inputs
def create_input_dataframe():
    data = {
        "customer_number": customer_number,
        "district": district,
        "CreditLimit": credit_limit,
        "invoice_number": invoice_number,
        "salesman_name": salesman_name,
        "Discount": discount,
        "subtotal_amount": subtotal_amount,
        "gst_amount": gst_amount,
        "total_amount": total_amount,
        "paid_amount": paid_amount,
        "line_number": line_number,
        "item_number": item_number,
        "product_price": product_price,
        "product_name": product_name,
        "base_quantity": base_quantity,
        f"province_{province}": 1,
        f"salesman_territory_{salesman_territory}": 1,
        f"unit_of_measurement_{unit_of_measurement}": 1,
        "invoice_date_day_of_week": invoice_date.weekday(),
        "invoice_date_month": invoice_date.month,
        "invoice_date_day_of_month": invoice_date.day,
        "delivery_order_date_day_of_week": delivery_order_date.weekday(),
        "delivery_order_date_month": delivery_order_date.month,
        "delivery_order_date_day_of_month": delivery_order_date.day,
        "delivery_date_day_of_week": delivery_date.weekday(),
        "delivery_date_month": delivery_date.month,
        "delivery_date_day_of_month": delivery_date.day,
        "due_date_day_of_week": due_date.weekday(),
        "due_date_month": due_date.month,
        "due_date_day_of_month": due_date.day,
        "days_until_due": (due_date - invoice_date).days,
        "days_until_delivery": (delivery_date - invoice_date).days,
        "days_until_delivery_order": (delivery_order_date - invoice_date).days,
    }
    df = pd.DataFrame(data, index=[0])

    # Add missing columns with 0 values
    missing_columns = set(model.feature_names_in_) - set(df.columns)
    for col in missing_columns:
        df[col] = 0

    return df[model.feature_names_in_]


# Make prediction
if st.sidebar.button("Predict"):
    input_df = create_input_dataframe()

    # Scale the input features
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display prediction
    st.header("Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Quantity", f"{prediction:.2f}")
    col2.metric("Rounded Prediction", f"{round(prediction)}")
    col3.metric("Confidence", f"{model.predict_proba(input_scaled)[0].max():.2%}")

    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame(
        {"feature": model.feature_names_in_, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    fig = px.bar(
        feature_importance.head(10),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Most Important Features",
    )
    st.plotly_chart(fig)

    # Partial dependence plots
    st.subheader("Partial Dependence Plots")
    top_features = feature_importance["feature"].head(3).tolist()
    for feature in top_features:
        fig = px.line(
            x=np.linspace(input_df[feature].min(), input_df[feature].max(), 100),
            y=model.predict(
                input_df.assign(
                    **{
                        feature: np.linspace(
                            input_df[feature].min(), input_df[feature].max(), 100
                        )
                    }
                )
            ),
            title=f"Partial Dependence Plot for {feature}",
        )
        fig.update_layout(xaxis_title=feature, yaxis_title="Predicted Quantity")
        st.plotly_chart(fig)

else:
    st.info(
        "Please fill in the input fields and click 'Predict' to get the sales prediction."
    )

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.subheader("How to use this app:")
st.sidebar.markdown("""
1. Fill in the input fields in the sidebar.
2. Click the 'Predict' button to get the sales prediction.
3. View the prediction results, feature importance, and partial dependence plots in the main area.
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit")
