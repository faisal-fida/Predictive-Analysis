import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import List

# Set page config
st.set_page_config(page_title="Sales Prediction App", layout="wide")


# Load the trained model
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
model_columns = [
    "customer_number",
    "salesman_name",
    "subtotal_amount",
    "gst_amount",
    "total_amount",
    "paid_amount",
    "item_number",
    "product_price",
    "salesman_territory_002-PSS09",
    "salesman_territory_002-PSS10",
    "salesman_territory_002-PSS11",
    "salesman_territory_002-PSS13",
    "due_date_day_of_week",
    "due_date_month",
    "due_date_year",
]

# Sidebar
st.sidebar.title("Input Parameters")


# Helper function to create input fields
def create_input_field(label, key, min_value=0, max_value=None, value=0):
    return st.sidebar.number_input(
        label, min_value=min_value, max_value=max_value, value=value, key=key
    )


# Create a dataframe from user inputs
customer_number = create_input_field(
    "Customer Number", "customer_number", min_value=1, max_value=1000, value=1
)
salesman_name = st.sidebar.text_input("Salesman Name", key="salesman_name")
subtotal_amount = create_input_field("Subtotal Amount", "subtotal_amount")
gst_amount = create_input_field("GST Amount", "gst_amount")
total_amount = create_input_field("Total Amount", "total_amount")
paid_amount = create_input_field("Paid Amount", "paid_amount")
item_number = create_input_field("Item Number", "item_number", min_value=1, value=1)
product_price = create_input_field("Product Price", "product_price")
province = st.sidebar.selectbox(
    "Province", ["NCR, SECOND DISTRICT", "RIZAL"], key="province"
)
salesman_territory = st.sidebar.selectbox(
    "Salesman Territory",
    [
        "002-PSS09",
        "002-PSS10",
        "002-PSS11",
        "002-PSS13",
    ],
    key="salesman_territory",
)

weekdays_to_numbers = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}
due_date_day_of_week = st.sidebar.selectbox(
    "Due Date Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    key="due_date_day_of_week",
)
due_date_day_of_week_num = weekdays_to_numbers[due_date_day_of_week]
due_date_month = st.sidebar.slider("Due Date Month", 1, 12, 1, key="due_date_month")
due_date_year = st.sidebar.slider(
    "Due Date Year", 2023, 2024, 2023, key="due_date_year"
)

# line_number = create_input_field("Line Number", "line_number", min_value=1, value=1)
# product_name = st.sidebar.text_input("Product Name", key="product_name")
# base_quantity = create_input_field("Base Quantity", "base_quantity")
# district = st.sidebar.text_input("District", key="district")
# credit_limit = create_input_field("Credit Limit", "credit_limit")
# invoice_number = create_input_field(
#     "Invoice Number", "invoice_number", min_value=1, max_value=1000, value=1
# )
# discount = create_input_field(
#     "Discount", "discount", min_value=0, max_value=100, value=0
# )
# unit_of_measurement = st.sidebar.selectbox(
#     "Unit of Measurement", ["C50", "PC", "PCK"], key="unit_of_measurement"
# )

# Main content
st.title("Sales Prediction App")


def create_input_dataframe():
    data = {
        # "Discount": discount,
        # "line_number": line_number,
        # "product_name": product_name,
        # "base_quantity": base_quantity,
        # "district": district,
        # "CreditLimit": credit_limit,
        # "invoice_number": invoice_number,
        # f"unit_of_measurement_{unit_of_measurement}": 1,
        # "invoice_date_day_of_week": invoice_date.weekday(),
        # "invoice_date_month": invoice_date.month,
        # "invoice_date_day_of_month": invoice_date.day,
        # "delivery_order_date_day_of_week": delivery_order_date.weekday(),
        # "delivery_order_date_month": delivery_order_date.month,
        # "delivery_order_date_day_of_month": delivery_order_date.day,
        # "delivery_date_day_of_week": delivery_date.weekday(),
        # "delivery_date_month": delivery_date.month,
        # "delivery_date_day_of_month": delivery_date.day,
        "customer_number": customer_number,
        "salesman_name": salesman_name,
        "subtotal_amount": subtotal_amount,
        "gst_amount": gst_amount,
        "total_amount": total_amount,
        "paid_amount": paid_amount,
        "item_number": item_number,
        "product_price": product_price,
        f"salesman_territory_{salesman_territory}": 1,
        "due_date_day_of_week": due_date_day_of_week_num,
        "due_date_month": due_date_month,
        "due_date_year": due_date_year,
    }

    df = pd.DataFrame(data, index=[0])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["customer_number", "salesman_name"]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df


if st.sidebar.button("Predict"):
    input_df = create_input_dataframe()
    input_df = encode_categorical_columns(input_df)

    columns_to_scale = [
        "subtotal_amount",
        "gst_amount",
        "total_amount",
        "paid_amount",
        "product_price",
    ]

    input_df = scale_features(input_df, columns_to_scale)

    # match the order
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]

    st.header("Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Quantity", f"{prediction:.2f}")
    col2.metric("Rounded Prediction", f"{round(prediction)}")

else:
    st.info(
        "Please fill in the input fields and click 'Predict' to get the sales prediction."
    )
