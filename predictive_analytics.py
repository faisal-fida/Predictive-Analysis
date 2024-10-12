import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_excel("data.xlsx")

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index

df.drop(cols_to_drop, axis=1, inplace=True)

cols_to_drop = df.columns[df.isnull().any()]

if df[cols_to_drop].value_counts().values.size > 0:
    print(df[cols_to_drop].value_counts())
else:
    df.drop(cols_to_drop, axis=1, inplace=True)

mostly_same_cols = df.columns

for col in mostly_same_cols:
    common_percentage = df[col].value_counts().max() / df.shape[0] * 100
    if common_percentage > 90:
        df.drop(col, axis=1, inplace=True)


column_mapping = {
    "CustNo": "customer_number",
    "CustName": "customer_name",
    "Address": "customer_address",
    "BarangayName": "district",
    "ProvinceName": "province",
    "SalesManTerritory": "salesman_territory",
    "InvNo": "invoice_number",
    "InvDt": "invoice_date",
    "DoDt": "delivery_order_date",
    "DeliveryDate": "delivery_date",
    "Salesman": "salesman_name",
    "DueDate": "due_date",
    "UOM": "unit_of_measurement",
    "ItemName": "product_name",
    "SubTotal": "subtotal_amount",
    "GstAmt": "gst_amount",
    "TotalAmt": "total_amount",
    "PaidAmt": "paid_amount",
    "Price": "product_price",
    "SubAmt": "sub_amount",
    "LineNo": "line_number",
    "ItemNo": "item_number",
    "Qty": "quantity",
    "BaseQty": "base_quantity",
    "ts": "timestamp",
}

df.rename(columns=column_mapping, inplace=True)

new_df = df.copy()


numeric_cols = new_df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    if (new_df[col] < 0).any():
        new_df[col] = new_df[col].abs()

new_df["district"] = new_df["district"].astype("category")
new_df["province"] = new_df["province"].astype("category")
new_df["salesman_territory"] = new_df["salesman_territory"].astype("category")
new_df["unit_of_measurement"] = new_df["unit_of_measurement"].astype("category")

new_df["invoice_date"] = pd.to_datetime(
    new_df["invoice_date"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
)
new_df["delivery_order_date"] = pd.to_datetime(
    new_df["delivery_order_date"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
)
new_df["delivery_date"] = pd.to_datetime(
    new_df["delivery_date"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
)
new_df["due_date"] = pd.to_datetime(
    new_df["due_date"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
)

new_df.drop(["customer_name", "customer_address", "timestamp"], axis=1, inplace=True)

new_df.head(2)

scaler = MinMaxScaler()
columns_to_scale = [
    "subtotal_amount",
    "gst_amount",
    "total_amount",
    "paid_amount",
    "product_price",
    "sub_amount",
    "base_quantity",
    "quantity",
]

new_df[columns_to_scale] = scaler.fit_transform(new_df[columns_to_scale])

label_encoder = LabelEncoder()
new_df["customer_number"] = label_encoder.fit_transform(new_df["customer_number"])
new_df["invoice_number"] = label_encoder.fit_transform(new_df["invoice_number"])
new_df["salesman_name"] = label_encoder.fit_transform(new_df["salesman_name"])
new_df["product_name"] = label_encoder.fit_transform(new_df["product_name"])

encoder = ce.CountEncoder(cols=["district"])
new_df["district"] = encoder.fit_transform(new_df["district"])

new_df = pd.get_dummies(new_df, columns=["province"], drop_first=True)
new_df = pd.get_dummies(new_df, columns=["salesman_territory"], drop_first=True)
new_df = pd.get_dummies(new_df, columns=["unit_of_measurement"], drop_first=True)

new_df["invoice_day_of_week"] = new_df["invoice_date"].dt.dayofweek
new_df["invoice_month"] = new_df["invoice_date"].dt.month
new_df["invoice_day_of_month"] = new_df["invoice_date"].dt.day

new_df["delivery_order_day_of_week"] = new_df["delivery_order_date"].dt.dayofweek
new_df["delivery_order_month"] = new_df["delivery_order_date"].dt.month
new_df["delivery_order_day_of_month"] = new_df["delivery_order_date"].dt.day

new_df["delivery_day_of_week"] = new_df["delivery_date"].dt.dayofweek
new_df["delivery_month"] = new_df["delivery_date"].dt.month
new_df["delivery_day_of_month"] = new_df["delivery_date"].dt.day

new_df["due_day_of_week"] = new_df["due_date"].dt.dayofweek
new_df["due_month"] = new_df["due_date"].dt.month
new_df["due_day_of_month"] = new_df["due_date"].dt.day

new_df["days_until_due"] = (new_df["due_date"] - new_df["invoice_date"]).dt.days
new_df["days_until_due"] = new_df["days_until_due"].fillna(0)

new_df["days_until_delivery"] = (
    new_df["delivery_date"] - new_df["invoice_date"]
).dt.days
new_df["days_until_delivery"] = new_df["days_until_delivery"].fillna(0)

new_df["days_until_delivery_order"] = (
    new_df["delivery_order_date"] - new_df["invoice_date"]
).dt.days
new_df["days_until_delivery_order"] = new_df["days_until_delivery_order"].fillna(0)

new_df.drop(
    ["invoice_date", "delivery_order_date", "delivery_date", "due_date", "sub_amount"],
    axis=1,
    inplace=True,
)

X = new_df.drop("quantity", axis=1)
y = new_df["quantity"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

random_forest_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_reduced.fit(X_train, y_train)

feature_importances = pd.Series(
    random_forest_reduced.feature_importances_, index=X_train.columns
)
feature_importances = feature_importances.sort_values(ascending=False)
feature_importances.plot(kind="bar", figsize=(10, 6))
plt.title("Feature Importance")
plt.show()


X_train_reduced = X_train[
    [
        "item_number",
        "customer_number",
        "product_price",
        "base_quantity",
        "total_amount",
        "gst_amount",
        "due_day_of_week",
        "due_month",
        "due_day_of_month",
    ]
]

random_forest_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_reduced.fit(X_train_reduced, y_train)

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(random_forest_reduced, f)
