import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import OneHotEncoder

# ------------------ Custom F1 Score ------------------
@register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Load saved objects
@st.cache_resource
def load_objects():
    with open(r"M:\DL\DL_FINAL\onehot_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open(r"M:\DL\DL_FINAL\minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = load_model(r"M:\DL\DL_FINAL\model.keras", custom_objects={'F1Score': F1Score()})
    return ohe, scaler, model

ohe, scaler, model = load_objects()

# Your categorical options (example, you should replace with your full lists)
product_options = [
    'apple honeycrisp organic', 'bag of organic bananas', 'banana', 'cucumber kirby',
    'honeycrisp apple', 'large lemon', 'limes', 'organic avocado', 'organic baby carrots',
    'organic baby spinach', 'organic blueberries', 'organic cucumber', 'organic fuji apple',
    'organic garlic', 'organic grape tomatoes', 'organic hass avocado', 'organic lemon',
    'organic raspberries', 'organic strawberries', 'organic whole milk', 'organic yellow onion',
    'organic zucchini', 'others', 'seedless red grapes', 'sparkling water grapefruit', 'strawberries', 'none'
]

department_options = [
    'alcohol', 'babies', 'bakery', 'beverages', 'breakfast', 'bulk', 'canned goods', 'dairy eggs',
    'deli', 'dry goods pasta', 'frozen', 'household', 'international', 'meat seafood', 'missing',
    'other', 'pantry', 'personal care', 'pets', 'produce', 'snacks', 'none'
]

aisle_options = [
    'baby food formula', 'bread', 'cereal', 'chips pretzels', 'crackers', 'eggs',
    'energy granola bars', 'fresh dips tapenades', 'fresh fruits', 'fresh herbs',
    'fresh vegetables', 'frozen meals', 'frozen produce', 'ice cream ice', 'juice nectars',
    'lunch meat', 'milk', 'none', 'packaged cheese', 'packaged vegetables fruits', 'refrigerated',
    'soft drinks', 'soup broth bouillon', 'soy lactosefree', 'water seltzer sparkling water', 'yogurt', 'none'
]

# Feature names (your full list, truncated here for brevity)
feature_names = [
    'final_product_name_apple honeycrisp organic', 'final_product_name_bag of organic bananas', 'final_product_name_banana',
    'final_product_name_cucumber kirby', 'final_product_name_honeycrisp apple', 'final_product_name_large lemon',
    'final_product_name_limes', 'final_product_name_organic avocado', 'final_product_name_organic baby carrots',
    'final_product_name_organic baby spinach', 'final_product_name_organic blueberries', 'final_product_name_organic cucumber',
    'final_product_name_organic fuji apple', 'final_product_name_organic garlic', 'final_product_name_organic grape tomatoes',
    'final_product_name_organic hass avocado', 'final_product_name_organic lemon', 'final_product_name_organic raspberries',
    'final_product_name_organic strawberries', 'final_product_name_organic whole milk', 'final_product_name_organic yellow onion',
    'final_product_name_organic zucchini', 'final_product_name_others', 'final_product_name_seedless red grapes',
    'final_product_name_sparkling water grapefruit', 'final_product_name_strawberries', 'final_department_name_alcohol',
    'final_department_name_babies', 'final_department_name_bakery', 'final_department_name_beverages',
    'final_department_name_breakfast', 'final_department_name_bulk', 'final_department_name_canned goods',
    'final_department_name_dairy eggs', 'final_department_name_deli', 'final_department_name_dry goods pasta',
    'final_department_name_frozen', 'final_department_name_household', 'final_department_name_international',
    'final_department_name_meat seafood', 'final_department_name_missing', 'final_department_name_other',
    'final_department_name_pantry', 'final_department_name_personal care', 'final_department_name_pets',
    'final_department_name_produce', 'final_department_name_snacks', 'final_aisle_name_baby food formula',
    'final_aisle_name_bread', 'final_aisle_name_cereal', 'final_aisle_name_chips pretzels', 'final_aisle_name_crackers',
    'final_aisle_name_eggs', 'final_aisle_name_energy granola bars', 'final_aisle_name_fresh dips tapenades',
    'final_aisle_name_fresh fruits', 'final_aisle_name_fresh herbs', 'final_aisle_name_fresh vegetables',
    'final_aisle_name_frozen meals', 'final_aisle_name_frozen produce', 'final_aisle_name_ice cream ice',
    'final_aisle_name_juice nectars', 'final_aisle_name_lunch meat', 'final_aisle_name_milk', 'final_aisle_name_none',
    'final_aisle_name_packaged cheese', 'final_aisle_name_packaged vegetables fruits', 'final_aisle_name_refrigerated',
    'final_aisle_name_soft drinks', 'final_aisle_name_soup broth bouillon', 'final_aisle_name_soy lactosefree',
    'final_aisle_name_water seltzer sparkling water', 'final_aisle_name_yogurt',
    'add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'product_reorder_ratio',
    'total_sales', 'weekday_sales', 'weekend_sales'
]

# Numerical columns names in order
numerical_cols = ['add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day',
                  'days_since_prior_order', 'product_reorder_ratio', 'total_sales',
                  'weekday_sales', 'weekend_sales']

st.title("Product Reorder Prediction")

# User inputs for categorical
product_name = st.selectbox("Select Product Name", product_options, index=product_options.index('none'))
department_name = st.selectbox("Select Department Name", department_options, index=department_options.index('none'))
aisle_name = st.selectbox("Select Aisle Name", aisle_options, index=aisle_options.index('none'))

# User inputs for numerical (set defaults to 0 or other appropriate values)
add_to_cart_order = st.number_input("Add To Cart Order", min_value=1, max_value=100, value=1)
order_dow = st.number_input("Order Day of Week (0=Sunday)", min_value=0, max_value=6, value=0)
order_hour_of_day = st.number_input("Order Hour of Day (0-23)", min_value=0, max_value=23, value=0)
days_since_prior_order = st.number_input("Days Since Prior Order", min_value=0, max_value=365, value=0)
product_reorder_ratio = st.number_input("Product Reorder Ratio", min_value=0.0, max_value=1.0, value=0.0, format="%.3f")
total_sales = st.number_input("Total Sales", min_value=0, value=0)
weekday_sales = st.number_input("Weekday Sales", min_value=0, value=0)
weekend_sales = st.number_input("Weekend Sales", min_value=0, value=0)


if st.button("Predict"):
    # Prepare categorical input
    input_cat_df = pd.DataFrame({
        'final_product_name': [product_name],
        'final_department_name': [department_name],
        'final_aisle_name': [aisle_name]
    })

    # One-hot encode categorical features
    encoded_cat = ohe.transform(input_cat_df)

    # Prepare numerical data
    numerical_data = pd.DataFrame({
        'add_to_cart_order': [add_to_cart_order],
        'order_dow': [order_dow],
        'order_hour_of_day': [order_hour_of_day],
        'days_since_prior_order': [days_since_prior_order],
        'product_reorder_ratio': [product_reorder_ratio],
        'total_sales': [total_sales],
        'weekday_sales': [weekday_sales],
        'weekend_sales': [weekend_sales]
    })

    # Match scaler expected input (ensures correct columns & order)
    numerical_data = numerical_data[scaler.feature_names_in_]
    # Scale numerical features
    scaled_numerical = scaler.transform(numerical_data)
    numerical_cols = scaler.feature_names_in_.tolist()

    # Combine encoded categorical and scaled numerical features
    final_input = np.hstack((encoded_cat, scaled_numerical))

    # Predict probability and class
    prediction_proba = model.predict(final_input)
  
    
    # Convert prediction_proba to a regular Python list (if it's a NumPy array)
    probs = prediction_proba[0].tolist()

    # Get the index of the maximum value (i.e., predicted class: 0 or 1)
    predicted_class = probs.index(max(probs))

    # Get the probability of class 1 (reorder = Yes)
    probability_of_reorder = probs[1]

    probability_of_reorder = prediction_proba[0][1]

    st.write(f"Prediction probability of reorder: {probability_of_reorder:.4f}")
    st.write(f"Predicted reorder: {'Yes' if predicted_class ==1 else 'No'}")

