import streamlit as st
import os, json
import pandas as pd

from src.pipeline.predict_pipeline import predict

# Loading important stuff
features_file_path = os.path.join("artifacts", "Features.json")
with open(features_file_path, "r") as file:
    features = json.load(file)
options_file_path = os.path.join("artifacts", "cat_var_options.json")
with open(options_file_path, "r") as file:
    options = json.load(file)

# App framework
st.title("Literature score preditor!")
st.divider()
st.header("Enter custom data OR upload csv files:")
st.divider()

# Custom data
custom_input = {}
for feat in features.keys():
    if features[feat] == "numerical_feature":
        st.write(f"Enter the value of {feat}:")
        custom_input[feat] = [st.number_input("Values range from 0 to 100", key = feat)]
    elif features[feat] == "categorical_feature":
         st.write(f"Select the options for {feat}:")
         custom_input[feat] = [st.selectbox(label = "Options", options = options[feat], key = feat)]
custom_input = pd.DataFrame(custom_input)

# If custom data is given, then predict:
if st.button("Predict"):
    preds = predict(custom_input)
    st.header("Predicted writing score")
    st.dataframe(preds)

# features_dict = {}
# for feat in features.keys():
#     if features[feat] == "numerical_feature":
#         features_dict[feat] = [50]
#     elif features[feat] == "categorical_feature":
#         features_dict[feat] = [options[feat][0]]

# features_df = pd.DataFrame(features_dict)

# columns_config = {}
# for feat in features.keys():
#     if features[feat] == "numerical_feature":
#         columns_config[feat] = st.column_config.NumberColumn()
#     elif features[feat] == "categorical_feature":
#         columns_config[feat] = st.column_config.SelectboxColumn(options = options[feat])

# st.data_editor(
#     features_df,
#     column_config = columns_config,
#     num_rows = "dynamic",
#     hide_index = True,
#     on_change = st_predictor, 
#     key = "user_input"
# )

# If custom data is given, then predict:
# features_arr = get_array_representation(features_df)
# if st.button("Predict"):
#     preds = predict(features_df)
#     st.header("Predictions")
#     st.dataframe(preds)

# Upload csv file
st.divider()
st.write("Note: The CSV file's column names should same as below")
inputs = st.file_uploader("Upload csv file:")

# If uploaded:
if inputs:
    # features_arr = get_array_representation(inputs)
    inputs = pd.read_csv(inputs)
    preds = predict(inputs)
    st.header("Predictions for writing score")
    st.dataframe(preds)
