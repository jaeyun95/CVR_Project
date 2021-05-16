from config import configure
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np

origin_columns = ["Sale","SalesAmountInEuro","Time_delay_for_conversion","click_timestamp","nb_clicks_1week","product_price",
           "product_age_group","device_type","audience_id","product_gender","product_brand","product_category1","product_category2",
           "product_category3","product_category4","product_category5","product_category6","product_category7","product_country",
           "product_id","product_title","partner_id","user_id"]
string_data = ["product_age_group","device_type","audience_id","product_gender","product_brand","product_category1","product_category2",
           "product_category3","product_category4","product_category5","product_category6","product_category7","product_country",
           "product_id","product_title","partner_id","user_id"]
one_hot_list = ["product_age_group","device_type","product_gender", "product_brand","product_category1","product_category2",
           "product_category3","product_category4","product_category5","product_category6","product_category7","product_country",]

def preprocessing(data_range):
    data_frame = pd.read_csv(configure['DATA_PATH']+configure['DATA_FILE'], delimiter='\t', names=origin_columns, low_memory=False,
                             nrows=2000000)[data_range[0]:data_range[1]]
    # drop other label
    data_frame = data_frame.drop("SalesAmountInEuro", 1)
    data_frame = data_frame.drop("Time_delay_for_conversion", 1)

    # click timestamp convert to day, hour, minutes
    data_frame["click_day"] = data_frame["click_timestamp"].map(lambda x: int(x / 10000))
    data_frame["click_hour"] = data_frame["click_timestamp"].map(lambda x: int(x / 100 % 100))
    data_frame["click_minute"] = data_frame["click_timestamp"].map(lambda x: int(x % 100))
    data_frame = data_frame.drop("click_timestamp", 1)

    # fill missing value
    data_frame['user_id'] = data_frame['user_id'].replace(-1, np.nan)
    data_frame['partner_id'] = data_frame['partner_id'].replace(-1, np.nan)
    data_frame['product_title'] = data_frame['product_title'].replace(-1, np.nan)
    data_frame['product_id'] = data_frame['product_id'].replace(-1, np.nan)
    data_frame['product_gender'] = data_frame['product_gender'].replace(-1, np.nan)
    data_frame = data_frame.interpolate(method='values')

    # make label enocder
    for ohl in string_data:
        label_encoder = LabelEncoder()
        data_frame[ohl] = label_encoder.fit_transform(data_frame[ohl])

    # seperate label frame
    label_frame = data_frame["Sale"]
    data_frame = data_frame.drop("Sale", 1)

    # scaling
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    data_frame[data_frame.columns] = minmax_scaler.fit_transform(data_frame[data_frame.columns])

    # data augmentation check
    data = data_frame.values
    label = label_frame.values
    
    return data, label
