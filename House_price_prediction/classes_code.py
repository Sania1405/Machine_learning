import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

class HousePricePredictor:
    def __init__(self, train_file):
        """
        This is the constructor. It sets up the model's components.
        """
        self.train_file = train_file

        self.nominal_cols = ['MSZoning', 'Alley', 'Neighborhood', 'HouseStyle', 'MasVnrType', 'Heating', 'Electrical', 'CentralAir', 'GarageType', 'Fence']
        self.ordinal_cols = ['ExterQual', 'KitchenQual', 'HeatingQC']
        self.columns_to_scale = ['GarageArea', 'PoolArea', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'TotalBath', 'Quality_and_Size', 'House_Age']

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), self.nominal_cols),
                ('ordinal', OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex'], ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ['Po', 'Fa', 'TA', 'Gd', 'Ex']]), self.ordinal_cols),
                ('scaler', StandardScaler(), self.columns_to_scale)
            ],
            remainder='passthrough'
        )
        self.model = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.upper_limits = {}
        self.lower_limits = {}
        self.ordinal_quality = ['Po', 'Fa', 'TA', 'Gd', 'Ex']


    @st.cache_data
    def load_and_train_model(_self):
        """
        This new function handles loading, processing, and training the model.
        """
        train_df = pd.read_csv(_self.train_file, usecols=[
            "MSZoning", "LotArea", "Alley", "Neighborhood", "HouseStyle", "OverallQual",
            "MasVnrType", "ExterQual", "Heating", "HeatingQC", "CentralAir", "Electrical",
            "BsmtFullBath", "FullBath", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd",
            "Fireplaces", "GarageType", "GarageArea", "PoolArea", "Fence", "YrSold", "YearRemodAdd",
            "SalePrice"
        ])
        
        # Handle missing values
        cols_to_fill = ['Alley', 'MasVnrType', 'Fence', 'GarageType']
        for col in cols_to_fill:
            train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
            
        train_df.dropna(subset=['Electrical', 'BsmtFullBath', 'KitchenQual', 'GarageArea'], inplace=True)
        
        # Feature Engineering
        train_df['TotalBath'] = train_df['BsmtFullBath'] + train_df['FullBath']
        train_df['Quality_and_Size'] = train_df['OverallQual'] * train_df['LotArea']
        train_df['House_Age'] = train_df['YrSold'] - train_df['YearRemodAdd']

        # Outlier handling using IQR method
        for col in ['Quality_and_Size', 'GarageArea', 'SalePrice', 'LotArea', 'House_Age']:
            percentile25 = train_df[col].quantile(0.25)
            percentile75 = train_df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 1.5 * iqr
            lower_limit = percentile25 - 1.5 * iqr
            
            _self.upper_limits[col] = upper_limit
            _self.lower_limits[col] = lower_limit
            
            train_df[col] = np.where(train_df[col] > upper_limit, upper_limit, train_df[col])
            train_df[col] = np.where(train_df[col] < lower_limit, lower_limit, train_df[col])

        # Drop original columns used for feature engineering
        _self.X_train = train_df.drop(['BsmtFullBath','FullBath','OverallQual','LotArea','YrSold','YearRemodAdd', 'SalePrice'], axis=1)
        _self.y_train = train_df['SalePrice']

        # Fit the ColumnTransformer and the LinearRegression model
        X_train_transformed = _self.column_transformer.fit_transform(_self.X_train)
        _self.model.fit(X_train_transformed, _self.y_train)


    def predict(self, user_input_df):
        """
        This method now handles feature engineering on the new input,
        caps outliers, and makes a prediction.
        """
        # Feature Engineering for user input
        user_input_df['TotalBath'] = user_input_df['BsmtFullBath'] + user_input_df['FullBath']
        user_input_df['Quality_and_Size'] = user_input_df['OverallQual'] * user_input_df['LotArea']
        user_input_df['House_Age'] = user_input_df['YrSold'] - user_input_df['YearRemodAdd']

        # Cap outliers in user input using the limits from the training data
        for col in ['Quality_and_Size', 'GarageArea', 'LotArea', 'House_Age']:
            if col in self.upper_limits:
                user_input_df[col] = np.where(user_input_df[col] > self.upper_limits[col], self.upper_limits[col], user_input_df[col])
            if col in self.lower_limits:
                 user_input_df[col] = np.where(user_input_df[col] < self.lower_limits[col], self.lower_limits[col], user_input_df[col])

        # Drop original columns used for feature engineering
        user_input_df = user_input_df.drop(['BsmtFullBath','FullBath','OverallQual','LotArea','YrSold','YearRemodAdd'], axis=1)
        
        # Reorder columns to match training data
        user_input_df = user_input_df[self.X_train.columns]
        
        # Transform the user input using the fitted ColumnTransformer
        user_input_transformed = self.column_transformer.transform(user_input_df)
        
        # Make the prediction
        return self.model.predict(user_input_transformed)[0]


# --- Streamlit Web Application ---

# Create a cached function to build and train the entire predictor object
@st.cache_resource
def get_predictor_instance():
    _predictor = HousePricePredictor(train_file="train.csv")
    _predictor.load_and_train_model()
    return _predictor

# Get the fully prepared predictor object
predictor = get_predictor_instance()

# The rest of the Streamlit app code
st.title("House Price Predictor")
st.write("Enter the details of the house to get a price prediction.")

with st.form("prediction_form"):
    # Streamlit input widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        ms_zoning = st.selectbox("MSZoning", options=predictor.X_train['MSZoning'].unique(), help="General zoning classification")
        lot_area = st.number_input("LotArea", value=10000, min_value=0, max_value=200000)
        alley = st.selectbox("Alley", options=predictor.X_train['Alley'].unique(), help="Type of alley access")
        neighborhood = st.selectbox("Neighborhood", options=predictor.X_train['Neighborhood'].unique(), help="Physical location within Ames city limits")
        house_style = st.selectbox("HouseStyle", options=predictor.X_train['HouseStyle'].unique(), help="Style of the dwelling")
        overall_qual = st.selectbox("OverallQual", options=range(1, 11), help="Rates the overall material and finish of the house (1-10)")
        mas_vnr_type = st.selectbox("MasVnrType", options=predictor.X_train['MasVnrType'].unique(), help="Masonry veneer type")
        year_remod_add = st.number_input("YearRemodAdd", value=2003, min_value=1950, max_value=2010, help="Remodel date (same as construction date if no remodel or addition)")
    
    with col2:
        exter_qual = st.selectbox("ExterQual", options=predictor.ordinal_quality, help="Evaluates the quality of the material on the exterior (Ex, Gd, Ta, Fa, Po)")
        heating = st.selectbox("Heating", options=predictor.X_train['Heating'].unique(), help="Type of heating")
        heating_qc = st.selectbox("HeatingQC", options=predictor.ordinal_quality, help="Heating quality and condition (Ex, Gd, Ta, Fa, Po)")
        central_air = st.selectbox("CentralAir", options=predictor.X_train['CentralAir'].unique(), help="Central air conditioning (Y/N)")
        electrical = st.selectbox("Electrical", options=predictor.X_train['Electrical'].unique(), help="Electrical system")
        bsmt_full_bath = st.number_input("BsmtFullBath", value=1, min_value=0, max_value=3, help="Basement full bathrooms")
        full_bath = st.number_input("FullBath", value=2, min_value=0, max_value=3, help="Above ground full bathrooms")
        yr_sold = st.number_input("YrSold", value=2008, min_value=2006, max_value=2010, help="Year sold (YYYY)")

    with col3:
        kitchen_abv_gr = st.number_input("KitchenAbvGr", value=1, min_value=0, max_value=3, help="Kitchens above grade")
        kitchen_qual = st.selectbox("KitchenQual", options=predictor.ordinal_quality, help="Kitchen quality (Ex, Gd, Ta, Fa, Po)")
        tot_rms_abv_grd = st.number_input("TotRmsAbvGrd", value=7, min_value=2, max_value=14, help="Total rooms above grade (not including bathrooms)")
        fireplaces = st.number_input("Fireplaces", value=1, min_value=0, max_value=3, help="Number of fireplaces")
        garage_type = st.selectbox("GarageType", options=predictor.X_train['GarageType'].unique(), help="Garage location")
        garage_area = st.number_input("GarageArea", value=450, min_value=0, max_value=1500, help="Size of garage in square feet")
        pool_area = st.number_input("PoolArea", value=0, min_value=0, max_value=800, help="Pool area in square feet")
        fence = st.selectbox("Fence", options=predictor.X_train['Fence'].unique(), help="Fence quality")

    submit_button = st.form_submit_button("Predict Price")


if submit_button:
    user_input = pd.DataFrame([[
        ms_zoning, lot_area, alley, neighborhood, house_style, overall_qual,
        mas_vnr_type, exter_qual, heating, heating_qc, central_air, electrical,
        bsmt_full_bath, full_bath, kitchen_abv_gr, kitchen_qual, tot_rms_abv_grd,
        fireplaces, garage_type, garage_area, pool_area, fence, yr_sold, year_remod_add
    ]], columns=['MSZoning', 'LotArea', 'Alley', 'Neighborhood', 'HouseStyle', 'OverallQual', 'MasVnrType', 'ExterQual', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'BsmtFullBath', 'FullBath', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType', 'GarageArea', 'PoolArea', 'Fence', 'YrSold', 'YearRemodAdd'])
    
    final_prediction = predictor.predict(user_input)
    
    st.success(f"The predicted house price is: ${final_prediction:,.2f}")