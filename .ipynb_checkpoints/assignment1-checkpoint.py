import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
# df = pd.read_csv("annual-enterprise-survey-2023-financial-year-provisional.csv")
df = pd.read_csv("Apple_price_to_clean.csv")

#ex1
# print(df.head())

# print(df.tail())

# print(df.info())

# print(df.isnull())

# print(df.dtypes)

#----------------------------------------------------------------------

# ex2

# print(df.isnull().sum())

# df_cleaned = df.dropna()
# print(df_cleaned)
# print(df_cleaned.isnull().sum())

# df_fill_mean = df.fillna(df.mean(numeric_only=True))
# print(df_fill_mean)
# df_fill_median = df.fillna(df.median(numeric_only=True))
# print(df_fill_median)

# df_filled_value = df.fillna(0)
# print(df_filled_value.isnull().sum())

# df_filled_ffill = df.fillna(method='ffill')
# print(df_filled_ffill)

# df_filled_bfill = df.fillna(method='bfill')
# print(df_filled_bfill)

#----------------------------------------------------------------------
# ex3

# data = pd.DataFrame({
#     'score': [100, 150, 200, 250, 300]
# })

# scaler = MinMaxScaler()
# data[['score']] = scaler.fit_transform(data[['score']])
# print(data)


# data = pd.DataFrame({
#     'Type': ['X', 'Y', 'Z', 'X'],
#     'Amount': [10, 5, 20, 15]
# })

# encoder = OneHotEncoder(sparse_output=False)
# encoded = encoder.fit_transform(data[['Type']])

# encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Type']))

# final_data = pd.concat([data.drop('Type', axis=1), encoded_df], axis=1)
# print(final_data)



# data = pd.DataFrame({
#     'Score': [8, 18, 28, 38, 48, 58, 68, 78, 88, 98]
# })

# intervals = [0, 25, 50, 75, 100]
# categories = ['Low', 'Medium', 'High', 'Very High']

# data['Score_Group'] = pd.cut(data['Score'], bins=intervals, labels=categories)
# print(data)

#----------------------------------------------------------------------

#ex4

# df = pd.DataFrame({
#     'Data1':[1, 2, 3, 4],
#     'Data2':[5, 6, 7, 8]
# })

# df['Result'] = df['Data1'] * df['Data2']
# print(df)


# df = pd.DataFrame({
#     'Feature1':[1, 2, 3],
#     'Feature2':[1, 2, 3]
# })

# poly = PolynomialFeatures(degree=2, include_bias = False)
# result = poly.fit_transform(df)

# print(result)



# df = pd.DataFrame({
#     'Date' : ['2023-01-01', '2023-05-15', '2024-07-20']
# })

# df['Date'] = pd.to_datetime(df['Date'])

# extracting year, month, day
# df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month
# df['Day'] = df['Date'].dt.day

# print(df)



# df = pd.DataFrame({
#     'Time' : ['05:30', '13:25', '19:00', '23:30']
# })

# df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

# def time_of_day(hour):
#     if 5 <= hour < 12:
#         return 'Morning'
#     elif 12 <= hour < 18:
#         return 'Afternoon'
#     elif 18 <= hour < 22:
#         return 'Evening'
#     else:
#         return 'Night'

# df['Time_hour'] = df['Time'].apply(lambda x:time_of_day(x.hour))
# print(df)

#----------------------------------------------------------------------

#ex5

# data = {
#    'Product': ['Desktop', 'Smartwatch', 'E-reader', 'Desktop', 'Smartwatch'],
#     'Price': [1200, 400, 350, 1200, 400],
#     'Store': ['Newegg', 'eBay', 'Target', 'Newegg', 'eBay']
# }
# df = pd.DataFrame(data)
# print(df)


# data = {
#     'Item': ['Monitor', 'Air Conditioner', 'Blender', 'Dishwasher', 'Toaster'],
#     'Price': [350, 750, 60, 700, 100]  
# }
# df = pd.DataFrame(data)
# print(df)
# Q1 = df['Price'].quantile(0.25)
# Q3 = df['Price'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# df_no_outliers = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
# print(df_no_outliers)


# data = {
#     'Employee': ['David', 'DAVID', 'Eve', 'EVE', 'Frank'],
#     'Department': ['Finance', 'finance', 'HR', 'hr', 'Support']
# }
# df = pd.DataFrame(data)
# df['Employee'] = df['Employee'].str.lower().str.capitalize()
# df['Department'] = df['Department'].str.capitalize()
# print(df)

#----------------------------------------------------------------------

#ex6
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# data = {
#     'Experience': [2, 7, 10, 3, 5],
#     'Hours_Worked': [30, 40, 50, 25, 35],
#     'Promoted': [0, 1, 1, 0, 1]  # 0 = Not Promoted, 1 = Promoted
# }

# df = pd.DataFrame(data)

# X = df.drop('Promoted', axis=1)  # 'Experience' and 'Hours_Worked'
# y = df['Promoted']  # Target: 'Promoted'

# # 70-30 and 80-20 splits
# X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=42)
# X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression()

# model.fit(X_train_70, y_train_70)
# predictions_70 = model.predict(X_test_30)
# accuracy_70 = accuracy_score(y_test_30, predictions_70)

# model.fit(X_train_80, y_train_80)
# predictions_80 = model.predict(X_test_20)
# accuracy_80 = accuracy_score(y_test_20, predictions_80)

# print(f"Accuracy with 70-30 split: {accuracy_70:.2f}")
# print(f"Accuracy with 80-20 split: {accuracy_80:.2f}")

#----------------------------------------------------------------------

#ex7

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# df = pd.DataFrame({
#     'Experience': [1, 5, None, 10, 3],
#     'Department': ['HR', 'IT', 'HR', 'Finance', 'IT'],
#     'Promoted': [0, 1, 1, 0, 1]
# })

# X = df.drop('Promoted', axis=1)
# y = df['Promoted']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# numerical_features = ['Experience']
# categorical_features = ['Department']

# numerical_pipeline = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')), 
#     ('scaler', StandardScaler())              
# ])

# categorical_pipeline = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')), 
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_pipeline, numerical_features),
#         ('cat', categorical_pipeline, categorical_features)
#     ]
# )

# full_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', RandomForestClassifier())
# ])

# full_pipeline.fit(X_train, y_train)

# predictions = full_pipeline.predict(X_test)
# report = classification_report(y_test, predictions)

# print(f"Classification Report:\n{report}")