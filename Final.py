#1."MERGING OF DATASETS"

pip install pandas

import pandas as pd
import numpy as py

aisles_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\New folder\aisles.csv")
aisles_df

products_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\New folder\products.csv")
products_df

aisles_product_df = pd.merge(aisles_df, products_df, on='aisle_id')
aisles_product_df

departments_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\New folder\departments.csv")
departments_df

aisles_product_departments_df1 = pd.merge(aisles_product_df, departments_df, on='department_id')
aisles_product_departments_df1

order_products_prior_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\New folder\order_products__prior.csv")
order_products_prior_df

order_products_train_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\New folder\order_products__train.csv")
order_products_train_df

order_products_train_prior_df = pd.concat([order_products_prior_df, order_products_train_df])
print(order_products_train_prior_df.shape)
order_products_train_prior_df.head(5)

orders_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\New folder\orders.csv")
orders_df

order_products_train_prior_orders_df2 = pd.merge(order_products_train_prior_df, orders_df, on='order_id')
order_products_train_prior_orders_df2

df = pd.merge(aisles_product_departments_df1, order_products_train_prior_orders_df2)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
df.sample(5)

df.to_csv("after_merged.csv") # converting df to csv

#2."DATA CLEANING 1" [Checking Nan Values]

df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\after_merged.csv")
df

df.columns

df['Unnamed: 0']

df.drop(columns = ["Unnamed: 0"], axis=1, inplace = True) # dropping  column unnamed

df

df["aisle_id"].unique()

df["aisle"].unique()

df["product_id"].unique()

df['product_name'].unique()

df['department_id'].unique()

df['department'].unique()

df['order_id'].unique()

df['add_to_cart_order'].unique()

df['reordered'].unique()

df['user_id'].unique()

df['eval_set'].unique()

df['order_number'].unique()

df['order_dow'].unique()

df['order_hour_of_day'].unique()

df['days_since_prior_order'].unique()

df.shape #count of rows and columns

df.isnull().values.any()  #checking null values

df.isnull().sum().sum()

#checking null and its sum
df.isnull().sum()

df.dtypes  #Checking datatypes

df[df.duplicated()].sum()

df[df.isnull().any(axis=1)]

df1 = df.copy()

df1['days_since_prior_order'].isna().sum()

df1['days_since_prior_order'].value_counts(dropna=False)

numerical_cols = df1.select_dtypes(['int64', 'float64']).columns
numerical_cols

for i in numerical_cols:
    corr = df1['days_since_prior_order'].corr(df1[i])
    print(f'corr{i} :', corr)

"""MULTIPLE IMPUTATION BY CHAINED EQUATIONS TO FILL MISSING VALUES IN VALUE CONFIGURATION"""

#need to enable iterative imputer explicity since its still experimental
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Define Imputer
imputer = IterativeImputer(random_state = 100, max_iter=10)

df1_train = df1.loc[:,['reordered','order_dow', 'order_hour_of_day' ,'days_since_prior_order']]
df1_train.sample(3)

df1_train['days_since_prior_order'].value_counts(dropna=False)

#fit on the dataset
imputer.fit(df1_train)

df_imputed = imputer.transform(df1_train)
df_imputed[:10]

df.loc[:, ['reordered','order_dow', 'order_hour_of_day' ,'days_since_prior_order']] = df_imputed

df['days_since_prior_order'].value_counts(dropna=False)

df

df.isnull().sum().sum()   #Re-checking null values after Imputation

df.isnull().sum()

df.to_csv('No_NaN_values.csv', index = False)

#3. "DATA CLEANING 2"  [Checking outliers]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\No_NaN_values.csv")
df

df.isnull().sum()



#Finding the minimum and Maximum value for the column "reordered"
min_value = df['reordered'].min()
print(f"Minimum value in '{'reordered'}':", min_value)
max_value = df['reordered'].max()
print(f"Maximum value in '{'reordered'}':", max_value)

#Finding the minimum and Maximum value for the column "add_to_cart_order"
min_value = df['add_to_cart_order'].min()
print(f"Minimum value in '{'add_to_cart_order'}':", min_value)
max_value = df['add_to_cart_order'].max()
print(f"Maximum value in '{'add_to_cart_order'}':", max_value)

df['add_to_cart_order'].describe()

#Finding the minimum and Maximum value for the column "order_hour_of_day"
min_value = df['order_hour_of_day'].min()
print(f"Minimum value in '{'order_hour_of_day'}':", min_value)
max_value = df['order_hour_of_day'].max()
print(f"Maximum value in '{'order_hour_of_day'}':", max_value)

#Finding the minimum and Maximum value for the column "days_since_prior_order"
min_value = df['days_since_prior_order'].min()
print(f"Minimum value in '{'days_since_prior_order'}':", min_value)
max_value = df['days_since_prior_order'].max()
print(f"Maximum value in '{'days_since_prior_order'}':", max_value)

#Finding the minimum and Maximum value for the column "order_dow"
min_value = df['order_dow'].min()
print(f"Minimum value in '{'order_dow'}':", min_value)
max_value = df['order_dow'].max()
print(f"Maximum value in '{'order_dow'}':", max_value)

df['add_to_cart_order_log'] = np.log1p(df['add_to_cart_order'])
plt.figure(figsize = (8,4))
sns.boxplot(x=df['add_to_cart_order_log'])
plt.title('Outliers in Log transformed add_to_cart_order')
plt.show()

"""IQR METHOD"""

Q1 = df['add_to_cart_order'].quantile(0.25)
Q3 = df['add_to_cart_order'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = 120

outliers = df[(df['add_to_cart_order'] < lower_bound) | (df['add_to_cart_order'] > upper_bound)]
outliers

upper_bound

"""Z SCORE METHOD"""

from scipy import stats

df['add_to_cart_order'] = np.abs(stats.zscore(df['add_to_cart_order']))
outliers = df[df['add_to_cart_order'] > 2]
outliers

from scipy.stats import skew
skewness = skew(df['add_to_cart_order'])
skewness

#Histogram with KDE
plt.figure(figsize = (8,4))
sns.histplot(df["add_to_cart_order"], bins=20, kde=True)
plt.title("Histogram & KDE plot for skewness")
plt.xlabel("add_to_cart_order")
plt.ylabel("Frequency")
plt.show()

"""For my learning process, i have tried all these pre_processing steps for outliers and skewness, even though there are no outliers and skewness in the dataset."""

df.to_csv('No_outliers.csv', index = False)

#4. "EDA" [Collinearity checking]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\No_outliers.csv")
pd.set_option('display.max_columns',None)
df.head(2)

df['department'].unique()

df.shape

numerical_cols = df.select_dtypes(include=[np.number]).columns
numerical_cols

"""FEATURE VS FEATURE

NUMERICAL VS CATEGORICAL
"""

df['department_id'].value_counts() #Categorical

df['add_to_cart_order'].value_counts()  #Numerical

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the plot aesthetics are clean
plt.figure(figsize=(12, 6))

# Scatterplot using seaborn
sns.scatterplot(
    x='department_id',
    y='add_to_cart_order',
    data=df,
    hue='department',
    style='department',
    palette='Set3'
)

# Set axis labels and title
plt.xlabel('Department ID')
plt.ylabel('Add to Cart Order')
plt.title('Department ID vs Add to Cart Order')

# Show legend and plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # optional, to keep legend outside
plt.tight_layout()
plt.show()

"""CATEGORICAL VS  TARGET (CATEGORICAL)"""

#'department_id vs reordered'
import seaborn as sns
sns.barplot(
    x='department_id',
    y='reordered',
    data=df,
    hue='department',       # this adds a different color for each department
    palette='Set2'               # Set2 palette for pretty colors
)

plt.xlabel('department_id')
plt.ylabel('reordered')
plt.title('department_id vs reordered')
plt.legend(title='department')  # adding a legend with a title
plt.show()

# Grouping of the product_id and product_name and finding the reorder ratio for each product,
# As there  are numerous categories in products i have done grouping to find the ratio and for a clear graphical representation
reorder_ratio_df = (
    df.groupby(['product_id', 'product_name'])['reordered']
    .mean()
    .reset_index()
    .rename(columns={'reordered': 'reorder_ratio'})
)
print(reorder_ratio_df)

top_10_reordered = reorder_ratio_df.sort_values(by='reorder_ratio', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(
    x='reorder_ratio',
    y='product_name',
    data=top_10_reordered,
    palette='Set2'
)

plt.xlabel('Reorder Ratio')
plt.ylabel('Product')
plt.title('Top 10 Reordered Products')



plt.tight_layout()
plt.show()

# finding the reorder ratio for each and every product name and plotting.
grouped_df_1 = df.groupby('product_name')['reordered'].value_counts(normalize = True).unstack()
grouped_df_1.head(10).plot(kind = 'bar')

#Filtering top 10 products by grouping  product name and reordered columns
grouped_df_2 = df.groupby('product_name')['reordered'].sum().reset_index()
grouped_df_2 = grouped_df_2.sort_values(by='reordered', ascending=False).head(10)
grouped_df_2

new_df = df[df['product_name'].isin(grouped_df_2['product_name'])]
new_df

plot_data = new_df.groupby('product_name')['reordered'].sum().reset_index()

# Sort by reordered count
plot_data = plot_data.sort_values(by='reordered', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_data, x='product_name', y='reordered', palette='Blues_r')

plt.title('Top 10 Most Reordered Products')
plt.xlabel('Product Name')
plt.ylabel('Total Reordered Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""COLLINEARITY CHECKING"""

df.columns

numerical_cols = df.select_dtypes(include=[np.number]).columns
numerical_cols

actual_numerical_cols = [ 'add_to_cart_order', 'order_id','user_id', 'order_number', 'days_since_prior_order']

actual_categorical_cols = ['aisle_id', 'aisle', 'product_id', 'product_name',  'department_id', 'department', 'reordered', 'eval_set', 'order_dow',
       'order_hour_of_day']

pos_cols_1 = []
neg_cols_1 = []
for col in actual_numerical_cols:
    corr = df[col].corr(df['add_to_cart_order'])
    print(f'Corr {col} :', corr)
    positive_corr = df[col].corr(df['add_to_cart_order']) > 0.2
    negative_corr = df[col].corr(df['add_to_cart_order']) < -0.2
    if positive_corr:
        pos_cols_1.append(col)
    elif negative_corr:
        neg_cols_1.append(col)

pos_cols_1

neg_cols_1

pos_cols_2= []
neg_cols_2 = []
for col in actual_numerical_cols:
    corr = df[col].corr(df['order_id'])
    print(f'Corr {col} :', corr)
    positive_corr = df[col].corr(df['order_id']) > 0.2
    negative_corr = df[col].corr(df['order_id']) < -0.2
    if positive_corr:
        pos_cols_2.append(col)
    elif negative_corr:
        neg_cols_2.append(col)

pos_cols_2

neg_cols_2

pos_cols_3= []
neg_cols_3 = []
for col in actual_numerical_cols:
    corr = df[col].corr(df['user_id'])
    print(f'Corr {col} :', corr)
    positive_corr = df[col].corr(df['user_id']) > 0.2
    negative_corr = df[col].corr(df['user_id']) < -0.2
    if positive_corr:
        pos_cols_3.append(col)
    elif negative_corr:
        neg_cols_3.append(col)

pos_cols_3

neg_cols_3

pos_cols_4= []
neg_cols_4 = []
for col in actual_numerical_cols:
    corr = df[col].corr(df['order_number'])
    print(f'Corr {col} :', corr)
    positive_corr = df[col].corr(df['order_number']) > 0.2
    negative_corr = df[col].corr(df['order_number']) < -0.2
    if positive_corr:
        pos_cols_4.append(col)
    elif negative_corr:
        neg_cols_4.append(col)

pos_cols_4

neg_cols_4

pos_cols_5 = []
neg_cols_5 = []
for col in actual_numerical_cols:
    corr = df[col].corr(df['days_since_prior_order'])
    print(f'Corr {col} :', corr)
    positive_corr = df[col].corr(df['days_since_prior_order']) > 0.2
    negative_corr = df[col].corr(df['days_since_prior_order']) < -0.2
    if positive_corr:
        pos_cols_5.append(col)
    elif negative_corr:
        neg_cols_5.append(col)

pos_cols_5

neg_cols_5

#Collinearity Checking
plt.figure(figsize=(10,10))
sns.heatmap(df[actual_numerical_cols].corr(),annot=True)
plt.show()

df.to_csv('after_eda.csv', index = False)


#5. "FEATURE ENGINEERING"[selection of Top 25 Products and selection of  departments & asiles associated with the top 25 products]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

new_df = pd.read_csv(r"F:\GUVI PROJECTS\DL_FINAL\after_eda.csv")
pd.set_option('display.max_columns',None)
new_df.head(2)

#df_backup = df.copy()

a = new_df['product_name'].value_counts().head(25).index.to_list()
a

new_df['product_name']

top_25_products = new_df['product_name'].value_counts().head(25).index.tolist()
top_25_products

new_df['product_name'] == "Organic Cucumber"

"Organic Cucumber" in new_df["product_name"].values

a= [str(x).strip().lower() for x in a]  #Normalising

new_df['cleaned_product_name'] = new_df['product_name'].astype(str).str.strip().str.lower()
new_df['cleaned_product_name']

top_25_products = new_df['cleaned_product_name'].value_counts().head(25).index.tolist()
top_25_products

new_df['final_product_name']=new_df['cleaned_product_name'].apply(lambda x : x if x in top_25_products else 'others')
new_df['final_product_name'].value_counts

new_df['final_product_name'].value_counts()

"""DEPARTMENTS"""

# 1️⃣ Find departments associated with top 25 products
top25_departments = new_df.loc[
    new_df['final_product_name'] != 'None', 'department'
].unique()
print("Departments associated with top 25 products:")
print(top25_departments)

a = new_df['department'].value_counts().head(25).index.to_list()
a

new_df['department']

top_25_departments = new_df['department'].value_counts().head(25).index.tolist()
top_25_departments

new_df['department'] == "international"

"international" in new_df["department"].values

a= [str(x).strip().lower() for x in a]  #Normalising
a

new_df['cleaned_department_name'] = new_df['department'].astype(str).str.strip().str.lower()
new_df['cleaned_department_name']

top_25_departments = new_df['cleaned_department_name'].value_counts().head(25).index.tolist()
top_25_departments

new_df['cleaned_department_name'].value_counts()

new_df['final_department_name']=new_df['cleaned_department_name'].apply(lambda x : x if x in top_25_departments else 'none')

new_df['final_department_name'].value_counts()

"""AISLES"""

# 1️⃣ Find aisles associated with top 25 products
top25_aisles = new_df.loc[
    new_df['final_product_name'] != 'None', 'aisle'
].unique()
print("aisles associated with products:")
print(top25_aisles)

a = new_df['aisle'].value_counts().head(25).index.to_list()
a

new_df['aisle']

top_25_aisles = new_df['aisle'].value_counts().head(25).index.tolist()
top_25_aisles

new_df['aisle'] == "fresh dips tapenades"

"cereal" in new_df["aisle"].values

a= [str(x).strip().lower() for x in a]  #Normalising

new_df['cleaned_aisle_name'] = new_df['aisle'].astype(str).str.strip().str.lower()
new_df['cleaned_aisle_name']

top_25_aisles = new_df['cleaned_aisle_name'].value_counts().head(25).index.tolist()
top_25_aisles

new_df['cleaned_aisle_name'].value_counts()

new_df['final_aisle_name']=new_df['cleaned_aisle_name'].apply(lambda x : x if x in top_25_aisles else 'none')

new_df['final_aisle_name'].value_counts()

"""AISLE ID"""

# 1️⃣ Find aisles associated with top 25 Aisles
top25_aisle_id = new_df.loc[
    new_df['aisle_id'] != '0000', 'aisle_id'
].unique()
print("aisles associated with aisle_id")
print(top25_aisle_id)

a = new_df['aisle_id'].value_counts().head(25).index.to_list()
a

new_df['aisle_id']

top25_aisle_id = new_df['aisle_id'].value_counts().head(25).index.tolist()
top25_aisle_id

new_df['aisle_id'] == "7"

new_df['final_aisle_id']=new_df['aisle_id'].apply(lambda x : x if x in top25_aisle_id else '0000')

new_df['final_aisle_id'].value_counts()

"""PRODUCT ID"""

# 1️⃣ Find product_ids associated with top 25 products
top25_product_id = new_df.loc[
    new_df['product_id'] != '0000', 'product_id'
].unique()
print("products associated with product_id")
print(top25_product_id)

a = new_df['product_id'].value_counts().head(25).index.to_list()
a

new_df['product_id']

top25_product_id = new_df['product_id'].value_counts().head(25).index.tolist()
top25_product_id

new_df['product_id'] == "30391"

new_df['final_product_id']=new_df['product_id'].apply(lambda x : x if x in top25_product_id else '0000')

new_df['final_product_id'].value_counts()

"""DEPARTMENT ID"""

# 1️⃣ Find product_ids associated with top 25 products
top25_department_id = new_df.loc[
    new_df['department_id'] != '0000', 'department_id'
].unique()
print("departments associated with department_id")
print(top25_department_id)

a = new_df['department_id'].value_counts().head(25).index.to_list()
a

new_df['department_id']

top25_department_id = new_df['department_id'].value_counts().head(25).index.tolist()
top25_department_id

new_df['department_id'] == "11"

new_df['final_department_id']=new_df['department_id'].apply(lambda x : x if x in top25_department_id else '0000')

new_df['final_department_id'].value_counts()

new_df

columns_to_drop = [
    'product_id', 'department_id', 'aisle_id',
    'cleaned_department_name', 'cleaned_product_name', 'cleaned_aisle_name',
    'department', 'product_name', 'aisle']

new_df.drop(columns=columns_to_drop, inplace=True)

new_df

"""FEATURE ENGINEERING

USER-BASED FEATURES
"""

# 1. Calculating the Average days between orders for user_id 161125
# Specify user ID
user_id = 161125

# Filter for user
user_data = new_df[ new_df['user_id'] == user_id]

# Calculate average days
average_days = user_data['days_since_prior_order'].mean()

print(f"Average days between orders for user {user_id}: {average_days}")

#Attaching a new column 'avg_days_btn_orders_for_user_161125' to the dataframe
# Specify user ID
user_id = 161125

# Check if user_id exists in the dataframe
if user_id in new_df['user_id'].values:
    # Filter for user
    user_data = new_df[new_df['user_id'] == user_id]

    # Calculate average days
    average_days = user_data['days_since_prior_order'].mean()

    # Create a new column with average days for that user only
    new_df['avg_days_btn_orders_for_user_161125'] = new_df['user_id'].apply(
        lambda x: average_days if x == user_id else 0)

    print(f"Average days between orders for user {user_id}: {average_days}")
else:
    print(f"User {user_id} not found in the DataFrame.")

# 2. Find the user id which has purchased organic fuji apple
# Filter the DataFrame for the specific product
filtered_df = new_df[new_df['final_product_name'] == 'organic fuji apple']

# Get unique user IDs
unique_users = filtered_df['user_id'].unique()

# Create a DataFrame with these user IDs who purchased organic fuji apple
users_with_organic_fuji_apple = pd.DataFrame({'user_id': unique_users})
users_with_organic_fuji_apple.head(10)

# 3. Total orders of the specific product 'organic fuji apple' purchased by a specific user id '145501'
# Specify user ID and product
user_id =   145501
product = 'organic fuji apple'

# Filter the data
user_product_data = new_df[(new_df['user_id'] == user_id) & (new_df['final_product_name'] == product)]

# Calculate total orders
total_orders = user_product_data.shape[0]

print(f"Total orders of '{product}' by user {user_id}: {total_orders}")

#Attaching a new column 'organic fuji apple by 145501' to the dataframe
import numpy as np

# Step 1: Define user and product
user_id = 145501
product = 'organic fuji apple'

# Step 2: Filter and count total orders of the product by the user
user_product_data = new_df[(new_df['user_id'] == user_id) & (new_df['final_product_name'] == product)]
total_orders = user_product_data.shape[0]

# Step 3: Attach the total_orders as a column (same value repeated only for matching rows)
new_df['organic_fuji_apple_by_145501'] = np.where(
    (new_df['user_id'] == user_id) & (new_df['final_product_name'] == product),
    total_orders,
    0
)

# Optional: View result
new_df[['user_id', 'final_product_name', 'organic_fuji_apple_by_145501']].tail(10)

new_df['organic_fuji_apple_by_145501'].value_counts()

new_df['user_id']

# 4. Find the total number of orders for each product by a specific user id
# Specify user ID
user_id = 173861
# Filter data for this user
user_orders_data = new_df[new_df['user_id'] == user_id]

# Count number of orders for each product by this user
user_product_counts = (
    user_orders_data['final_product_name']
    .value_counts()
    .reset_index()
)

user_product_counts.columns = ['final_product_name', 'order_count']

print(user_product_counts)

# Plot bar graph
plt.figure(figsize=(10, 6))
plt.bar(user_product_counts['final_product_name'], user_product_counts['order_count'], color='skyblue')
plt.xlabel('Product Name')
plt.ylabel('Number of Orders')
plt.title(f"Number of Orders for Each Product by User {user_id}")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Attaching a new_column 'order_count_by_173861' to the dataframe
import numpy as np

# Step 1: Filter data for the specific user
user_id = 173861
user_orders_data = new_df[new_df['user_id'] == user_id]

# Step 2: Count product orders by this user
user_product_counts = (
    user_orders_data['final_product_name']
    .value_counts()
    .reset_index()
)
user_product_counts.columns = ['final_product_name', 'order_count']

# Step 3: Create a dictionary for quick lookup
product_order_dict = dict(zip(user_product_counts['final_product_name'], user_product_counts['order_count']))

# Step 4: Create the column using np.where and a list comprehension
new_df['order_count_by_173861'] = np.where(
    new_df['user_id'] == user_id,
    new_df['final_product_name'].map(product_order_dict).fillna(0),
    0
)
new_df.head(10)

new_df[new_df['user_id'] == 173861][['user_id', 'final_product_name', 'order_count_by_173861']].head(10)

#5. From a specific user id find, what all products he has purchased and reordered ratio of the product
# Specify the user_id
user_id =    192463

# Filter the DataFrame for that user
user_data = new_df[new_df['user_id'] == user_id]

# Group by product and calculate reorder stats
product_stats = user_data.groupby('final_product_name')['reordered'].mean().reset_index()

#Rename column for clarity
product_stats.rename(columns={'reordered': 'reordered_ratio'}, inplace=True)

# Save as a new DataFrame
user_product_reorder_df = product_stats

# Print DataFrame to verify
print(user_product_reorder_df)

# Plot as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(user_product_reorder_df['final_product_name'], user_product_reorder_df['reordered_ratio'], color='skyblue')
plt.xlabel('Product')
plt.ylabel('Reordered Ratio')
plt.title(f"Reordered Ratio of Products for User {user_id}")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Attaching a new column 'user_192463_reordered_ratio' to DataFrame
import numpy as np

# Step 1: Specify the user ID
user_id = 192463

# Step 2: Filter data for this user
user_data = new_df[new_df['user_id'] == user_id]

# Step 3: Group by product and compute reordered ratio
product_stats = (
    user_data.groupby('final_product_name')['reordered']
    .mean()
    .reset_index()
)
product_stats.rename(columns={'reordered': 'reordered_ratio'}, inplace=True)

# Step 4: Convert to dictionary for lookup
product_reorder_dict = dict(zip(product_stats['final_product_name'], product_stats['reordered_ratio']))

# Step 5: Attach to main DataFrame using np.where
new_df['user_192463_reordered_ratio'] = np.where(
    new_df['user_id'] == user_id,
    new_df['final_product_name'].map(product_reorder_dict).fillna(0),
    0
)

new_df[new_df['user_id'] == 192463][['user_id', 'final_product_name', 'reordered', 'user_192463_reordered_ratio']].head(10)

"""USER PRODUCT INTERACTION FEATURES (how many times user reordered the product)"""

#1.From a specific user find what all products he has purchased , and find the number of times reordered
# Specify your user ID
user_id =  178958

# Filter the data for this user
user_data = new_df[new_df['user_id'] == user_id]

# Group by product and calculate the number of times reordered
# Assuming 'reordered' is 1 when reordered and 0 when not
product_reorders = user_data.groupby('final_product_name')['reordered'].sum().reset_index()

# Rename columns for clarity
product_reorders.columns = ['final_product_name', 'total_reorders']

# Save this as a new DataFrame
reorders_df = product_reorders.copy()

print(reorders_df)

# Plot as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(reorders_df['final_product_name'], reorders_df['total_reorders'], color='skyblue')
plt.xlabel('Products')
plt.ylabel('Number of Reorders')
plt.title(f"Number of times reordered for each product by user {user_id}")
plt.xticks(rotation=45, ha='right')  # Rotate product names for readability
plt.tight_layout()
plt.show()

#Attaching a new column 'user_178958_total_reorders' to the dataframe
import numpy as np
import pandas as pd

def user_product_reorders(new_df, user_id):
    """
    For a given user_id, compute how many times each product was reordered
    and attach it to the original DataFrame using an if-like condition.
    """
    # Step 1: Filter the data for this user
    user_data = new_df[new_df['user_id'] == user_id]

    # Step 2: Group by product and calculate total reorders
    product_reorders = (
        user_data.groupby('final_product_name')['reordered']
        .sum()
        .reset_index()
    )
    product_reorders.columns = ['final_product_name', 'total_reorders']

    # Step 3: Convert to dictionary for quick lookup
    reorder_dict = dict(zip(product_reorders['final_product_name'], product_reorders['total_reorders']))

    # Step 4: Create new column in the main DataFrame
    new_df[f'user_{user_id}_total_reorders'] = np.where(
        new_df['user_id'] == user_id,
        new_df['final_product_name'].map(reorder_dict).fillna(0),
        0
    )

    return new_df

new_df = user_product_reorders(new_df, user_id=178958)

new_df[new_df['user_id'] == 178958][['user_id', 'final_product_name', 'user_178958_total_reorders']].head(10)

"""PRODUCT BASED FEATURES"""

#1. Find the reorder ratio for each and every product
df_1 = new_df.groupby('final_product_name')['reordered'].value_counts(normalize = True).unstack()
df_1.head(25).plot(kind = 'bar')

# Attaching  new columns '0 and 1' reorder ratio column for the each product
def product_reorder_ratio(new_df):
    """
    Calculates the reorder ratio (reordered = 1 probability) for each product
    and attaches it as a new column to the main DataFrame.
    """
    # Step 1: Group and normalize by product name
    df_ratio = new_df.groupby('final_product_name')['reordered'] \
                 .value_counts(normalize=True) \
                 .unstack(fill_value=0)

    # Step 2: Extract only the probability of reordered = 1
    df_ratio_1 = df_ratio.get(1, pd.Series(0, index=df_ratio.index))  # fallback in case 1 is missing

    # Step 3: Convert to dictionary
    reorder_ratio_dict = df_ratio_1.to_dict()

    # Step 4: Attach to main DataFrame
    new_df['product_reorder_ratio'] = new_df['final_product_name'].map(reorder_ratio_dict).fillna(0)

    return new_df

new_df = product_reorder_ratio(new_df)

#Preview code
df_1 = new_df.groupby('final_product_name')['reordered'].value_counts(normalize=True).unstack()
df_1

#2. From a product, find the number of times reordered for one specific product

import pandas as pd
import matplotlib.pyplot as plt

# Let’s say the specific product you’re analyzing
specific_product = 'strawberries'

# Filter for that product
product_data = new_df[new_df['final_product_name'] == specific_product]

# Extract 'reordered' column (0 or 1)
reorders = product_data['reordered']

print(f"Reorder counts for '{specific_product}':\n{reorders.value_counts()}")

# Plot histogram
plt.figure(figsize=(6, 4))
plt.hist(reorders, bins=2, color='skyblue', edgecolor='black')
plt.xlabel('Reordered (0=No, 1=Yes)')
plt.ylabel('Count')
plt.title(f"Reorder distribution for '{specific_product}'")
plt.xticks([0, 1])
plt.show()

# Attaching a new column ' strawberries_total_reorders' to the new dataframe*

def strawberries_total_reorders(df, product_name):
    """
    For a specific product, count how many times it was reordered (reordered = 1)
    and attach that value as a new column to the DataFrame.
    """
    # Filter for the specific product
    product_data = new_df[new_df['final_product_name'] == product_name]

    # Count number of times it was reordered (i.e., reordered == 1)
    total_reorders = product_data['reordered'].sum()

    # Attach to main DataFrame: if product matches, assign the count, else 0
    new_df[f'{product_name}_total_reorders'] = new_df['final_product_name'].apply(
        lambda x: total_reorders if x == product_name else 0
    )

    return new_df

#PREVIEW CODE
#Attach total reorders column for the product
new_df = strawberries_total_reorders(new_df, 'strawberries')

# Preview the rows related to the specific product
preview = new_df[new_df['final_product_name'] == 'strawberries'][
    ['final_product_name', 'reordered', 'strawberries_total_reorders']
].head(10)

print(preview)

new_df

new_df.isnull().sum()

#Top 10 products with Weekend Sale ratio
import pandas as pd

# 1. Filter to weekend orders (Saturday = 6, Sunday = 0)
new_df['is_weekend'] = new_df['order_dow'].isin([0, 6]).astype(int)

# 2. Total sales per product
product_total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

# 3. Weekend sales per product
product_weekend_sales = new_df[new_df['is_weekend'] == 1].groupby('final_product_name').size().reset_index(name='weekend_sales')

# 4. Merge and compute ratio
product_sales = pd.merge(product_total_sales, product_weekend_sales, on='final_product_name', how='left')
product_sales['weekend_sales'] = product_sales['weekend_sales'].fillna(0)

product_sales['weekend_sale_ratio'] = product_sales['weekend_sales'] / product_sales['total_sales']

# 5. Sort by highest weekend sale ratio
popular_weekend_products = product_sales.sort_values(by='weekend_sale_ratio', ascending = True)

# Optional: Filter products with at least 100 total sales for significance
popular_weekend_products = popular_weekend_products[popular_weekend_products['total_sales']>=1000]

# View top 10
print(popular_weekend_products)

import pandas as pd

def get_top_weekend_products(new_df, min_total_sales=1000, top_n=10, merge_back=False):
    """
    Computes the top N products with the highest weekend sale ratio.

    Parameters:
        new_df (DataFrame): The input DataFrame with 'order_dow' and 'final_product_name' columns.
        min_total_sales (int): Minimum total sales to consider for ratio (default=1000).
        top_n (int): Number of top products to return based on weekend sale ratio.
        merge_back (bool): If True, merges the ratio info back into `new_df`.

    Returns:
        DataFrame: Top N products sorted by weekend sale ratio.
        (optional) DataFrame: Updated new_df with weekend sale ratio per product if merge_back=True.
    """
    # Step 1: Identify weekend orders
    new_df = new_df.copy()
    new_df['is_weekend'] = new_df['order_dow'].isin([0, 6]).astype(int)

    # Step 2: Total sales per product
    product_total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

    # Step 3: Weekend sales per product
    product_weekend_sales = new_df[new_df['is_weekend'] == 1].groupby('final_product_name').size().reset_index(name='weekend_sales')

    # Step 4: Merge and compute ratio
    product_sales = pd.merge(product_total_sales, product_weekend_sales, on='final_product_name', how='left')
    product_sales['weekend_sales'] = product_sales['weekend_sales'].fillna(0)
    product_sales['weekend_sale_ratio'] = product_sales['weekend_sales'] / product_sales['total_sales']

    # Step 5: Filter products with minimum total sales
    product_sales = product_sales[product_sales['total_sales'] >= min_total_sales]

    # Step 6: Sort by highest weekend sale ratio and select top N
    top_products = product_sales.sort_values(by='weekend_sale_ratio', ascending=False).head(top_n)

    def get_top_weekend_products(new_df, min_total_sales=1000, top_n=10, merge_back=False):
    # ... your existing code ...

     return top_products  # <-- Add this line

#Top 10 products with Weekend Sale ratio
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the plot data is sorted and limited to top 10
top_10_weekend_products = popular_weekend_products.tail(10)



# Create a horizontal bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_10_weekend_products,
    x='final_product_name',
    y='weekend_sale_ratio',
    palette="plasma"
)

# Add labels and title
plt.title('Top 10 Products with Highest Weekend Sale Ratio')
plt.xlabel('Product Name')
plt.ylabel('Weekend sale Ratio')
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()

print(new_df.columns)
print(new_df.head())

import pandas as pd

# ✅ No need to recreate 'is_weekday' since it already exists
new_df['is_weekday'] = new_df['order_dow'].isin([1, 2, 3, 4, 5]).astype(int)
# 1. Total number of orders per product
product_total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')



# 2. Total number of weekday orders per product
# ✅ Use the existing 'is_weekday' column
product_weekday_sales = new_df[new_df['is_weekday'] >1] \
    .groupby('final_product_name') \
    .size() \
    .reset_index(name='weekday_sales')

# 3. Merge total and weekday sales
product_sales = pd.merge(product_total_sales, product_weekday_sales, on='final_product_name', how='left')
product_sales['weekday_sales'] = product_sales['weekday_sales'].fillna(0)

# 4. Compute weekday sale ratio
product_sales['weekday_sale_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']

# 5. Sort by highest weekday sale ratio
popular_weekday_products = product_sales.sort_values(by='weekday_sale_ratio', ascending=False)

# 6. Filter products with at least 1000 total sales
popular_weekday_products = popular_weekday_products[popular_weekday_products['total_sales'] >= 1000]

# 7. View top 10 weekday-heavy products
print(popular_weekday_products.head(10))

import pandas as pd

# Ensure 'new_df' has 'order_dow' and 'final_product_name'
# 1. Add a column to indicate if the order was on a weekday (Monday=1 to Friday=5)
new_df['is_weekday'] = new_df['order_dow'].isin([1, 2, 3, 4, 5]).astype(int)

# 2. Total number of orders per product
product_total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

# 3. Total number of weekday orders per product
product_weekday_sales = new_df[new_df['is_weekday'] == 1].groupby('final_product_name').size().reset_index(name='weekday_sales')

# 4. Merge total and weekday sales
product_sales = pd.merge(product_total_sales, product_weekday_sales, on='final_product_name', how='left')
product_sales['weekday_sales'] = product_sales['weekday_sales'].fillna(0)

# 5. Compute weekday sale ratio
product_sales['weekday_sale_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']

# 6. Sort by highest weekday sale ratio
popular_weekday_products = product_sales.sort_values(by='weekday_sale_ratio', ascending=False)

# 7. Optional: Filter products with at least 1000 total sales
popular_weekday_products = popular_weekday_products[popular_weekday_products['total_sales'] >= 1000]

# 8. View top 10 weekday-heavy products
print(popular_weekday_products.head(10))

#Top 10 products with highest Weekday sale ratio
import pandas as pd

# Assuming new_df has 'order_dow' and 'final_product_name' columns

# 1. Add a column to indicate if the order was on a weekday (Monday=1 to Friday=5)
new_df['is_weekday'] = new_df['order_dow'].isin([1, 2, 3, 4, 5]).astype(int)

# 2. Total number of orders per product
product_total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

# 3. Total number of weekday orders per product
product_weekday_sales = new_df[new_df['is_weekday'] == 1].groupby('final_product_name').size().reset_index(name='weekday_sales')

# 4. Merge total and weekday sales, compute weekday sale ratio
product_sales = pd.merge(product_total_sales, product_weekday_sales, on='final_product_name', how='left')
product_sales['weekday_sales'] = product_sales['weekday_sales'].fillna(0)

product_sales['weekday_sale_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']

# 5. Sort by highest weekday sale ratio
popular_weekday_products = product_sales.sort_values(by='weekday_sale_ratio', ascending=False)

# Optional: Filter to only products with at least 1000 total sales
popular_weekday_products = popular_weekday_products[popular_weekday_products['total_sales'] >= 1000]

# View top 10 weekday-heavy products
print(popular_weekday_products.head(10))

# 1. Total number of sales per product
product_total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

# 2. Total number of weekday orders per product (from new_df to match above)
product_weekday_sales = new_df[new_df['is_weekday'] == 1].groupby('final_product_name').size().reset_index(name='weekday_sales')

# 3. Merge total and weekday sales
product_sales = pd.merge(product_total_sales, product_weekday_sales, on='final_product_name', how='left')
product_sales['weekday_sales'] = product_sales['weekday_sales'].fillna(0)

# 4. Compute weekday sale ratio
product_sales['weekday_sale_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']

# 5. Sort by highest weekday sale ratio
popular_weekday_products = product_sales.sort_values(by='weekday_sale_ratio', ascending=False)

# 6. Optional: Filter products with at least 1000 total sales
popular_weekday_products = popular_weekday_products[popular_weekday_products['total_sales'] >= 1000]

# 7. View top 10 weekday-heavy products
print(popular_weekday_products.head(10))

# 6. Select top 10
top_10 = popular_weekday_products.head(10)

# 7. Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x='weekday_sale_ratio', y='final_product_name', palette='Blues_d')
plt.title('Top 10 Products with Highest Weekday Sale Ratio')
plt.xlabel('Weekday Sale Ratio')
plt.ylabel('Product Name')
plt.tight_layout()
plt.show()

# Comparison of the Weekday and Weekend Sales
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create weekday and weekend flags
new_df['is_weekday'] = new_df['order_dow'].isin([1, 2, 3, 4, 5]).astype(int)
new_df['is_weekend'] = new_df['order_dow'].isin([0, 6]).astype(int)

# 2. Total sales per product
total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

# 3. Weekday sales
weekday_sales = new_df[new_df['is_weekday'] == 1].groupby('final_product_name').size().reset_index(name='weekday_sales')

# 4. Weekend sales
weekend_sales = new_df[new_df['is_weekend'] == 1].groupby('final_product_name').size().reset_index(name='weekend_sales')

# 5. Merge all
product_sales = total_sales.merge(weekday_sales, on='final_product_name', how='left') \
                           .merge(weekend_sales, on='final_product_name', how='left')

# Fill NaNs with 0
product_sales[['weekday_sales', 'weekend_sales']] = product_sales[['weekday_sales', 'weekend_sales']].fillna(0)

# 6. Compute ratios
product_sales['weekday_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']
product_sales['weekend_ratio'] = product_sales['weekend_sales'] / product_sales['total_sales']

# 7. Filter products with sufficient sales
product_sales = product_sales[product_sales['total_sales'] >= 1000]

# 8. Difference in ratios
product_sales['ratio_diff'] = (product_sales['weekday_ratio'] - product_sales['weekend_ratio']).abs()

# 9. Top 10 products with highest difference in behavior
top_diff_products = product_sales.sort_values(by='ratio_diff', ascending=False)

# 10. Melt for plotting
melted = top_diff_products.melt(id_vars='final_product_name',
                                value_vars=['weekday_ratio', 'weekend_ratio'],
                                var_name='Day_Type', value_name='Sale_Ratio')

# 11. Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x='Sale_Ratio', y='final_product_name', hue='Day_Type', palette='Set1')
plt.title('Products: Weekday vs Weekend Sale Ratio')
plt.xlabel('Sale Ratio')
plt.ylabel('Product Name')
plt.legend(title='Day Type')
plt.tight_layout()
plt.show()

import pandas as pd

def analyze_weekday_weekend_sales(data: pd.DataFrame, min_sales_threshold: int = 1000) -> pd.DataFrame:
    """
    Analyze weekday vs weekend sales behavior for products.

    Parameters:
    - data: DataFrame containing at least 'final_product_name' and 'order_dow'
    - min_sales_threshold: Minimum total sales per product to be included in the analysis

    Returns:
    - DataFrame with total, weekday, weekend sales, ratios, and absolute ratio difference
    """

    # Add weekday/weekend flags
    data['is_weekday'] = data['order_dow'].isin([1, 2, 3, 4, 5]).astype(int)
    data['is_weekend'] = data['order_dow'].isin([0, 6]).astype(int)

    # Aggregate sales
    total_sales = data.groupby('final_product_name').size().reset_index(name='total_sales')
    weekday_sales = data[data['is_weekday'] == 1].groupby('final_product_name').size().reset_index(name='weekday_sales')
    weekend_sales = data[data['is_weekend'] == 1].groupby('final_product_name').size().reset_index(name='weekend_sales')

    # Merge all
    product_sales = total_sales.merge(weekday_sales, on='final_product_name', how='left') \
                               .merge(weekend_sales, on='final_product_name', how='left')

    # Fill missing sales with 0
    product_sales[['weekday_sales', 'weekend_sales']] = product_sales[['weekday_sales', 'weekend_sales']].fillna(0)

    # Compute ratios
    product_sales['weekday_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']
    product_sales['weekend_ratio'] = product_sales['weekend_sales'] / product_sales['total_sales']

    # Filter products with sufficient sales
    product_sales = product_sales[product_sales['total_sales'] >= min_sales_threshold]

    # Calculate absolute difference in ratios
    product_sales['ratio_diff'] = (product_sales['weekday_ratio'] - product_sales['weekend_ratio']).abs()

    # Sort by difference
    product_sales = product_sales.sort_values(by='ratio_diff', ascending=False)

    return product_sales

df = new_df
result_df = analyze_weekday_weekend_sales(df)

# Show top 10 products with most distinct weekday vs weekend behavior
top10 = result_df.head(10)
print(top10[['final_product_name', 'total_sales', 'weekday_ratio', 'weekend_ratio', 'ratio_diff']])

# Compute absolute difference in ratios
product_sales['ratio_diff'] = (product_sales['weekday_ratio'] - product_sales['weekend_ratio']).abs()

product_sales['weekday_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']
product_sales['weekend_ratio'] = product_sales['weekend_sales'] / product_sales['total_sales']

# ✅ Add this before merge
product_sales['ratio_diff'] = (product_sales['weekday_ratio'] - product_sales['weekend_ratio']).abs()

# Merge selected sales info into new_df
new_df = new_df.merge(
    product_sales[['final_product_name', 'total_sales', 'weekday_sales', 'weekend_sales',
                   'weekday_ratio', 'weekend_ratio', 'ratio_diff']],
    on='final_product_name',
    how='left'
)

# Fill missing values with 0 (for products not in product_sales)
cols_to_fill = ['total_sales', 'weekday_sales', 'weekend_sales',
                'weekday_ratio', 'weekend_ratio', 'ratio_diff']

new_df[cols_to_fill] = new_df[cols_to_fill].fillna(0)

# Optional: display the updated DataFrame
print(new_df.head())

new_df

#Comparison of Top5 Weekday and Top5 Weekend sales products
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Flags for weekday and weekend
new_df['is_weekday'] = new_df['order_dow'].isin([1, 2, 3, 4, 5]).astype(int)
new_df['is_weekend'] = new_df['order_dow'].isin([0, 6]).astype(int)

# 2. Total sales
total_sales = new_df.groupby('final_product_name').size().reset_index(name='total_sales')

# 3. Weekday sales
weekday_sales = new_df[new_df['is_weekday'] == 1].groupby('final_product_name').size().reset_index(name='weekday_sales')

# 4. Weekend sales
weekend_sales = new_df[new_df['is_weekend'] == 1].groupby('final_product_name').size().reset_index(name='weekend_sales')

# 5. Merge
product_sales = total_sales.merge(weekday_sales, on='final_product_name', how='left') \
                           .merge(weekend_sales, on='final_product_name', how='left')
product_sales[['weekday_sales', 'weekend_sales']] = product_sales[['weekday_sales', 'weekend_sales']].fillna(0)

# 6. Compute ratios
product_sales['weekday_ratio'] = product_sales['weekday_sales'] / product_sales['total_sales']
product_sales['weekend_ratio'] = product_sales['weekend_sales'] / product_sales['total_sales']

# 7. Filter by significance
product_sales = product_sales[product_sales['total_sales'] >= 1000]

# 8. Get top 10 weekend-heavy products
top_weekend = product_sales.sort_values(by='weekend_ratio', ascending=False).head(5)

# 9. Get top 10 weekday-heavy products
top_weekday = product_sales.sort_values(by='weekday_ratio', ascending=False).head(5)

# 10. Combine and drop duplicates (in case of overlap)
top_combined = pd.concat([top_weekend, top_weekday]).drop_duplicates(subset='final_product_name')

# 11. Melt for plotting
melted = top_combined.melt(id_vars='final_product_name',
                           value_vars=['weekday_ratio', 'weekend_ratio'],
                           var_name='Day_Type', value_name='Sale_Ratio')

# 12. Plot
plt.figure(figsize=(12, 7))
sns.barplot(data=melted, x='Sale_Ratio', y='final_product_name', hue='Day_Type', palette='Set2')
plt.title('Top 5 Weekday vs Weekend Products')
plt.xlabel('Sale Ratio')
plt.ylabel('Product Name')
plt.legend(title='Day Type')
plt.tight_layout()
plt.show()

new_df.isnull().sum()

new_df.to_csv('before_encoding.csv', index = False)


#06. "SAMPLING"  [Reducing the dataset size for the purpose of DL]
import pandas as pd
import numpy as np

new_df = pd.read_csv(r"M:\DL\DL_FINAL\before_encoding.csv")
pd.set_option('display.max_columns',None)
new_df.head(2)

# Randomly keep 75% of the data
reduced_df = new_df.sample(frac=0.50, random_state=42)  # random_state ensures reproducibility

# Save reduced dataset (optional)
reduced_df.to_csv("reduced_dataset.csv", index=False)

# Show results
print(f"Original dataset size: {len(new_df)}")
print(f"Reduced dataset size: {len(reduced_df)}")
print(reduced_df.head())

#07. "ENCODING"  [Encoding and Scaling]
import pandas as pd
import numpy as np
import sklearn

df = pd.read_csv(r"M:\DL\DL_FINAL\reduced_dataset.csv")

pd.set_option('display.max_columns',None)

df.head(5)

print(df.dtypes)

import pandas as pd

df['reordered'] = df['reordered'].astype(float)
df['order_dow'] = df['order_dow'].astype(float)
df['order_hour_of_day'] = df['order_hour_of_day'].astype(float)
df['organic_fuji_apple_by_145501'] = df['organic_fuji_apple_by_145501'].astype(float)
df['total_sales'] = df['total_sales'].astype(float)
df['weekday_sales'] = df['weekday_sales'].astype(float)
df['weekend_sales'] = df['weekend_sales'].astype(float)
df['strawberries_total_reorders'] = df['strawberries_total_reorders'].astype(float)

columns_to_drop = ['order_id', 'user_id', 'eval_set', 'add_to_cart_order_log', 'order_number', 'final_aisle_id','final_product_id',	'final_department_id',
				'is_weekday',	'is_weekend', 'avg_days_btn_orders_for_user_161125',
                     'organic_fuji_apple_by_145501', 'order_count_by_173861', 'user_192463_reordered_ratio', 'user_178958_total_reorders',
                         'strawberries_total_reorders', 'weekday_ratio', 'weekend_ratio', 'ratio_diff']

df.drop(columns= columns_to_drop, inplace=True)

nominal_cols = ['final_product_name', 'final_department_name', 'final_aisle_name']

df.dtypes

from sklearn.preprocessing import OneHotEncoder
import pickle

ohe = OneHotEncoder()
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = ohe.fit_transform(df[nominal_cols])




encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(nominal_cols))


# Save the fitted OneHotEncoder to a pickle file
with open('onehot_encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)

encoded_df

from sklearn.preprocessing import MinMaxScaler
import pickle

# Columns to normalize
cols_to_normalize = ['add_to_cart_order',  'order_dow', 'order_hour_of_day','days_since_prior_order','product_reorder_ratio', 'total_sales', 'weekday_sales','weekend_sales']


scaler = MinMaxScaler()

# Fit and transform specified columns and create a new DataFrame
df_normalized = df.copy()  # Optional: copy the original to preserve it

df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
# Save the scaler as a pickle file
with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(df_normalized.head())

df_final = pd.concat([encoded_df, df_normalized], axis=1)
df_final

df_final.to_csv('before_splitting.csv', index = False)



#08. "SPLITTING" [TRAINING AND TESTING THE MODEL]

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import tensorflow as tf

df = pd.read_csv(r"M:\DL\DL_FINAL\before_splitting.csv")
pd.set_option('display.max_columns',None)
df

df.isnull().sum().value_counts()

columns_to_drop = ['final_product_name', 'final_department_name', 'final_aisle_name' ]

df.drop(columns= columns_to_drop, inplace=True)

y = df["reordered"]

x = df.drop(columns = "reordered", axis = 1)

x.shape

y.shape

y

"""ONE HOT ENCODER FOR CATEGORICAL COLUMN"""

#One hot encoder used and it stores the data in matrix format
from tensorflow.keras.utils import to_categorical

ynew = to_categorical(y)
ynew

ynew.shape

"""TRAIN TEST SPLIT"""

from sklearn.model_selection import train_test_split

x_train, x_test, ynew_train, ynew_test = train_test_split(x,ynew, test_size = 0.25)

"""DESIGNING THE ARCHITECTURE"""

x_train.shape

import joblib

# Save the list of features
joblib.dump(list(x_train.columns), "features_used.pkl")

x_train.shape[1]

12682164*0.1

(12682164-1268216.4)/4096

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# ---------------------------
# Sample input (define these properly before running this script)
# Example placeholders (replace with your actual data loading logic)
# x should be of shape (n_samples, 81), y should be 0 or 1
# ---------------------------
# For demonstration, using random data — replace with your own
x = np.random.rand(120000, 81)  # Feature matrix
y = np.random.randint(0, 2, size=(120000,))  # Binary labels (0 or 1)

# One-hot encode the labels
ynew = tf.keras.utils.to_categorical(y, num_classes=2)

# Split the dataset
x_train, x_test, ynew_train, ynew_test = train_test_split(x, ynew, test_size=0.25, random_state=42)

# ---------------------------
# Define custom F1 Score metric
# ---------------------------
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities to predicted class labels
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

# ---------------------------
# Build the model
# ---------------------------
model = Sequential([
    Input(shape=(81,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Output layer for two classes
])

# ---------------------------
# Compile the model
# ---------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Because labels are one-hot encoded
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        F1Score()
    ]
)

# ---------------------------
# Train the model (small batch for quick run)
# ---------------------------
model.fit(
    x_train[:1000000], ynew_train[:1000000],  # Train on subset
    epochs=100,
    batch_size=512,
    validation_data=(x_test[:500000], ynew_test[:500000]),
    verbose=1
)

# ---------------------------
# Save the model
# ---------------------------
model.save("model.keras", save_format="keras")
print("✅ Model saved as model.keras")

import os

file_name = "model.keras"

# Check if file exists
if os.path.exists(file_name):
    full_path = os.path.abspath(file_name)
    print("✅ Model exists.")
    print("📍 Full path:", full_path)
else:
    print("❌ model.keras not found in current directory.")

model.summary()

#result_df = pd.DataFrame(result.history)     #gives the reult in json file
#result_df

"""CONFUSION MATRIX"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Generate synthetic binary classification data
X, y = make_classification(n_samples=3992958, n_features=80,
                           n_informative=79, n_redundant=0,
                           random_state=42)

# Split into train and test sets
x_train, x_test, ynew_train, ynew_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)

# Train a binary classifier
model = LogisticRegression()
model.fit(x_train, ynew_train)

# Predict class labels
y_pred = model.predict(x_test)

# Compute confusion matrix
cm = confusion_matrix(ynew_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Optional: classification report
print("\nClassification Report:")
print(classification_report(ynew_test, y_pred))

# Plot confusion matrix with labels and percentages
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['0', '1']
counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels_full = [f"{l}\n{c}\n{p}" for l, c, p in zip(labels, counts, percentages)]
labels_matrix = np.asarray(labels_full).reshape(2, 2)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=labels_matrix, fmt='', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Binary Classification Confusion Matrix')
plt.show()

"""ROC-AUC CURVE"""

#import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential



#Compute ROC curve
fpr, tpr, thresholds = roc_curve(ynew_test, y_pred)

# Compute AUC score
auc_score = roc_auc_score(ynew_test, y_pred)
print(f"AUC: {auc_score:.3f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Deep Learning)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

"""📈 What is AUC?
AUC = 1.0 → Perfect classifier.

AUC = 0.5 → Model is guessing randomly (bad).

AUC > 0.9 → Excellent model.

AUC ~ 0.7–0.8 → Decent performance.

AUC < 0.5 → Worse than random (model may be misclassifying).

"""

#09. "STREAMLIT CODE" [FOR DEPLOYING THE DEEP LEARNING MODEL]

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



























