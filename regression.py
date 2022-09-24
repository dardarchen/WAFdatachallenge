import pandas as pd 

df = pd.read_csv("cleaned_data.csv") 

# Final data cleaning
indices = df["description"] != df["description"]
indices = [not x for x in indices]
df = df[indices] 
indices = df["host_location"] != df["host_location"]
indices = [not x for x in indices]
df = df[indices]
df = df.drop(["name", "description", "host_location", "host_neighbourhood", 
"host_verifications", "neighbourhood", "neighbourhood_cleansed", "property_type", 
"room_type", "amenities", "first_review", "last_review"])

# Creating categorical variables 
new_col = df["neighborhood_overview"] == df["neighborhood_overview"] 
new_col = [1 if x else 0 for x in new_col]
df["neighborhood_overview"] = new_col
new_col = df["host_about"] == df["host_about"] 
new_col = [1 if x else 0 for x in new_col]
df["host_about"] = new_col

# Converting text variables to numerical ones 
host_is_superhost = [1 if x else 0 for x in df["host_is_superhost"]]
host_identity_verified = [1 if x else 0 for x in df["host_identity_verified"]]
df["host_is_superhost"] = host_is_superhost 
df["host_identity_verified"] = host_identity_verified 
df = df.drop("host_has_profile_pic", axis=1)

# Turning date into just the year
years = [int(x[:4]) for x in df["host_since"]]
df["host_since"] = years

# Turning response time into a number
response_speed = {"within an hour":1, "within a few hours":2, "within a day":3, "a few days or more":4}
times = [response_speed[x] for x in df["host_response_time"]]
df["host_response_time"] = times

# Turning percentages into decimals 
host_response_rate = [int(x[:x.index("%")])/100 for x in df["host_response_rate"]]
df["host_response_rate"] = host_response_rate
host_acceptance_rate = [int(x[:x.index("%")])/100 for x in df["host_acceptance_rate"]]
df["host_acceptance_rate"] = host_acceptance_rate
df.drop("bathrooms_text", axis=1) 

# Removing outlier prices, ratings, and response rates
df = df[df["price"] < 2000]
df = df[df["review_scores_rating"] > 3]
df = df[df["host_response_rate"] > 0.5]
df = df.drop("bathrooms_text", axis=1) 

# Creating a correlation map between all variables 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import StandardScaler

scaler = StandardScaler()

map = pd.read_csv("processed_data.csv")
map = map.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
map.columns = ["NO", "HS", "HA", "HRT", "HRR", "HAR", "HIS", "HLC", "HIV", "LA", 
               "LO", "ACC", "BR", "BED", "PR", "MIN", "MAX", "NUM", "RSR", "RSA",
               "RSC", "RSCH", "RSCO", "RSL", "RSV", "RPM"]
map = map[["HLC", "HIV", "LA", 
               "LO", "ACC", "BR", "BED", "PR", "MIN", "MAX", "NUM", "RSR", "RSA",
               "RSC", "RSCH", "RSCO", "RSL", "RSV", "RPM"]]
corr = map.corr()
f, ax = plt.subplots(figsize=(20, 20))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, cmap=cmap)
plt.savefig("heatmap.png")

# Creating training and testing data 
#Next, we need to standardize all the data before passing into linear regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

# Getting train and test data
data = pd.read_csv("processed_data.csv")
data = data.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
data.columns = ["NO", "HS", "HA", "HRT", "HRR", "HAR", "HIS", "HLC", "HIV", "LA", 
               "LO", "ACC", "BR", "BED", "PR", "MIN", "MAX", "NUM", "RSR", "RSA",
               "RSC", "RSCH", "RSCO", "RSL", "RSV", "RPM"]
data = data[["HLC", "LA", "LO", "ACC", "NUM", "RSR", "RPM", "PR"]]
train = data.sample(frac=0.9)
train_label = train[["PR"]]
train_data = train.drop("PR", axis=1)
test = data.drop(train.index)
test_label = test[["PR"]]
test_data = test.drop("PR", axis=1)

# Standardizing train and test data 
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
train_label = np.array(train_label)
test_data = scaler.fit_transform(test_data)
test_label = np.array(test_label)

# Running linear regression
reg = LinearRegression()
reg.fit(train_data, train_label)
reg.score(train_data, train_label) 

# Running linear regression 100 times, resampling each time, to look at average performance 
scaler = StandardScaler()
data = pd.read_csv("processed_data.csv")
data = data.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
data.columns = ["NO", "HS", "HA", "HRT", "HRR", "HAR", "HIS", "HLC", "HIV", "LA", 
              "LO", "ACC", "BR", "BED", "PR", "MIN", "MAX", "NUM", "RSR", "RSA",
              "RSC", "RSCH", "RSCO", "RSL", "RSV", "RPM"]
data = data[["HLC", "LA", "LO", "ACC", "NUM", "RSR", "RPM", "PR"]]

n = 100
total_train = 0 
total_test = 0
total_coeffs = np.zeros(7)
for i in range(n):
  train = data.sample(frac=0.9)
  train_label = train[["PR"]]
  train_data = train.drop("PR", axis=1)
  test = data.drop(train.index)
  test_label = test[["PR"]]
  test_data = test.drop("PR", axis=1)

  train_data = scaler.fit_transform(train_data)
  train_label = np.array(train_label)

  test_data = scaler.fit_transform(test_data)
  test_label = np.array(test_label)

  reg = LinearRegression()
  reg.fit(train_data, train_label)
  total_train += reg.score(train_data, train_label)
  total_test += reg.score(test_data, test_label)
  total_coeffs += reg.coef_[0]

average_train = total_train / n
average_test = total_test / n
average_coeffs = total_coeffs / n
print(average_train, average_test)
print(average_coeffs)

# Finding most important predictors
predictors = ["HLC", "LA", "LO", "ACC", "NUM", "RSR", "RPM", "PR"]
average_coeffs

zipped_lists = zip(average_coeffs, predictors)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
list1, list2 = [ list(tuple) for tuple in tuples]

coefficients = pd.DataFrame(data={"Predictor":list2, "Coefficient":list1})
sns.barplot(data=coefficients, x="Predictor", y="Coefficient", color="indianred")
plt.title("Predictor vs Coefficient in Linear Regression Model")
plt.savefig("coefficients.png")