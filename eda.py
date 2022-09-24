import pandas as pd 

df = pd.read_csv("f2022_datachallenge.csv")
df.isna().sum()

# First, we find the NaN values in the dataframe and drop the corresponding entries. 
# Looks like host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost, 
# host_listings_count, host_has_profile_pic, host_identity_verified, bedrooms, beds. 
df = df[(df["host_is_superhost"] == True) | (df["host_is_superhost"] == False)]
indices = df["host_response_time"] != df["host_response_time"]
indices = [not x for x in indices]
df = df[indices]
indices = df["review_scores_value"] != df["review_scores_value"]
indices = [not x for x in indices]
df = df[indices]
indices = (df["bedrooms"] != df["bedrooms"]) | (df["beds"] != df["beds"])
indices = [not x for x in indices]
df = df[indices]
indices = df["host_acceptance_rate"] != df["host_acceptance_rate"]
indices = [not x for x in indices]
df = df[indices]
indices = df["bathrooms_text"] != df["bathrooms_text"]
indices = [not x for x in indices]
df = df[indices]

# Creating geographical plots 
import geoplot as gplt
import geopandas as gpd

usa = gpd.read_file("geospatialdata.json")
prices = df[["price", "longitude", "latitude"]]

from shapely.geometry import Point

prices["coordinates"] = list(prices[["longitude", "latitude"]].values)
prices["coordinates"] = prices["coordinates"].apply(Point)
prices_gsp = gpd.GeoDataFrame(prices, geometry="coordinates")

# Finding and removing outliers in price
outliers = prices["price"] > 500
keep = [not x for x in outliers]
prices_normal = prices[keep]
prices_outlier = prices[outliers]
prices_normal_gsp = gpd.GeoDataFrame(prices_normal, geometry="coordinates")
prices_outlier_gsp = gpd.GeoDataFrame(prices_outlier, geometry="coordinates")

#Plotting the normal prices 
fig, ax = plt.subplots(1, figsize=(10, 6))
base = usa[usa["NAME"] == "Hawaii"].plot(ax=ax, color="darkgray")
prices_normal_gsp.plot(ax=base, column="price", marker = "o", markersize=12, figsize=(15, 10), cmap="plasma", 
                       legend = True, legend_kwds={'label': "Price in local currency",'orientation':"vertical"})
_ = ax.axis("off")
ax.set_title("Normal Daily Airbnb Prices Across Hawaii", fontsize=15)
plt.savefig('normal_prices.png',bbox_inches='tight')

# Plotting num of people accomodated
accommodates = df[["accommodates", "longitude", "latitude"]]
accommodates["coordinates"] = list(accommodates[["longitude", "latitude"]].values)
accommodates["coordinates"] = accommodates["coordinates"].apply(Point)
accommodates.hist(column="accommodates")

# Splitting into above and under 9 
less_than_nine = accommodates[accommodates["accommodates"] < 9]
more_than_nine = accommodates[accommodates["accommodates"] >= 9]
one_gsp = gpd.GeoDataFrame(less_than_nine, geometry="coordinates")
two_gsp = gpd.GeoDataFrame(more_than_nine, geometry="coordinates")

# Plotting less than nine
fig, ax = plt.subplots(1, figsize=(10, 6))
base = usa[usa["NAME"] == "Hawaii"].plot(ax=ax, color="darkgray")
one_gsp.plot(ax=base, column="accommodates", marker = "o", markersize=12, figsize=(15, 10), cmap="viridis", 
                       legend = True, legend_kwds={'label': "Maximum # of accommodatable people",'orientation':"vertical"})
_ = ax.axis("off")
ax.set_title("Airbnb Accommodations Across Hawaii (<9)", fontsize=15)
plt.savefig('accommodates_less.png',bbox_inches='tight')

# Making histogram plots 
import seaborn as sns 

price = df[["price"]]
sns.histplot(data=price, x="price", bins=30, color="rebeccapurple", log_scale=True)
plt.savefig('price_histogram.png',bbox_inches='tight')

accommodates = df[["accommodates"]]
sns.histplot(data=accommodates, x="accommodates", bins=10, color="teal")
plt.savefig('accommodates_histogram.png',bbox_inches='tight')
