import pandas as pd

df = pd.read_csv("processed_data.csv")
df = df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

# We can't use kmeans on categorical data. For instance, if categories were 1=banana, 2=orange, 3=pear, 
# kmeans would consider banana closer to orange as opposed to pear because the cost function used 
# is euclidean distance.
df = df.drop(["neighborhood_overview", "host_about", "host_response_time", "host_is_superhost", 
    "host_identity_verified"], axis=1)

# Owing to the curse of dimensionality (the higher the number of features, the further apart the 
# vectors become) which can make clustering difficult, we want to further reduce the number of features.
# We can use only review_scores_rating and accommodates and drop their related predictors.
df = df.drop(["bedrooms", "beds", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
    "review_scores_communication", "review_scores_location", "review_scores_value"], axis=1)

# To even further reduce the dimensions, we'll standardize the data and run PCA.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

scaler = StandardScaler()
data = scaler.fit_transform(df)
pca = PCA(n_components=3) 
pca.fit(data)
reduced = pca.transform(data)

# Now, we can perform the clustering. But first, we visualize our data with a 3D plot.
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

pca_data = pd.DataFrame(data={"Component 1": reduced[:, 0], "Component 2": reduced[:, 1], "Component 3": reduced[:, 2]})
pca_data = pca_data.sample(frac=0.1)
x = pca_data["Component 1"]
y = pca_data["Component 2"]
z = pca_data["Component 3"]

ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

ax.scatter(x, y, z)

plt.show()

# Visually, it looks like there are 4 different groups on the component 1 and component 2 plane.
# We choose to consider just these 2 components to cluster on. 

# Spectral clustering
from sklearn.cluster import SpectralClustering
import seaborn as sns

test = pca_data.copy() 
test["Component 1"] = 2 * test["Component 1"]
test["Component 2"] = 2 * test["Component 2"]
cluster = SpectralClustering(n_clusters=5, assign_labels='discretize').fit(test)
test["Group"] = cluster.labels_
test["Group"] = test["Group"] + 1
test = test[test["Group"] != 3]
test["Group"] = [x if x == 1 or x == 2 else x - 1 for x in test["Group"]]
sns.scatterplot(data=test, x="Component 1", y="Component 2", hue="Group", palette="Spectral")
plt.title("PCA Components of Listings")
plt.savefig("spectral.png")

# Visualizing the clusters geographically
from shapely.geometry import Point

locations = df.loc[test.index][["latitude", "longitude"]]
locations = locations.join(test[["Group"]])
locations["geometry"] = list(locations[["longitude", "latitude"]].values)
locations["geometry"] = locations["geometry"].apply(Point)

import geoplot as gplt
import geopandas as gpd

usa = gpd.read_file("geospatialdata.json")
locations_gsp = gpd.GeoDataFrame(locations, geometry="geometry")
fig, ax = plt.subplots(1, figsize=(10, 6))
base = usa[usa["NAME"] == "Hawaii"].plot(ax=ax, color="darkgray")
locations_gsp.plot(ax=base, column="Group", cmap="Spectral", markersize=10, vmin=1,vmax=4,
                       legend = False, legend_kwds={'label': "Group",'orientation':"vertical"})
_ = ax.axis("off")
ax.set_title("Clustered Groups Across Hawaii", fontsize=15)
plt.savefig('clustered_map.png',bbox_inches='tight')