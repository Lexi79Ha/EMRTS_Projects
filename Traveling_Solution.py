import pandas as pd

# %%
capitals = pd.read_csv(r"C:\Users\lmhmo\PycharmProjects\Traveling_Politcian\.venv\us-state-capitals.csv")
# %%

# %%
import matplotlib.pyplot as plt

plt.scatter(capitals['longitude'], capitals['latitude'])
# %%
from sklearn.cluster import KMeans

# Create a KMeans instance with the desired number of clusters and n_init
kmeans = KMeans(n_clusters=14, n_init=5, random_state=42)

# Fit the model to the longitude and latitude data
kmeans.fit(capitals[['longitude', 'latitude']])

# Predict the cluster for each data point
capitals['cluster'] = kmeans.predict(capitals[['longitude', 'latitude']])

# %%
plt.scatter(capitals['longitude'], capitals['latitude'], c=capitals['cluster'])
plt.show(
)
# %%
# Create a scatter plot
plt.scatter(capitals['longitude'], capitals['latitude'], c=capitals['cluster'])

# Calculate the cluster centers
centers = kmeans.cluster_centers_

# Loop through the cluster centers
for i, center in enumerate(centers):
    # Annotate the plot with the cluster numbers at the cluster centers
    plt.annotate(i, (center[0], center[1]), fontsize=12, color='red')

# Display the plot
plt.show()

# %%
import folium

map = folium.Map(location=[capitals['latitude'].mean(), capitals['longitude'].mean()], zoom_start=4)

# Define a color palette for 10 groups
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'pink', 'gray', 'black', "yellow", "pink",
          "darkred", "darkblue", "darkgreen"]

# Add markers to the map
for _, row in capitals.iterrows():
    color = colors[row['cluster']]
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        tooltip=f"{row['name']} (Cluster {row['cluster']})"
    ).add_to(map)

# Display the map
map
# %%
clusters = {}

# Loop over the rows in the DataFrame
for _, row in capitals.iterrows():
    # If the cluster is not in the dictionary, add it
    if row['cluster'] not in clusters:
        clusters[row['cluster']] = []

    # Add the name and description of the location to the cluster
    clusters[row['cluster']].append({'name': row['name'], "latitude": row['latitude'], 'longitude': row['longitude']})

# Get the cluster numbers in sorted order
sorted_clusters = sorted(clusters.keys())


# %%
import itertools
from geopy.distance import geodesic

# %%
cluster_7 = clusters[7]

start_location_name = 'Iowa'
start_location = next((d for d in cluster_7 if d["name"] == start_location_name), None)
# %%
if start_location is None:
    print(f"No location named {start_location_name} found in cluster_7")

else:
    cluster_7 = [d for d in cluster_7 if d["name"] != start_location_name]  # Add start_location to cluster_10
    routes = list(itertools.permutations(cluster_7))
    routes = [(start_location,) + route for route in routes]
# %%
distances = []
for route in routes:
    total_distance = 0
    for i in range(len(route) - 1):
        location1 = route[i]
        location2 = route[i + 1]
        distance = geodesic((location1['latitude'], location1['longitude']),
                            (location2['latitude'], location2['longitude'])).miles
        total_distance += distance
    distances.append(total_distance)

# Find the shortest route
min_distance7 = min(distances)
min_route7 = routes[distances.index(min_distance7)]


# %%
last_state_7 = min_route7[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_7 = clusters[7]  # Assuming clusters[2] is defined
cluster_2 = clusters[2]

last_state_7 = min_route7[-1]
start_location_name = last_state_7['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_7 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_7")
else:
    # Add start_location to cluster_10
    cluster_2.append(start_location)

    # Create a copy of cluster_10 without the start_location for generating the permutations
    cluster_2_without_start = [location for location in cluster_2 if location["name"] != start_location_name]
    routes = list(itertools.permutations(cluster_2_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance2 = min(distances)
    min_route2 = routes[distances.index(min_distance2)]



# %%
last_state_2 = min_route2[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_2 = clusters[2]  # Assuming clusters[2] is defined
cluster_6 = clusters[6]

last_state_2 = min_route2[-1]
start_location_name = last_state_2['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_2 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_2")
else:
    # Add start_location to cluster_10
    cluster_6.append(start_location)

    # Create a copy of cluster_10 without the start_location for generating the permutations
    cluster_6_without_start = [location for location in cluster_6 if location["name"] != start_location_name]
    routes = list(itertools.permutations(cluster_6_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance6 = min(distances)
    min_route6 = routes[distances.index(min_distance6)]



# %%
last_state_6 = min_route6[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_6 = clusters[6]  # Assuming clusters[2] is defined
cluster_8 = clusters[8]

last_state_6 = min_route6[-1]
start_location_name = last_state_6['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_6 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_6")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_8 = [location for location in cluster_8 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_8.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_8_without_start = [location for location in cluster_8 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_8_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance8 = min(distances)
    min_route8 = routes[distances.index(min_distance8)]



# %%
last_state_8 = min_route8[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_8 = clusters[8]  # Assuming clusters[2] is defined
cluster_0 = clusters[0]

last_state_8 = min_route8[-1]
start_location_name = last_state_8['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_8 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_8")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_0 = [location for location in cluster_0 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_0.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_0_without_start = [location for location in cluster_0 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_0_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance0 = min(distances)
    min_route0 = routes[distances.index(min_distance0)]


# %%

last_state_0 = min_route0[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_0 = clusters[0]
cluster_11 = clusters[11]  # Assuming clusters[2] is defined

last_state_0 = min_route0[-1]
start_location_name = last_state_0['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_0 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_0")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_11 = [location for location in cluster_11 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_11.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_11_without_start = [location for location in cluster_11 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_11_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance11 = min(distances)
    min_route11 = routes[distances.index(min_distance11)]


# %%
last_state_11 = min_route11[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_11 = clusters[11]
cluster_4 = clusters[4]  # Assuming clusters[2] is defined

last_state_11 = min_route11[-1]
start_location_name = last_state_11['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_11 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_11")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_4 = [location for location in cluster_4 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_4.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_4_without_start = [location for location in cluster_4 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_4_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance4 = min(distances)
    min_route4 = routes[distances.index(min_distance4)]


# %%
last_state_4 = min_route4[-1]
# %%
import itertools
from geopy.distance import geodesic

cluster_4 = clusters[4]
cluster_10 = clusters[10]  # Assuming clusters[2] is defined

last_state_4 = min_route4[-1]
start_location_name = last_state_4['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_4 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_4")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_10 = [location for location in cluster_10 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_10.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_10_without_start = [location for location in cluster_10 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_10_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance10 = min(distances)
    min_route10 = routes[distances.index(min_distance10)]


# %%
last_state_10 = min_route10[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_10 = clusters[10]
cluster_3 = clusters[3]  # Assuming clusters[2] is defined

last_state_10 = min_route10[-1]
start_location_name = last_state_10['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_10 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_10")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_3 = [location for location in cluster_3 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_3.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_3_without_start = [location for location in cluster_3 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_3_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance3 = min(distances)
    min_route3 = routes[distances.index(min_distance3)]

# %%
last_state_3 = min_route3[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_3 = clusters[3]
cluster_12 = clusters[12]  # Assuming clusters[2] is defined

last_state_3 = min_route3[-1]
start_location_name = last_state_3['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_3 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_3")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_12 = [location for location in cluster_12 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_12.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_12_without_start = [location for location in cluster_12 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_12_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance12 = min(distances)
    min_route12 = routes[distances.index(min_distance12)]


# %%
last_state_12 = min_route12[-1]

# %%
import itertools
from geopy.distance import geodesic

cluster_12 = clusters[12]
cluster_1 = clusters[1]  # Assuming clusters[2] is defined

last_state_12 = min_route12[-1]
start_location_name = last_state_12['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_12 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_12")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_1 = [location for location in cluster_1 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_1.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_1_without_start = [location for location in cluster_1 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_1_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance1 = min(distances)
    min_route1 = routes[distances.index(min_distance1)]

# %%
last_state_1 = min_route1[-1]


# %%
import itertools
from geopy.distance import geodesic

cluster_1 = clusters[1]
cluster_9 = clusters[9]  # Assuming clusters[2] is defined

last_state_1 = min_route1[-1]
start_location_name = last_state_1['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_1 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_1")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_9 = [location for location in cluster_9 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_9.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_9_without_start = [location for location in cluster_9 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_9_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance9 = min(distances)
    min_route9 = routes[distances.index(min_distance9)]

# %%
last_state_9 = min_route9[-1]
# %%
import itertools
from geopy.distance import geodesic

cluster_9 = clusters[9]
cluster_5 = clusters[5]  # Assuming clusters[2] is defined

last_state_9 = min_route9[-1]
start_location_name = last_state_9['name']

# Pull start_location from cluster_2
start_location = next((d for d in cluster_9 if d["name"] == start_location_name), None)

if start_location is None:
    print(f"No location named {start_location_name} found in cluster_9")
else:
    # Remove start_location from cluster_0 if it's already there
    cluster_5 = [location for location in cluster_5 if location["name"] != start_location_name]

    # Add start_location to cluster_0
    cluster_5.append(start_location)

    # Create a copy of cluster_0 without the start_location for generating the permutations
    cluster_5_without_start = [location for location in cluster_5 if location["name"] != start_location_name]

    routes = list(itertools.permutations(cluster_5_without_start))
    routes = [(start_location,) + route for route in routes]

    distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            location1 = route[i]
            location2 = route[i + 1]
            distance = geodesic((location1['latitude'], location1['longitude']),
                                (location2['latitude'], location2['longitude'])).miles
            total_distance += distance
        distances.append(total_distance)

    # Find the shortest route
    min_distance5 = min(distances)
    min_route5 = routes[distances.index(min_distance5)]
# %%
last_state_5 = min_route5[-1]

# %%
import itertools
from geopy.distance import geodesic, distance

cluster_5 = clusters[5]
cluster_13 = clusters[13]  # Assuming clusters[2] is defined

last_state_5 = min_route5[-1]
start_location_name = last_state_5['name']

end_location_name = cluster_13[1]
end_location_name = end_location_name['name']
# Pull start_location from cluster_2
start_location = next((d for d in cluster_5 if d["name"] == start_location_name), None)
end_location = next((d for d in cluster_13 if d["name"] == end_location_name), None)
if start_location is None:
    print(f"No location named {start_location_name} found in cluster_13")
else:
    cluster_13 = [location for location in cluster_13 if location["name"] != start_location_name]
cluster_13.append(start_location)
cluster_13_without_start = [location for location in cluster_13 if location["name"] != start_location_name]

# Find the location object for DC in cluster_13
dc_location = next((d for d in cluster_13 if d["name"] == "DC"), None)

routes = list(itertools.permutations(cluster_13_without_start))
routes = [(start_location,) + route for route in routes]

# Create a new list of routes that end with dc_location
routes = [route + (dc_location,) for route in routes]

if end_location is None:
    print(f"No location named {end_location_name} found in cluster_13")
else:
    cluster_13 = [location for location in cluster_13 if location["name"] != end_location_name]
    cluster_13.append(end_location)
    cluster_13_without_end = [location for location in cluster_13 if location["name"] != end_location_name]
    routes = list(itertools.permutations(cluster_13_without_end))
    routes = [route + (end_location,) for route in routes]

distances = []
for route in routes:
    total_distance = 0
    for i in range(len(route) - 1):
        location1 = route[i]
        location2 = route[i + 1]
        distance = geodesic((location1['latitude'], location1['longitude']),
                            (location2['latitude'], location2['longitude'])).miles
        total_distance += distance
    distances.append(total_distance)

# Find the shortest route
min_distance13 = min(distances)
min_route13 = routes[distances.index(min_distance13)]

# %%
total_distance = (
            min_distance0 + min_distance1 + min_distance2 + min_distance3 + min_distance4 + min_distance5 + min_distance6 + min_distance7 + min_distance8 + min_distance9 + min_distance10 + min_distance11 + min_distance12 + min_distance13)
print(
    f"The Politician will have to travel a total of {total_distance:.2f} miles to visit every state capital while traveling from Iowa to DC.")
print(f"Cluster 1{[location['name'] for location in min_route7]}")
print(f"Cluster 2{[location['name'] for location in min_route2]}")
print(f"Cluster 3{[location['name'] for location in min_route6]}")
print(f"Cluster 4{[location['name'] for location in min_route8]}")
print(f"Cluster 5{[location['name'] for location in min_route0]}")
print(f"Cluster 6{[location['name'] for location in min_route11]}")
print(f"Cluster 7{[location['name'] for location in min_route4]}")
print(f"Cluster 8{[location['name'] for location in min_route10]}")
print(f"Cluster 9{[location['name'] for location in min_route3]}")
print(f"Cluster 10{[location['name'] for location in min_route12]}")
print(f"Cluster 11{[location['name'] for location in min_route1]}")
print(f"Cluster 12{[location['name'] for location in min_route9]}")
print(f"Cluster 13{[location['name'] for location in min_route5]}")
print(f"Cluster 14{[location['name'] for location in min_route13]}")
# %%
