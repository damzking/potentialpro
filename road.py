import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#road=pd.read_csv('region_traffic_by_road_type.csv')
vehicle=pd.read_csv('region_traffic_by_vehicle_type.csv')
vehicle.dropna()
corrs = vehicle.corr()
sns.heatmap(corrs, xticklabels=corrs.columns, yticklabels=corrs.columns, vmin=-1, center=0, vmax=1, cmap='PuOr', annot=True)
plt.show()
print(vehicle.head)
print(vehicle['Region_name'].value_counts())
#vehicle['Region_name']=pd.get_dummies(vehicle.Region_name)
region_dummies=pd.get_dummies(vehicle['Region_name'],prefix='region2')
print(region_dummies)
vehicle = pd.concat([vehicle, region_dummies], axis=1)
#vehicle.groupby("Region_name").agg(meanrating="pedal_cycles","mean")
print(vehicle.groupby("Region_name").agg(meanrating=("pedal_cycles", "mean")))
print(vehicle.groupby("Region_name").agg(meanrating=("all_motor_vehicles", "mean")))
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score


scaler = StandardScaler()


total_link_length_km = vehicle['total_link_length_km']
all_hgvs = vehicle['all_hgvs']
all_motor_vehicles = vehicle['all_motor_vehicles']
cars_and_taxis = vehicle['cars_and_taxis']
buses_and_coaches = vehicle['buses_and_coaches']
lgvs = vehicle['lgvs']
pedal_cycles = vehicle['pedal_cycles']

# Reshape the columns to ensure they have the correct shape for scaling
reshaped_total_link_length_km = np.array(total_link_length_km).reshape(-1, 1)
reshaped_all_hgvs = np.array(all_hgvs).reshape(-1, 1)
reshaped_all_motor_vehicles = np.array(all_motor_vehicles).reshape(-1, 1)
reshaped_cars_and_taxis = np.array(cars_and_taxis).reshape(-1, 1)
reshaped_buses_and_coaches = np.array(buses_and_coaches).reshape(-1, 1)
reshaped_lgvs = np.array(lgvs).reshape(-1, 1)
reshaped_pedal_cycles = np.array(pedal_cycles).reshape(-1, 1)

# Scale the columns using the scaler object
total_link_length_km_scaled = scaler.fit_transform(reshaped_total_link_length_km)
all_hgvs_scaled = scaler.fit_transform(reshaped_all_hgvs)
all_motor_vehicles_scaled = scaler.fit_transform(reshaped_all_motor_vehicles)
cars_and_taxis_scaled = scaler.fit_transform(reshaped_cars_and_taxis)
buses_and_coaches_scaled = scaler.fit_transform(reshaped_buses_and_coaches)
lgvs_scaled = scaler.fit_transform(reshaped_lgvs)
pedal_cycles_scaled = scaler.fit_transform(reshaped_pedal_cycles)

# Update the DataFrame with the scaled columns
vehicle['total_link_length_km'] = total_link_length_km_scaled
vehicle['all_hgvs'] = all_hgvs_scaled
vehicle['all_motor_vehicles'] = all_motor_vehicles_scaled
vehicle['cars_and_taxis'] = cars_and_taxis_scaled
vehicle['buses_and_coaches'] = buses_and_coaches_scaled
vehicle['lgvs'] = lgvs_scaled
vehicle['pedal_cycles'] = pedal_cycles_scaled

print(vehicle['pedal_cycles'])



#Exploratory Data Analysis on Numercal Variables using Histograms and Barcharts
#sns.displot(data=vehicle, x="all_hgvs", kind="hist", binwidth = 2);
#sns.displot(data=vehicle, x="total_link_length_km", kind="hist", binwidth = 500);
#sns.displot(data=vehicle, x="buses_and_coaches",hue="Region_name", kind="hist", binwidth = 2, multiple = "stack");
#sns.displot(data=vehicle, x="all_hgvs",hue="Region_name", kind="hist", binwidth = 50000, multiple = "stack");
#sns.displot(data=vehicle, x="buses_and_coaches",hue="Region_name", kind="hist", binwidth = 2, multiple = "stack");
plt.plot()
plt.show

#print(vehicle.columns)

#plt.show()
#Exploratory Data Analysis on Numerical Variables Using Boxplots
#sns.boxplot(data=vehicle,x='Region_name',y='all_hgvs')
#sns.boxplot(data=vehicle,x='Region_name',y='all_motor_vehicles')
#sns.boxplot(data=vehicle,x='Region_name',y='total_link_length_miles')
#sns.boxplot(data=vehicle,x='Region_name',y='cars_and_taxis')
#sns.boxplot(data=vehicle,x='Region_name',y='buses_and_coaches')
#sns.boxplot(data=vehicle,x='Region_name',y='lgvs')
#sns.boxplot(data=vehicle,x='Region_name',y='pedal_cycles')//
#sns.boxplot(data=vehicle, palette='rocket')
plt.plot()
plt.show()
#print(vehicle.groupby("Region_name").agg(meanrating=("pedal_cycles", "mean")))
#print(vehicle.groupby("Region_name").agg(meanrating=("all_motor_vehicles", "mean")))
#print(vehicle.groupby("Region_name").agg(meanratingforcarsandtaxi=("cars_and_taxis", "mean")))
#print(vehicle.groupby("Region_name").agg(meanratingforcarsandtaxi=("buses_and_coaches", "mean")))
# Using Min-max scaler on the numerical variables 
#Exploratory Data Analysis on Categorical Variables
#sns.countplot(data=vehicle,x='Region_name')
#print(vehicle.groupby("Region_name")["ons_code"].value_counts)
#vehicle['year'].astype(object)
#vehicle.groupby('year')['income'].sum()
#Seperating Target and features


y= vehicle['total_link_length_km']
X= vehicle.drop(['region_id','Region_name','ons_code','year','total_link_length_miles','total_link_length_km'],axis=1)
print(X.columns)

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
# Assuming you have already imported necessary libraries and instantiated the scaler object
model1 = sm.OLS.from_formula('total_link_length_km ~ pedal_cycles',data=vehicle).fit()
model2 = sm.OLS.from_formula('total_link_length_km ~ buses_and_coaches+ cars_and_taxis+pedal_cycles',data=vehicle).fit()
print(model1.params)
print(model2.params)
# Extract the columns from the DataFrame
#Using L1 regularization or lasso 
model = LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1000)
model.fit(X_train,y_train)
PRED_TRAIN = model.predict(X_train)
PRED_TEST = model.predict(X_test)
print(PRED_TRAIN,PRED_TEST)
print("Linear regression_test",mean_squared_error(y_test,PRED_TEST))
print("linear regression_train",mean_squared_error(y_train,PRED_TRAIN))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.05)
lasso.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
pred_train = lasso.predict(X_train)
pred_test = lasso.predict(X_test)
training_mse = mean_squared_error(y_train, pred_train)
test_mse = mean_squared_error(y_test, pred_test)
print('Training Error_lasso:',  training_mse)
print('Test Error_lasso:', test_mse)

#using L2 regularization or ridge
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=0.05)
ridge.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
pred_train_2 = ridge.predict(X_train)
pred_test_2 = ridge.predict(X_test)
training_mse_2 = mean_squared_error(y_train, pred_train_2)
test_mse_2 = mean_squared_error(y_test, pred_test_2)
print('Training Error_ridge:',  training_mse_2)
print('Test Error_ridge:', test_mse_2)


#Hyperparameter Tuning of L1 and L2
from sklearn.model_selection import GridSearchCV
param_grid = { 'alpha':[0.001,0.01,0.1,1.0,10.0]}

lasso = Lasso()

# Perform grid search with cross-validation
grid = GridSearchCV(lasso, param_grid, cv=5)
grid.fit(X_train, y_train)

# Predict on train and test data
y_pred_train = grid.predict(X_train)
y_pred_test = grid.predict(X_test)

print("r2 score_lasso_ :",r2_score(y_test,y_pred_test))
print("r2 score_lasso_train:",r2_score(y_train,y_pred_train))
# Calculate train and test MSE
train_mse = np.mean((y_pred_train - y_train)**2)
test_mse = np.mean((y_pred_test - y_test)**2)


# Print the best hyperparameters
print("train_mse:", train_mse)
print("test_mse:", test_mse)
print("Best hyperparameters:", grid.best_params_)



# Create a linear  regression model
ridge = Ridge()

# Perform grid search with cross-validation
grid = GridSearchCV(ridge, param_grid, cv=5)
grid.fit(X_train, y_train)

# Predict on train and test data
y_pred_train_r = grid.predict(X_train)
y_pred_test_r = grid.predict(X_test)

print("r2 score_ridge_ :",r2_score(y_test,y_pred_test_r))
print("r2 score_ridge_train:",r2_score(y_train,y_pred_train_r))


# Calculate train and test MSE
train_mse = np.mean((y_pred_train - y_train)**2)
test_mse = np.mean((y_pred_test - y_test)**2)


# Print the best hyperparameters
print("train_mse:", train_mse)
print("test_mse:", test_mse)
print("Best hyperparameters:", grid.best_params_)



#USING UNSUPERVISED LEARNING TO CLASSIFY THE GROUPS AVAILABLE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
#X_test = pd.DataFrame(X_test, columns=['pedal_cycles', 'all_motor_vehicles', 'cars_and_taxis', 'buses_and_coaches', 'lgvs', 'all_hgvs'])

featuresToCluster = vehicle[['pedal_cycles', 'all_motor_vehicles', 'cars_and_taxis', 'buses_and_coaches', 'lgvs', 'all_hgvs']]
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(featuresToCluster)
    inertias.append(kmeans.inertia_)

# Visualize the elbow plot
plt.plot(range(1, 11), inertias)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
scaler = StandardScaler()
featuresToCluster_scaled = scaler.fit_transform(featuresToCluster)

from sklearn.cluster import KMeans

# Instantiate and fit the KMeans model
classifier = KMeans(n_clusters=4)
classifier.fit(featuresToCluster_scaled)

# Predict cluster labels
cluster_labels = classifier.predict(featuresToCluster_scaled)
vehicle['clusters'] = cluster_labels

# Print the cluster labels for the first few records
print(vehicle['clusters'].head())

# Group the data by clusters and calculate the mean values
grouped = vehicle.groupby('clusters')[['pedal_cycles', 'all_motor_vehicles', 'cars_and_taxis', 'buses_and_coaches', 'lgvs', 'all_hgvs']].mean()

# Print the results
print(grouped)

#scaler = StandardScaler()
#featuresToCluster_scaled = scaler.fit_transform(featuresToCluster)
#classifier = KMeans(n_clusters=3)
#classifier.fit(featuresToCluster_scaled.values)
#cluster_labels = classifier.predict(featuresToCluster_scaled)
#vehicle['clusters'] = cluster_labels
#print(vehicle.clusters.head())

vehicle = pd.concat([vehicle, pd.Series(cluster_labels, name='clusters')], axis=1)

#print(vehicle.columns)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# assume that X is your feature matrix and y is your cluster labels
X_embedded = TSNE(n_components=3).fit_transform(featuresToCluster)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y)
plt.show()


import matplotlib.pyplot as plt

# Plot the clusters using scatter plots
plt.scatter(featuresToCluster_scaled[cluster_labels==0, 0], featuresToCluster_scaled[cluster_labels==0, 1], s=50, color='red', label='Cluster 1')
plt.scatter(featuresToCluster_scaled[cluster_labels==1, 0], featuresToCluster_scaled[cluster_labels==1, 1], s=50, color='blue', label='Cluster 2')
plt.scatter(featuresToCluster_scaled[cluster_labels==2, 0], featuresToCluster_scaled[cluster_labels==2, 1], s=50, color='green', label='Cluster 3')
plt.scatter(classifier.cluster_centers_[:, 0], classifier.cluster_centers_[:, 1], s=100, color='black', label='Centroids')
plt.title('Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()




