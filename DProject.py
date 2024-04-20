import os
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.layers import Dense
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
sns.set_theme(style="dark")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, __all__
from sklearn import tree
import statsmodels.api as sm
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

#Q.3
dframe1 = pd.read_csv('input1_df.csv')

print("The first lines on the database:")
print(dframe1.head())
print()
print(dframe1.info())
print()
print("Descriptive statistics Of the database: ")
print(dframe1.describe())
print()

#Q.4
duplicates = dframe1[dframe1.duplicated('Title')]
print()
print(duplicates)
dframe1 = dframe1.drop_duplicates()
dframe1.reset_index(drop=True, inplace=True)
dframe1.to_csv("Dup.csv",index=False)


overall_mode1 = dframe1['Age'].mode().iloc[0]
# print(overall_mode1)
dframe1.loc[:, 'Age'] = dframe1['Age'].fillna(overall_mode1)
# dframe1.to_csv('mode1.csv',index=False)


overall_mean1 = dframe1["IMDb"].mean()
# print(overall_mean1)
dframe1["IMDb"] = dframe1["IMDb"].fillna(overall_mean1)
# dframe1.to_csv('AVG1.csv',index=False)


overall_mode2 = dframe1['date_added to rating'].mode().iloc[0]
# print(overall_mode2)
dframe1.loc[:, 'date_added to rating'] = dframe1['date_added to rating'].fillna(overall_mode2)
# dframe1.to_csv('mode2.csv',index=False)


overall_mode3 = dframe1['type'].mode().iloc[0]
# print(overall_mode3)
dframe1.loc[:, 'type'] = dframe1['type'].fillna(overall_mode3)
# dframe1.to_csv('mode3.csv',index=False)


overall_mode4 = dframe1['country'].mode().iloc[0]
# print(overall_mode4)
dframe1.loc[:, 'country'] = dframe1['country'].fillna(overall_mode4)
# dframe1.to_csv('mode4.csv',index=False)


overall_mode5 = dframe1['duration'].mode().iloc[0]
# print(overall_mode5)
dframe1.loc[:, 'duration'] = dframe1['duration'].fillna(overall_mode5)
# dframe1.to_csv('mode5.csv',index=False)

# print(dframe1.info())


#Q.5
print()
pv1_df = pd.pivot_table(dframe1, index='country',columns='type', values='IMDb', aggfunc='mean')
print("pv1_df")
print(pv1_df)

print()
pv2_df = pd.pivot_table(dframe1, index='type',columns='Age', values='Rotten Tomatoes', aggfunc='max',fill_value=0)
print("pv2_df")
print(pv2_df)

print()
pv3_df = pd.pivot_table(dframe1, index='release year',columns='type', values='Rotten Tomatoes', aggfunc='count',fill_value=0)
print("pv3_df")
print(pv3_df)


#Q.6
#pv1
#1
pv1_df.plot(kind='bar', figsize=(10, 6))
plt.title('Mean IMDb Ratings by Country and Type')
plt.xlabel('Country')
plt.ylabel('Mean IMDb Rating')
plt.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#2
plt.figure(figsize=(10, 6))
for column in pv1_df.columns:
    plt.plot(pv1_df.index, pv1_df[column], marker='o', label=column)

plt.title('Mean IMDb Ratings by Country and Type')
plt.xlabel('Country')
plt.ylabel('Mean IMDb Rating')
plt.legend(title='Movie Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#pv2
#1
plt.figure(figsize=(10, 6))
sns.heatmap(pv2_df, cmap='viridis')
plt.title('Max Heatmap of Rotten Tomatoes by Type and Age')
plt.show()

#2
melted_df = pv2_df.reset_index().melt(id_vars='type', var_name='Age', value_name='Rotten Tomatoes')
melted_df.dropna(inplace=True)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Rotten Tomatoes', hue='type', data=melted_df, s=100)
plt.title('Max Scatter Plot of Rotten Tomatoes by Age and Type')
plt.xlabel('Age')
plt.ylabel('Rotten Tomatoes')
plt.show()


#pv3
#1
fig = px.area(pv3_df, x=pv3_df.index, y=['Movie', 'TV Show'],
title='Rotten Tomatoes Count Over Release Years (Area Chart)',
labels={'value': 'Rotten Tomatoes Count', 'release year': 'Release Year'})
fig.show()

#2
plt.figure(figsize=(10, 6))
sns.kdeplot(data=pv3_df, x='Movie', y='TV Show', fill=True)
plt.title('Density Contour Plot of Rotten Tomatoes Count by Type')
plt.xlabel('Movie Count')
plt.ylabel('TV Show Count')
plt.show()


#Q.7
max = dframe1["IMDb"].max()
print(max)
result = []
for number in dframe1["IMDb"]:
    result.append(abs(number)/max)
dframe1["IMDb_norm"] = result
# dframe1.to_csv("norm.csv",index=False)


X = dframe1.iloc[:, [2,14]].values
np.set_printoptions(suppress=True)
print(X)

wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)

y_kmeans = kmeans.fit_predict(X)
# np.set_printoptions(threshold=np.inf)
print(y_kmeans)
print()

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c='red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c='blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c='green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=100, c='cyan', label = 'Cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[: , 1], s=200, c= 'black',label= 'Centroids')

plt.title('Clusters')
plt.legend()
plt.show()


#Q.8
x = dframe1['release year']
y = dframe1['IMDb']

a,b = np.polyfit(x,y,deg = 1)
y_est = a*x+b
y_err = x.std() * np.sqrt(1/len(x)+ (x-x.mean()) ** 2/np.sum((x-x.mean())**2))

fig, ax = plt.subplots()
ax.plot(x,y_est,'-')
ax.fill_between(x,y_est - y_err, y_est + y_err, alpha = 0.2)
ax.plot(x,y,'o', color ='tab:brown')
fig.show()
plt.waitforbuttonpress()


x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

significance = model.pvalues['release year'] < 0.05
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj

print("Significance:", "Yes" if significance else "No")
print("R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r_squared)


#Q.9
#gini
features_names = ['IMDb','Rotten Tomatoes']

max_depth = 4
model_gain_name = 'gini'

x_data = dframe1[features_names]
y_data = dframe1['Prime Video']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion=model_gain_name, max_depth=max_depth)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("\n"+model_gain_name+" - Model Performance:")
print("Accuracy = ", accuracy_score(y_test,y_pred))
print("F1_score = ", metrics.f1_score(y_test, y_pred))
print("Recall_score = ", metrics.recall_score(y_test,y_pred))
print("Precision = ", metrics.precision_score(y_test,y_pred))

confusion_m = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion_m)

preds_prob = clf.predict_proba(x_test)[:, 1]
print('ROC AUC', metrics.roc_auc_score(y_test, preds_prob))

text_representation_gini = tree.export_text(clf)
#print(text_representation_gini)
with open("decision_tree_gini.log", "w") as fout:
    fout.write(text_representation_gini)

fig = plt.figure(figsize=(24,10))
_ = tree.plot_tree(clf,
                   feature_names=(['IMDb','Rotten Tomatoes']),
                   class_names=("Not Exist", 'Exist'),
                   rounded=True,
                   filled=True)
fig.savefig("decision_tree_gini.png")
plt.show()


#entropy
features_names = ['IMDb','Rotten Tomatoes']

max_depth = 4
model_gain_name = 'entropy'

x_data = dframe1[features_names]
y_data = dframe1['Prime Video']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion=model_gain_name, max_depth=max_depth)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("\n"+model_gain_name+" - Model Performance:")
print("Accuracy = ", accuracy_score(y_test,y_pred))
print("F1_score = ", metrics.f1_score(y_test, y_pred))
print("Recall_score = ", metrics.recall_score(y_test,y_pred))
print("Precision = ", metrics.precision_score(y_test,y_pred))

confusion_m = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(confusion_m)

preds_prob = clf.predict_proba(x_test)[:, 1]
print('ROC AUC', metrics.roc_auc_score(y_test, preds_prob))

text_representation_entropy = tree.export_text(clf)
#print(text_representation_entropy)
with open("decision_tree_entropy.log", "w") as fout:
    fout.write(text_representation_entropy)

fig = plt.figure(figsize=(24,10))
_ = tree.plot_tree(clf,
                   feature_names=(['IMDb','Rotten Tomatoes']),
                   class_names=("Not Exist", 'Exist'),
                   rounded=True,
                   filled=True)
fig.savefig("decision_tree_entropy.png")
plt.show()

print()


#Q.10
features_names = ['IMDb','Rotten Tomatoes']

X = dframe1[features_names]
y = dframe1['Prime Video']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Max: ',X_train.max())
print('Min: ', X_train.min())

s_model = Sequential()

# input + hidden layer
s_model.add(Dense(32, input_shape=(2,), activation='relu'))

# output layer
s_model.add(Dense(1, activation='sigmoid'))

s_model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=[tf.keras.metrics.AUC()])

s_model.fit(x=X_train,
          y=y_train,
          epochs=50,
          verbose = 2,
          batch_size=256,
          validation_data=(X_test, y_test))

s_losses = pd.DataFrame(s_model.history.history)

plt.figure(figsize=(7.5,2.5))
sns.lineplot(data=s_losses,lw=3)
plt.xlabel('Epochs')
plt.ylabel('')
plt.title('Training Loss per Epoch')
sns.despine()

s_model.predict(X_test)
s_y_pred = (s_model.predict(X_test) >= 0.4)[:,0]

confusion_matrix = confusion_matrix(y_test, s_y_pred)
plt.figure(figsize = (12,8))
sns.set(font_scale=1.1)
ax = sns.heatmap(confusion_matrix, cmap='Reds', annot=True, fmt='d', square=True,
                 xticklabels=['Predicted Negative', 'Predicted Positive'],
                 yticklabels=['Actual Negative', 'Actual Positive'])

ax.invert_yaxis()
ax.invert_xaxis()
plt.show()

print("Accuracy = ", accuracy_score(y_test,s_y_pred))
print("F1_score = ", metrics.f1_score(y_test, s_y_pred))
print("Recall_score = ", metrics.recall_score(y_test,s_y_pred))
print("Precision = ", metrics.precision_score(y_test,s_y_pred))

