import pandas as pd
# from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.cm as cm
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

from geo_codare import geo_codare_db

test = pd.read_csv("eval.csv")
test = test.loc[test['Judet'] == 'Bucuresti']

geo_codare_db(test)

X_test_coords = pd.DataFrame(test, columns=['Latitudine','Longitudine'])
X_test_features = pd.DataFrame(test, columns=['Suprafata construita desfasurata', 'Suprafata teren', 'An'])
Y_test = pd.DataFrame(test, columns=['Pret'])

def sanitie_year(y):
    return y.replace('<', '').replace('>', '')

def prepare_features_for_model(feature_df):
    # feature_df['Finisaje'] = feature_df['Finisaje'].replace({
    #     "inferior": 0,
    #     "Inferior": 0,
    #     "standard": 1,
    #     "Standard": 1,
    #     "superior": 2,
    #     "Superior": 2,
    # })
    # feature_df.fillna('nan', inplace=True)
    # feature_df['Tip Artera'] = feature_df['Tip Artera'].replace({
    #     "Strada": 0,
    #     "Alee": 1,
    #     "Bulevard": 2,
    #     "Drum": 3,
    #     "Sosea": 4,
    #     "nan": 5,
    # })
    feature_df['An'] = feature_df['An'].apply(sanitie_year)

def predict(X_train, Y_train, X_test):
    Y_train = Y_train.values.ravel()

    svclassifier = SVC(kernel='linear', C=1.3)
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    svclassifier.fit(X_train, Y_train)
    rf.fit(X_train, Y_train)
    knn.fit(X_train, Y_train)

    svm_pred = svclassifier.predict(X_test)
    rf_pred = rf.predict(X_test)
    knn_pred = knn.predict(X_test)
    knn_neighbours = knn.kneighbors(X_test, return_distance=False)

    return (svm_pred, rf_pred, knn_pred, knn_neighbours
    )

def compute(train_file, Bucuresti_only=True):
    data = pd.read_csv(train_file)
    data = data.loc[data['Judet'] == 'Bucuresti']
    X = pd.DataFrame(data, columns=['Latitudine','Longitudine'])

    NUM_CLUSTERS = int(len(X.index)/15);

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=300, n_init=10, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

    clustered_properties = kmeans.fit_predict(X)
    knn.fit(X, clustered_properties)

    clusters_predicted_for_test_data = knn.predict(X_test_coords)

    X["cluster"] = clustered_properties
    prepare_features_for_model(X_test_features)
    result = pd.DataFrame(columns=['predict1 (SVM)',
    'predict2 (Random Forest)',
    'predict3 (KNN)',
    'mean_predict',
    'ground truth',
    'knn_neighbours',
    'collegues',
    'X'
    ])
    i = 0
    for c in clusters_predicted_for_test_data:
        cluster_collegues = X.loc[X['cluster']==c]
        indexes = cluster_collegues.index.values.astype(int)
        cluster_collegues_all_features = data.loc[indexes]

        cluster_collegues_relevant_features = pd.DataFrame(cluster_collegues_all_features,
            columns=['Suprafata', 'Suprafata2', 'An'])
        cluster_collegues_prices = pd.DataFrame(cluster_collegues_all_features, columns=['Pret'])

        prepare_features_for_model(cluster_collegues_relevant_features)
        print(i)
        # print(len(cluster_collegues_all_features))
        # print(cluster_collegues_relevant_features, cluster_collegues_prices, X_test_features.iloc[[i]])
        (svm, rf, knn, knn_neighbours) = predict(cluster_collegues_relevant_features, cluster_collegues_prices, X_test_features.iloc[[i]])

        result.loc[i] = ("{:,}".format(int(svm[0])),
        "{:,}".format(int(rf[0])),
        "{:,}".format(int(knn[0])),
        "{:,}".format(int((knn[0] + rf[0])/2)),
        Y_test.iloc[i]['Pret'],
        knn_neighbours[0],
        cluster_collegues_all_features.to_dict(orient='records'),
        test.loc[int(X_test_features.iloc[i].name)].to_json())
        # print(Y_test.iloc[i]['Pret'])
        i = i + 1

    return result
