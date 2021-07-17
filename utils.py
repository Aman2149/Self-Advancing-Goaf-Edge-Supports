# Run requirements.txt by --> pip install -r requrirements.txt in the terminal.

# Loading the packages
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm
from dbscan1d.core import DBSCAN1D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors


# Cleaning the text file
def split_clean(string):
    return string.replace(',', '').replace('\n', '').split(' ')


# Loading and converting the text file to CSV format
def convert(name):
    if name.split('.')[-1] == 'txt':
        txt = open(name, 'r')
        lines = txt.readlines()[94:-1]
        df = pd.DataFrame(list(map(split_clean, lines)))
        df.columns = ['Date', 'Timestamp', 'Leg-1', 'Leg-2', 'Conv']

    return df


# Convolving the dataframe
def ConvResamp(df, timeframe):
    dfNew = df.copy()

    df.loc[:, 'DateTime'] = pd.to_datetime(df.Date.astype(str) + ' ' + df.Timestamp.astype(str), errors='coerce')
    df.set_index('DateTime', append=False, inplace=True)

    conv = {
        'Timestamp': 'last',
        'Leg-1': 'last',
        'Leg-2': 'last',
        'Conv': 'last'
    }

    dfNew = df.resample(timeframe).agg(conv)
    dfNew.dropna(inplace=True)
    dfNew.columns = ['Timestamp', 'Leg-1', 'Leg-2', 'Conv']
    dfNew[['Leg-1', 'Leg-2', 'Conv']] = dfNew[['Leg-1', 'Leg-2', 'Conv']].astype(float)
    return dfNew


# Finding peaks of Leg-1 and Leg-2 load on SAGES
def finding_peaks(df, feature):
    peaks, pts = find_peaks(ConvResamp(df, '1min')[feature], distance=1, height=df[feature].astype('float').mean(),
                            width=1, prominence=(0.5, 25))
    return peaks, pts


# Smoothing out the soft edges
def smoothing_conv(df):
    peaks, pts = find_peaks(ConvResamp(df, '1min')['Conv'], distance=1, height=df['Conv'].astype('float').mean(),
                            width=7)
    return peaks, pts


# Plotting the XR5 data file
def plotting_peaks(df):
    plt.figure(figsize=(10, 10))

    peaks1, pts1 = finding_peaks(df, 'Leg-1')
    plt.plot(ConvResamp(df, '1min')['Leg-1'].values)
    plt.plot(peaks1, np.array(ConvResamp(df, '1min')['Leg-1'])[peaks1], "x")

    peaks2, pts2 = finding_peaks(df, 'Leg-2')
    plt.plot(ConvResamp(df, '1min')['Leg-2'].values)
    plt.plot(peaks2, np.array(ConvResamp(df, '1min')['Leg-2'])[peaks2], "x")

    peaks3, pts3 = smoothing_conv(df)
    plt.plot(savgol_filter(ConvResamp(df, '1min')['Conv'].values, 5, 3))

    plt.show()


# Developing the AI model to predict the duration of yielding
# Plotting the clusters 
# Please modify the min_samples and eps according to convenience
def ai_model_cluster(df, eps=1000, min_samples=3):
    """
    :param df: Dataframe retained after converting text file to CSV format
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
                This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
                        This includes the point itself.
    :return: Data variation plots, DBSCAN Cluster plot, Duration of each cluster.
    """

    peaks, pts = finding_peaks(df, 'Leg-2')
    peaks1, pts1 = finding_peaks(df, 'Leg-1')

    plt.figure(figsize=(40, 20))
    plt.subplot(2, 1, 1)

    plt.plot(ConvResamp(df, '1min')['Leg-1'].values)
    plt.plot(peaks1, np.array(ConvResamp(df, '1min')['Leg-1'])[peaks1], "x")

    plt.plot(ConvResamp(df, '1min')['Leg-2'].values)
    plt.plot(peaks, np.array(ConvResamp(df, '1min')['Leg-2'])[peaks], "x")
    plt.plot(savgol_filter(ConvResamp(df, '1min')['Conv'].values, 5, 3))

    plt.subplot(2, 1, 2)
    db = DBSCAN1D(eps=eps, min_samples=min_samples)

    # Get labels for each point
    labels = db.fit_predict(peaks)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Plot result
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = ['yellow', 'blue', 'lightgreen', 'red', 'orange', 'purple', 'cyan', 'snow', 'deeppink']
    print(colors)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy = peaks.reshape(-1, 1)[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 0], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    plt.title('number of clusters: %d' % n_clusters_)
    plt.show()
    newdf = pd.DataFrame(
        {
            'Values': ConvResamp(df, '1min')['Leg-2'].iloc[peaks].values,
            'Cluster': db.labels_,
            'Datetime': ConvResamp(df, '1min')['Leg-2'].iloc[peaks].index,
            'Peaks': peaks
        }
    )
    newdf.set_index('Datetime', append=False, inplace=True)
    newdf = newdf[newdf.Cluster > -1]
    for i in set(newdf['Cluster'].values):
        print(newdf[newdf['Cluster'] == i].iloc[-1].name - newdf[newdf['Cluster'] == i].iloc[0].name)
