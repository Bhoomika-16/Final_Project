import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import pandas as pd
import numpy as np
import matplotlib.lines as lines
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition, preprocessing, model_selection
from keras import models, layers, activations, losses, optimizers, metrics
from keras.callbacks import EarlyStopping
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
# from test import normal_events_df, abnormal_events_df, cut_off
from tkinter import *


def if_alg1(dtfr, cf):
    arr1 = np.array(dtfr)
    n_nedf = len(arr1)
    outliers_fraction = cf
    n_outliers = int(outliers_fraction * n_nedf)
    X = arr1[:, [0, 1]]
    rng = np.random.RandomState(42)
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    iForest = IsolationForest(n_estimators=20, verbose=2)
    iForest.fit(X)
    pred = iForest.predict(X)
    plt1.scatter(X[:, 0], X[:, 1], c=pred, cmap='RdBu')
    plt1.savefig('a1.png')
    # plt1.show()
    pred_scores = -1 * iForest.score_samples(X)
    plt1.scatter(X[:, 0], X[:, 1], c=pred_scores, cmap='RdBu')
    plt1.colorbar(label='Simplified Anomaly Score')
    plt1.savefig('a2.png')
    # plt1.show()
    col3, col4 = st.columns(2)

    with col3:
        st.header("Normal Event Anomaly")
        st.image("a1.png")
        plt1.close()
    with col4:
        st.header("Simplified Normal Event Anomaly Score")
        st.image("a2.png")
        plt1.close()


def if_alg2(abed, cf):
    arr1 = np.array(abed)
    n_nedf = len(arr1)
    outliers_fraction = cf
    n_outliers = int(outliers_fraction * n_nedf)
    X = arr1[:, [0, 1]]
    rng = np.random.RandomState(42)
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    iForest = IsolationForest(n_estimators=20, verbose=2)
    iForest.fit(X)
    pred = iForest.predict(X)
    plt1.scatter(X[:, 0], X[:, 1], c=pred, cmap='RdBu')
    plt1.savefig('b1.png')
    # plt2.show()
    pred_scores = -1 * iForest.score_samples(X)
    plt1.scatter(X[:, 0], X[:, 1], c=pred_scores, cmap='RdBu')
    plt1.colorbar(label='Simplified Anomaly Score')
    plt1.savefig('b2.png')
    # plt2.show()
    col5, col6 = st.columns(2)

    with col5:
        st.header("Abnormal Event Anomaly")
        st.image("b1.png")
        #plt2.close()

    with col6:
        st.header("Simplified Abnormal Event Anomaly Score")
        st.image("b2.png")
        plt1.close()

# if_alg1(normal_events_df, cut_off)

# if_alg2(abnormal_events_df, cut_off)
