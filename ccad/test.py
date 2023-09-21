import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.lines as lines
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition, preprocessing, model_selection
from keras import models, layers, activations, losses, optimizers, metrics
from keras.callbacks import EarlyStopping
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from PIL import Image
from ifa import if_alg1, if_alg2

st.set_page_config(layout='wide')
logo = Image.open("logo.png")
logo = logo.resize((250, 250))
logo1 = Image.open("vtu1.png")
logo1 = logo1.resize((250, 250))
c1, mid, c2, c3 = st.columns([1, 4, 20, 3])
with c1:
    st.image(logo, width=180)
with c2:
    st.title('DON BOSCO INSTITUTE OF TECHNOLOGY')
    st.header("Department of Computer science and Engineering")

with c3:
    st.image(logo1, width=180)

# col1, col2, col3 = st.columns([10, 40, 1])
# with col2:
#     st.header("Department of Computer science and Engineering")
#     st.subheader("PROJECT WORK PHASE-II-18CSP83")
st.markdown("<h1 style='text-align: center; color: blue'>AUTOENCODERS FOR ANOMALY DETECTION USING ISOLATION FOREST ALGORITHM</h1>",
            unsafe_allow_html=True)
# st.title(":blue[AUTOENCODERS FOR ANOMALY DETECTION USING ISOLATION FOREST ALGORITHM]")

with st.expander("ABOUT THE PROJECT"):
    st.write(""" The Autoencoders and Isolation Forest project is a powerful tool for detecting anomalies in large datasets. By leveraging the capabilities of autoencoders and isolation forest algorithms, this project offers an effective solution for identifying unusual patterns in data. 
    This approach is particularly useful in applications such as fraud detection, cybersecurity, and predictive maintenance, where the ability to identify outliers is crucial.""")
st.subheader("Add files to detect anomalies")

uploaded_file = st.file_uploader('choose a csv file')


# for uploaded_file in df:
#     df = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     st.write(df)

def ln(nedf, aedf, ps, co):
    mse_df = pd.concat([nedf, aedf])
    plot = sns.lineplot(x=mse_df.n, y=mse_df.mse, hue=mse_df.anomaly)

    line = lines.Line2D(
        xdata=np.arange(0, ps),
        ydata=np.full(ps, co),
        color='#CC2B5E',
        linewidth=1.5,
        linestyle='dashed')

    plot.add_artist(line)
    plt.title('Threshold: {threshold}'.format(threshold=round(co, 4)))
    plt.savefig('f1.png')


# def if_alg1(dtfr, cf):
#     arr1 = np.array(dtfr)
#     n_nedf = len(arr1)
#     outliers_fraction = cf
#     n_outliers = int(outliers_fraction * n_nedf)
#     X = arr1[:, [0, 1]]
#     rng = np.random.RandomState(42)
#     X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
#     iForest = IsolationForest(n_estimators=20, verbose=2)
#     iForest.fit(X)
#     pred = iForest.predict(X)
#     plt1.scatter(X[:, 0], X[:, 1], c=pred, cmap='RdBu')
#     plt1.savefig('a1.png')
#     pred_scores = -1 * iForest.score_samples(X)
#     plt1.scatter(X[:, 0], X[:, 1], c=pred_scores, cmap='RdBu')
#     plt1.colorbar(label='Simplified Anomaly Score')
#     plt1.savefig('a2.png')
#     col3, col4 = st.columns(2)
#
#     with col3:
#         st.header("Normal Event Anomaly")
#         st.image("a1.png")
#
#     with col4:
#         st.header("Simplified Normal Event Anomaly Score")
#         st.image("a2.png")
#
#
# def if_alg2(abed, cf):
#     arr1 = np.array(abed)
#     n_nedf = len(arr1)
#     outliers_fraction = cf
#     n_outliers = int(outliers_fraction * n_nedf)
#     X = arr1[:, [0, 1]]
#     rng = np.random.RandomState(42)
#     X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
#     iForest = IsolationForest(n_estimators=20, verbose=2)
#     iForest.fit(X)
#     pred = iForest.predict(X)
#     plt2.scatter(X[:, 0], X[:, 1], c=pred, cmap='RdBu')
#     plt2.savefig('b1.png')
#     pred_scores = -1 * iForest.score_samples(X)
#     plt2.scatter(X[:, 0], X[:, 1], c=pred_scores, cmap='RdBu')
#     plt2.colorbar(label='Simplified Anomaly Score')
#     plt2.savefig('b2.png')
#     col5, col6 = st.columns(2)
#
#     with col5:
#         st.header("Abnormal Event Anomaly")
#         st.image("b1.png")
#
#     with col6:
#         st.header("Simplified Abnormal Event Anomaly Score")
#         st.image("b2.png")


if uploaded_file is not None:
    # # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    #
    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)
    #
    # # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.write(df.describe())

    x = df[df.columns[1:30]].to_numpy()
    y = df[df.columns[30]].to_numpy()

    # prepare data
    df = pd.concat([pd.DataFrame(x), pd.DataFrame({'anomaly': y})], axis=1)
    print(df)
    normal_events = df[df['anomaly'] == 0]
    abnormal_events = df[df['anomaly'] == 1]

    normal_events = normal_events.loc[:, normal_events.columns != 'anomaly']
    abnormal_events = abnormal_events.loc[:, abnormal_events.columns != 'anomaly']
    print(normal_events)
    print(abnormal_events)

    # scaling
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df.drop('anomaly', 1))

    scaled_data = scaler.transform(normal_events)

    # 80% percent of dataset is designated to training
    train_data, test_data = model_selection.train_test_split(scaled_data, test_size=0.2)

    n_features = x.shape[1]
    print(n_features)
    # model
    encoder = models.Sequential(name='encoder')
    encoder.add(layer=layers.Dense(units=20, activation=activations.relu, input_shape=[n_features]))
    encoder.add(layers.Dropout(0.1))
    encoder.add(layer=layers.Dense(units=10, activation=activations.relu))
    encoder.add(layer=layers.Dense(units=5, activation=activations.relu))

    decoder = models.Sequential(name='decoder')
    decoder.add(layer=layers.Dense(units=10, activation=activations.relu, input_shape=[5]))
    decoder.add(layer=layers.Dense(units=20, activation=activations.relu))
    decoder.add(layers.Dropout(0.1))
    decoder.add(layer=layers.Dense(units=n_features, activation=activations.sigmoid))

    autoencoder = models.Sequential([encoder, decoder])

    autoencoder.compile(
        loss=losses.MSE,
        optimizer=optimizers.Adam(),
        metrics=[metrics.mean_squared_error])

    # train model
    es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, restore_best_weights=True)
    history = autoencoder.fit(x=train_data, y=train_data, epochs=5, verbose=1, validation_data=[test_data, test_data],
                              callbacks=[es])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('f0.png')
    # plt.close('f0.png')
    # st.image('f0.png',caption='Model Loss')

    autoencoder.save('autoencoder.h5')

    train_predicted_x = autoencoder.predict(x=train_data)
    train_events_mse = losses.mean_squared_error(train_data, train_predicted_x)
    cut_off = np.percentile(train_events_mse, 95)

    print('cut_off:', round(cut_off, 4))
    # cut_off: 0.0013734778214711686

    #####
    plot_samples = 10
    # normal event
    real_x = test_data[:plot_samples].reshape(plot_samples, n_features)
    predicted_x = autoencoder.predict(x=real_x)
    normal_events_mse = losses.mean_squared_error(real_x, predicted_x)
    normal_events_df = pd.DataFrame({
        'mse': normal_events_mse,
        'n': np.arange(0, plot_samples),
        'anomaly': np.zeros(plot_samples)})

    # abnormal event
    abnormal_x = scaler.transform(abnormal_events)[:plot_samples].reshape(plot_samples, n_features)
    predicted_x = autoencoder.predict(x=abnormal_x)
    abnormal_events_mse = losses.mean_squared_error(abnormal_x, predicted_x)
    abnormal_events_df = pd.DataFrame({
        'mse': abnormal_events_mse,
        'n': np.arange(0, plot_samples),
        'anomaly': np.ones(plot_samples)})

    ln(normal_events_df, abnormal_events_df, plot_samples, cut_off)
    # mse_df = pd.concat([normal_events_df, abnormal_events_df])
    # plot = sns.lineplot(x=mse_df.n, y=mse_df.mse, hue=mse_df.anomaly)
    #
    # line = lines.Line2D(
    #     xdata=np.arange(0, plot_samples),
    #     ydata=np.full(plot_samples, cut_off),
    #     color='#CC2B5E',
    #     linewidth=1.5,
    #     linestyle='dashed')
    #
    # plot.add_artist(line)
    # plt.title('Threshold: {threshold}'.format(threshold=round(cut_off, 4)))
    # plt.savefig('f1.png')
    # plt.close('f1.png')

    # with st.container():
    # st.columns(st.image('f0.png', caption='Model Loss'))
    # st.columns(st.image('f1.png', caption='Anomalies detected'))

    col1, col2 = st.columns(2)

    with col1:
        st.header("Model Loss")
        st.image("f0.png")

    with col2:
        st.header("Anomalies Detected")
        st.image("f1.png")
        # plt.close('f1.png')

    plt.close()
    if_alg1(normal_events_df, cut_off)

    if_alg2(abnormal_events_df, cut_off)

col1, col2, col3 = st.columns([30, 15, 30])
with col1:
    st.write("PROJECT BATCH: B7")
    st.caption(":blue[Akanksha Srivastava(1DB19CS007)]")
    st.caption(":blue[Akshatha A D(1DB19CS008)]")
    st.caption(":blue[Bhoomika B Poojari(1DB19CS024)]")
    st.caption(":blue[Gagana(1DB19CS050)]")
with col2:
    st.write("")
with col3:
    st.write("PROJECT GUIDE:")
    st.caption(":blue[Dr. Venugeetha Y]")
    st.caption(":blue[Professor, DBIT]")
