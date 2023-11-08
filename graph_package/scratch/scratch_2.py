# from graph_package.configs.directories import Directories
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mpl_toolkits.mplot3d


def pca_2d(X):
    # 2D PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduced_2d = pca.transform(X)
    fig_2d = plt.figure(figsize=(8, 6))
    ax_2d = fig_2d.add_subplot(111)
    ax_2d.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1])
    ax_2d.set_title("2D Scatter Plot")
    plt.show()


def pca_3d(X):
    # 3D PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("Singular val:", pca.singular_values_)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    X_reduced = pca.transform(X)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        s=40,
    )
    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])
    plt.show()


def pca_2d_w_kmeans(X, num_components):
    # third
    reduced_data = PCA(n_components=num_components).fit_transform(X)
    kmeans = KMeans(init="k-means++", n_clusters=6, n_init=4)
    kmeans.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


emb_save_path = r"C:\Users\Mads-\Documents\Universitet\Kandidat\5_semester\thesis\GRLDrugProp\data\embeddings\rescal_100.csv"

X = pd.read_csv(emb_save_path)
num_components = 4
pca_2d(X)
pca_3d(X)
pca_2d_w_kmeans(X, num_components)
