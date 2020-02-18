import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy

from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs, make_moons
from sklearn.datasets import load_digits

import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    circles = np.array(np.swapaxes(circles, 0, 1))

    plt.plot(circles[:,0], circles[:,1], '.k')
    plt.show()

    return (circles)

def draw_circles():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t), np.sin(t)])
    circle2 = np.matrix([2 * np.cos(t), 2 * np.sin(t)])

    circles = np.concatenate((circle1, circle2), axis=1)
    circles = np.array(np.swapaxes(circles, 0, 1))

    plt.plot(circles[:,0], circles[:,1], '.k')
    plt.show()

    return (circles)



def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()

    return (apml)


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    print(data.shape)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()

def get_microarray_data(data_path='microarray_data.pickle',
                        genes_path='microarray_genes.pickle',
                        conds_path='microarray_conds.pickle'):
    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    return (data)


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    #if X.shape[1] != Y.shape[1]:
        #raise AssertionError(f"Dim of X {X.shape} does not match the Dim of Y {Y.shape}.")

    distances = [np.linalg.norm(x - Y, axis=1) for x in X]
    return (distances)



def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    centroid = np.mean(X, axis=0)
    return (centroid)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    # Choose the first centroid uniformly at random among the data points.
    start = X[np.random.choice(X.shape[0], 1, replace=False), :]
    centroids = start

    for idx in range(1, k):
        #For each point xi compute the distance.
        distances = np.array(metric(X, centroids))

        # keep the lowest of the computed distances between all choosen centroids.
        dist = np.array([cdist.min() for cdist in distances])

        # Prepare for choosing by weightend probability distribution.
        one_d = dist.flatten()
        normed = [float(i)/sum(one_d) for i in one_d]
        #normed[np.isnan(normed)] = 0

        # Sample the next centroid from points with probability proportional to distance.
        choice = X[np.random.choice(X.shape[0], 1, replace=False, p=normed), :]
        centroids = np.append(centroids, choice, axis=0)

    return (centroids)


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    # Initialize the centroids.
    centroids = init(X, k, metric)

    for idx in range(iterations):
        dist = np.array(metric(X, centroids))
        # Update cluster indicators.
        clustering = np.array([point.argmin() for point in dist])
        # Update centroids.
        new_centroids = np.array([center(X[clustering == i]) for i in range(k)])
        #Check for convergence.
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return (centroids, clustering)


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.exp(-(X ** 2/(2 * sigma**2)))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    similarity = np.zeros((X.shape))
    for row in range(X.shape[0]):
        idx = np.argpartition(X[row], m)
        similarity[row][idx[:k]] = 1

    return (similarity)


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    N = X.shape[0]

    # 1. Compute a similarity graph (Laplacian matrix)
    s = np.array([np.linalg.norm(x - X, axis=1) for x in X])
    w = similarity(s, similarity_param)
    d = np.diag(np.sum(w, axis=1) ** (-1/2))
    l = np.identity(N) - d.dot(w).dot(d)

    # 2. Project the data onto a low-dimensional space
    eig_val, eig_vect = scipy.sparse.linalg.eigs(l, k, sigma=0, which='LM', maxiter=2*N)
    tmp = eig_vect.real
    rows_norm = np.linalg.norm(tmp, axis=1, ord=2)
    Y = (tmp.T / rows_norm).T

    # 3. Create clusters
    _, clustering = kmeans(Y, k)

    return (clustering)


def elbow_evaluation(X, max_k, method=kmeans, similarity_param=0.2):
    '''
    The Elbow Method - Gives back a list of cost function values for a range of k values.
    :param X: A NxD data matrix.
    :param max_k: The maximum number of desired clusters that should be tested. - Range: 1 - max_k
    :param method: The method (kmeans++ or spectral) you want to evaluate.
    :return: list of values computed for a specific k mit cost function.
    '''
    N = X.shape[0]
    sse_list = []
    for k in range(1, max_k + 1):
        print(k)
        centers, clustering = method(X, k)
        result = 10.0
        for c in range(centers.shape[0]):
            tmp_n = np.where(clustering == c)[0].shape[0]
            tmp_cluster = X[np.where(clustering == c)]
            tmp_dist = np.array([np.linalg.norm(x - centers[0]) for x in tmp_cluster])
            tmp_result = np.sum(tmp_dist) / tmp_n
            if tmp_result < result:
                result = tmp_result

        sse = result
        sse_list.append(sse)

    return (sse_list)


def methods_comparison(n=8000):
    '''
    This function was created in order to compare tSNE with PCA. Run this for checking the results.
    :param n: Number for creating a subset.
    '''
    # Load the dataset.
    mnist = fetch_mldata("MNIST original")
    X = mnist.data / 255.0
    y = mnist.target

    # Prepare the dataset.
    feat_cols = ['pixel'+ str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    rndperm = np.random.permutation(df.shape[0])
    df_subset = df.loc[rndperm[:n],:].copy()

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_subset[feat_cols].values)

    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1]
    df_subset['pca-three'] = pca_result[:,2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(1)
    sns.scatterplot(x="pca-one", y="pca-two", hue="y", palette=sns.color_palette("hls", 10), data=df_subset, legend="full", alpha=0.3)

    # Perform tSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df_subset[feat_cols].values)

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(2)
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y", palette=sns.color_palette("hls", 10), data=df_subset, legend="full", alpha=0.3)
    plt.show()



def frange(start, stop, step):
    '''
    function in order to create a range of float values. (Not implemented yet in python, only for integer)
    '''
    i = start
    while i < stop:
         yield i
         i += step


if __name__ == '__main__':
    #X, y_true = make_blobs(n_samples=800, centers=4, cluster_std=0.40, random_state=0)
    #X, y_true = make_moons(n_samples=300, noise=.05)
    #X = circles_example()

    #methods_comparison()

    #plt.figure(1)
    #plt.scatter(X[:, 0], X[:, 1], s=50)
    #plt.show()


    #centers, labels = kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init)
    #similarity_param = 0.25
    #labels = spectral(X, k, similarity_param, similarity=gaussian_kernel)

    #sse_eval = elbow_evaluation(X, max)

    """
    for similarity_param in frange(0.02, 0.5, 0.02):
        print(similarity_param)

        labels = spectral(X, k, similarity_param, similarity=gaussian_kernel)

        plt.figure(1)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
        #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.show()
    """
