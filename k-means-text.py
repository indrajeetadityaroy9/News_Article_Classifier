import numpy as np
from scipy.sparse import vstack, csr_matrix, issparse
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    shuffle=True,
    random_state=42,
)
labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

def fit_and_evaluate(km, X, n_runs=5, labels=labels):
    """
    Fit the given KMeans model multiple times and report average metrics.
    """
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        pred_labels = km.labels_
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, pred_labels))
        scores["Completeness"].append(metrics.completeness_score(labels, pred_labels))
        scores["V-measure"].append(metrics.v_measure_score(labels, pred_labels))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, pred_labels)
        )
        # Sampling for silhouette if dataset is large could be done, but here it's fixed
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, pred_labels, sample_size=2000)
        )

    # Print average ± std for each metric
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")


def visualize_clusters_with_categories(X, cluster_labels, true_labels, newsgroup_names, sample_size=3000):
    """
    Visualize clustered data in 2D using PCA, with each cluster labeled by the 
    most commonly occurring true category (newsgroup) within that cluster.
    """
    # If the dataset is large, sample a subset for visualization
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
        cluster_labels_sample = cluster_labels[idx]
        true_labels_sample = true_labels[idx]
    else:
        X_sample = X
        cluster_labels_sample = cluster_labels
        true_labels_sample = true_labels

    # Determine the most common true category in each cluster
    cluster_to_category = {}
    for cluster_id in np.unique(cluster_labels_sample):
        # Extract the true labels of points in this cluster
        cluster_true_labels = true_labels_sample[cluster_labels_sample == cluster_id]
        # Find the most frequent category
        most_common_category = np.bincount(cluster_true_labels).argmax()
        cluster_to_category[cluster_id] = most_common_category

    # Convert to dense if needed for PCA
    if issparse(X_sample):
        X_dense = X_sample.toarray()
    else:
        X_dense = X_sample

    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    reduced_X = pca.fit_transform(X_dense)

    # Plot each cluster with the category it most resembles
    plt.figure(figsize=(10, 8))
    for cluster_id, category_id in cluster_to_category.items():
        cluster_points = reduced_X[cluster_labels_sample == cluster_id]
        label_name = newsgroup_names[category_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=label_name, alpha=0.5, edgecolors='w')

    plt.title('Clusters Labeled by Most Common Actual Newsgroup Category')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.legend()
    plt.show()


vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
X_tfidf = vectorizer.fit_transform(dataset.data)

class KMeans:
    """
    A simple KMeans clustering implementation supporting sparse input.
    """
    def __init__(self, n_clusters, max_iterations=300, tolerance=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Randomly select initial centroids
        initial_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[initial_indices] if issparse(X) else csr_matrix(X[initial_indices])

        for iteration in range(self.max_iterations):
            # Compute distances to all centroids
            distances = euclidean_distances(X, self.centroids)
            # Assign labels based on closest centroid
            new_labels = np.argmin(distances, axis=1)

            # If no change in labels, we can consider stopping early
            if self.labels_ is not None and np.array_equal(new_labels, self.labels_):
                break

            self.labels_ = new_labels

            # Compute new centroids
            new_centroids = []
            for cluster_idx in range(self.n_clusters):
                cluster_samples = X[self.labels_ == cluster_idx]
                if cluster_samples.shape[0] > 0:
                    # Mean along axis=0
                    mean_centroid = cluster_samples.mean(axis=0)
                    # Convert mean to CSR if necessary
                    if not issparse(mean_centroid):
                        mean_centroid = csr_matrix(mean_centroid)
                    new_centroids.append(mean_centroid)
                else:
                    # Reinitialize empty cluster centroid at random
                    random_idx = np.random.randint(0, X.shape[0])
                    if issparse(X):
                        new_centroids.append(X[random_idx])
                    else:
                        new_centroids.append(csr_matrix(X[random_idx]))

            new_centroids = vstack(new_centroids)

            # Check for convergence: sum of squared differences
            centroid_diff = (new_centroids - self.centroids).power(2).sum()
            self.centroids = new_centroids

            if centroid_diff < self.tolerance:
                break

    def set_params(self, **params):
        """
        Update parameters for the KMeans instance.
        """
        for key, value in params.items():
            setattr(self, key, value)

print("Custom KMeans Scores:")
kmeans_custom = KMeans(n_clusters=true_k, max_iterations=300, tolerance=0.0001, random_state=42)
fit_and_evaluate(kmeans_custom, X_tfidf, labels=labels)

print("\nScikit-learn KMeans Scores:")
kmeans_sklearn = SKLearnKMeans(n_clusters=true_k, max_iter=300, tol=0.0001, random_state=42)
fit_and_evaluate(kmeans_sklearn, X_tfidf, labels=labels)

# Visualize the clusters of the custom KMeans
kmeans_custom.fit(X_tfidf)
visualize_clusters_with_categories(X_tfidf, kmeans_custom.labels_, labels, dataset.target_names)

# Visualize clusters from scikit-learn KMeans
kmeans_sklearn.fit(X_tfidf)
visualize_clusters_with_categories(X_tfidf, kmeans_sklearn.labels_, labels, dataset.target_names)
