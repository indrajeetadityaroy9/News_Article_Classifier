import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#################################################################
# Load Dataset
#################################################################

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    shuffle=True,
    random_state=42,
)

labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]


#################################################################
# Evaluate Fitness
#################################################################
def fit_and_evaluate(km, X, n_runs=5):
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")


newsgroup_names = dataset.target_names

def visualize_clusters(X, labels, n_clusters):
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X.toarray())  # Convert X to dense format if necessary
    
    # Create a scatter plot for each cluster using its corresponding newsgroup name
    plt.figure(figsize=(10, 8))  # Optional: Adjust figure size for better visualization
    for i in range(n_clusters):
        cluster_points = reduced_X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=newsgroup_names[i], alpha=0.5, edgecolors='w')
    
    plt.title('Cluster Visualization with Newsgroup Names')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.legend()
    plt.show()
#################################################################
# Vectorize 
#################################################################
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(dataset.data)


#################################################################
# (TODO): Implement K-Means  
#################################################################

# Custom KMeans class
class KMeans:
    def __init__(self, n_clusters, max_iterations, tolerance, random_state):
        # Number of clusters
        self.n_clusters = n_clusters
        # Maximum number of iterations for convergence
        self.max_iterations = max_iterations
        # Tolerance for convergence check
        self.tolerance = tolerance
        # Random state for reproducibility
        self.random_state = random_state
        # List to store centroids of clusters
        self.centroids = []
        # List to store labels of each data point
        self.labels_ = []

    def fit(self, X):
        # Set the random seed
        np.random.seed(self.random_state)
        # Randomly select initial centroids from the dataset
        initial_centroid_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[initial_centroid_indices]

        for iteration in range(self.max_iterations):
            # Calculate distances from each sample to each centroid
            distances_to_centroids = euclidean_distances(X, self.centroids)
            # Assign each sample to the closest centroid
            self.labels_ = np.argmin(distances_to_centroids, axis=1)
            # Initialize list to hold the updated centroids
            updated_centroids = []
            # Compute centroids as the mean of samples assigned to each cluster
            for cluster_index in range(self.n_clusters):
                # Samples belonging to the current cluster
                cluster_samples = X[self.labels_ == cluster_index]
                # If the cluster has at least one sample, calculate its new centroid
                if cluster_samples.shape[0] > 0:
                    mean_centroid = cluster_samples.mean(axis=0)
                    if isinstance(mean_centroid, np.ndarray):
                        # reshape method is used to ensure that mean_centroid is in the correct 2D format
                        # csr_matrix converts into matrix
                        mean_centroid = csr_matrix(mean_centroid.reshape(1, -1))
                    updated_centroids.append(mean_centroid)
                else:
                    # Keep the original centroid if a cluster has no samples
                    updated_centroids.append(csr_matrix(self.centroids[cluster_index].toarray().reshape(1, -1)))
            # Stack the updated centroids into a single matrix
            updated_centroids = vstack(updated_centroids)
            # Calculate the difference between the updated centroids and the previous centroids
            centroid_deltas = updated_centroids - self.centroids
            # Compute the total shift of all centroids for convergence checking
            centroids_shift = np.sum(centroid_deltas.multiply(centroid_deltas).sum(axis=1))
            # Update centroids for the next iteration
            self.centroids = updated_centroids
            # Break if centroids have shifted less than the tolerance threshold
            if centroids_shift < self.tolerance:
                break

    def set_params(self, **params):
        # Method to update parameters of the KMeans instance
        for key, value in params.items():
            setattr(self, key, value)


# Evaluation of Custom KMeans
print("Custom KMeans Scores:")
kmeans_custom = KMeans(n_clusters=true_k, max_iterations=300, tolerance=0.0001, random_state=42)
fit_and_evaluate(kmeans_custom, X_tfidf)

# Evaluation of Scikit-learn KMeans
print("\nScikit-learn KMeans Scores:")
kmeans_sklearn = SKLearnKMeans(n_clusters=true_k, max_iter=300, tol=0.0001, random_state=42)
fit_and_evaluate(kmeans_sklearn, X_tfidf)

visualize_clusters(X_tfidf, kmeans_custom.labels_, true_k)

# Or after fitting the Scikit-learn KMeans:
visualize_clusters(X_tfidf, kmeans_sklearn.labels_, true_k)
