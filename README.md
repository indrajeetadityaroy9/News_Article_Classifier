# News Article Classifier with Custom K-Means Implementation

Development and evaluation of a custom K-Means clustering algorithm tailored for text data, applied to the 20 Newsgroups dataset.Traditional clustering methods, such as K-Means, are commonly used on numeric datasets. Applying these methods directly to text data requires transforming documents into numeric feature vectors.Term Frequency-Inverse Document Frequency (TF-IDF) is utilized to convert raw text into a sparse, high-dimensional feature space suitable for clustering. Once the data is vectorized, two clustering pipelines are executed in parallel: Custom K-Means (From-scratch K-Means algorithm with enhancements such as k-means++ initialization, improved convergence checks, and handling of empty clusters) and Scikit-learn KMeans (Optimized KMeans implementation from Scikit-learn, serving as a baseline to benchmark the custom approach).

## Key Implementation Components

- Custom K-Means algorithm implemented from scratch, the custom class supports K-means++ initialization for improved initial centroid selection, robust handling of empty clusters by reassigning centroids when necessary, iterative centroid updates based on mean positions of assigned samples and customizable convergence criteria and maximum iterations.

- The Scikit-learn KMeans class is used as a benchmark to evaluate clustering quality metrics:
  - **Homogeneity**: Measures whether each cluster contains only samples from a single class.
  - **Completeness**: Measures whether all samples of a given class are assigned to the same cluster.
  - **V-Measure**: Harmonic mean of homogeneity and completeness.
  - **Adjusted Rand Index (ARI)**: Measures similarity between the true labels and the clustering assignments, adjusted for chance.
  - **Silhouette Coefficient**: Assesses how well-separated the clusters are, based on distances between samples.
