from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification, make_moons
import io
import base64
import pydotplus
import graphviz
from flask import Flask
from sklearn.datasets import make_blobs
from flask import Flask, render_template, request, jsonify


# Use the Agg backend for matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/svm')
def svm():
    # Select dataset based on user input
    dataset = request.args.get('dataset', 'linear_separable')
    if dataset == 'linear_separable':
        X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    elif dataset == 'nonlinear_separable':
        X, y = make_moons(noise=0.2, random_state=0)
    else:
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = iris.target

    # Get SVM parameters from user input
    kernel = request.args.get('kernel', 'linear')
    C = float(request.args.get('C', 1.0))
    gamma = request.args.get('gamma', 'scale')
    degree = int(request.args.get('degree', 3))

    # Create SVM model
    svc = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree).fit(X, y)

    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Plot decision boundary
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary\nKernel={kernel}, C={C}, Gamma={gamma}, Degree={degree}')

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return render_template('svm.html', plot_url=plot_url)



@app.route('/knn')
def knn_visualization():
    # Get form inputs
    k = int(request.args.get('k', 5))
    n_classes = int(request.args.get('n_classes', 2))  # Reduce number of classes
    n_points = int(request.args.get('n_points', 100))

    # Generate random data
    X, y = make_classification(n_samples=n_points, n_features=2, n_classes=n_classes,
                               n_clusters_per_class=1, n_redundant=0, random_state=42)

    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # Create a meshgrid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict the labels for meshgrid points
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'KNN Decision Boundary (K={k}, Classes={n_classes}, Points={n_points})')

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()  # Close the plot to avoid memory leaks
    return render_template('knn.html', plot_url=plot_url)



@app.route('/kmeans')
def kmeans():
    # Get parameters from the form
    n_clusters = int(request.args.get('n_clusters', 3))
    init = request.args.get('init', 'k-means++')
    n_init = int(request.args.get('n_init', 10))
    max_iter = int(request.args.get('max_iter', 300))
    algorithm = request.args.get('algorithm', 'lloyd')
    
    # Create synthetic dataset
    X, _ = make_blobs(n_samples=300, centers=n_clusters, cluster_std=0.60, random_state=0)
    
    # Fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, algorithm=algorithm, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    
    # Plot the centroids
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'KMeans Clustering (n_clusters={n_clusters})')
    
    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('kmeans.html', plot_url=plot_url)
@app.route('/decision_tree', methods=['GET'])
def decision_tree():
    # Get parameters from the form
    max_depth = int(request.args.get('max_depth', 3))
    min_samples_split = int(request.args.get('min_samples_split', 2))
    min_samples_leaf = int(request.args.get('min_samples_leaf', 1))

    # Create synthetic dataset
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

    # Fit Decision Tree model
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=0)
    clf.fit(X, y)

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'])

    # Save plot to a string in base64 format
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('decision_tree.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
