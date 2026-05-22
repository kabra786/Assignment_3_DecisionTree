import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title):

    # Define ranges
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Create mesh grid
    x_values, y_values = np.meshgrid(
        np.arange(min_x, max_x, 1),
        np.arange(min_y, max_y, 1)
    )

    # Predict output
    mesh_output = classifier.predict(
        np.c_[x_values.ravel(), y_values.ravel()]
    )

    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot decision boundary
    plt.figure()

    plt.pcolormesh(
        x_values,
        y_values,
        mesh_output,
        cmap=plt.cm.gray,
        shading='auto'
    )

    # Plot points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=75,
        edgecolors='black',
        linewidth=1,
        cmap=plt.cm.Paired
    )

    plt.title(title)