# Text Clustering Analysis

This repository contains a Jupyter Notebook (`your_notebook_name.ipynb`) that performs text clustering analysis on survey response data from an Excel file. The notebook explores different clustering algorithms, including K-Means, DBSCAN, and Hierarchical Clustering (Agglomerative Clustering), to group similar text responses.

## Notebook Contents

The notebook includes the following sections:

1.  **Data Loading and Preprocessing:**
    *   Mounts Google Drive to access the Excel file.
    *   Reads the survey response data from a specified Excel file and sheet.
    *   Concatenates all columns into a single text column for clustering.
    *   Includes text preprocessing steps such as converting to lowercase and removing punctuation (in some sections).

2.  **K-Means Clustering:**
    *   Uses `TfidfVectorizer` for feature extraction with TF-IDF.
    *   Applies `StandardScaler` for normalization.
    *   Implements the Elbow Method to help determine the optimal number of clusters.
    *   Applies the K-Means algorithm and interprets the clusters by identifying the top features for each centroid.

3.  **DBSCAN Clustering:**
    *   Uses `TfidfVectorizer` for feature extraction with TF-IDF.
    *   Applies `StandardScaler` for normalization (in one section).
    *   Applies the DBSCAN algorithm to identify clusters based on density.
    *   Prints a condensed output of the text responses within each cluster.

4.  **Hierarchical Clustering (Agglomerative Clustering):**
    *   Uses `TfidfVectorizer` for feature extraction with TF-IDF.
    *   Applies `StandardScaler` for normalization.
    *   Applies the Agglomerative Clustering algorithm to perform hierarchical clustering.
    *   Prints a condensed output of the text responses within each cluster.

## Prerequisites

To run this notebook, you will need:

*   Python 3.x
*   Google Colab environment or a local Jupyter Notebook setup.
*   Required libraries: `pandas`, `scikit-learn`, `matplotlib`, `nltk`.

You can install the required libraries using pip:


Note: `openpyxl` is required by pandas to read `.xlsx` files.

## Data

The notebook expects an Excel file containing the survey responses. The file path and sheet name are specified within the notebook. Make sure to update these paths to point to your data file.

## Usage

1.  Open the Jupyter Notebook (`your_notebook_name.ipynb`) in Google Colab or your local Jupyter environment.
2.  Ensure that the required libraries are installed.
3.  Update the `excel_file_path` and `sheet_name` variables to point to your data file.
4.  Run the cells sequentially to perform the data loading, preprocessing, and clustering analysis.
5.  Modify the parameters for each clustering algorithm (`n_clusters` for K-Means and Agglomerative Clustering, `eps` and `min_samples` for DBSCAN) to experiment with different clustering results.
6.  Analyze the output of each clustering algorithm to understand the groupings of text responses.

## Customization

*   **Stop words:** The notebook uses NLTK's English stop words. You can customize the `custom_stopwords` list to include or exclude specific words relevant to your data.
*   **TF-IDF parameters:** Adjust `max_features` in `TfidfVectorizer` to control the number of features used for clustering.
*   **Clustering parameters:** Experiment with the parameters of each clustering algorithm to achieve desired clustering results.

## Contributing

Feel free to fork this repository and adapt the code for your own text clustering tasks.


