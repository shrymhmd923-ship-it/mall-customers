# Customer Segmentation Using Unsupervised Learning

## Project Overview

This project applies unsupervised machine learning techniques to segment mall customers based on their age, annual income, and spending score.

The objective is to identify groups of customers with similar characteristics and understand their behavior to support data-driven marketing strategies.

## Project Objectives

- Explore and inspect customer data.
- Prepare numerical features for clustering.
- Apply feature scaling using StandardScaler.
- Implement K-Means clustering.
- Evaluate different cluster counts using the Elbow Method and Silhouette Score.
- Analyze cluster sizes and customer profiles.
- Apply Agglomerative Hierarchical Clustering.
- Visualize customer clusters and hierarchical relationships.

## Dataset

The dataset contains the following columns:

- `CustomerID`: Unique customer identifier.
- `Gender`: Customer gender.
- `Age`: Customer age.
- `Annual Income (k$)`: Annual income in thousands of dollars.
- `Spending Score (1-100)`: Customer spending score from 1 to 100.

The clustering models use three features:

- Age
- Annual Income
- Spending Score

CustomerID is excluded because it is an identifier rather than a meaningful behavioral feature. Gender is excluded from this initial version to focus on numerical features.

## Machine Learning Algorithms

### 1. K-Means Clustering

K-Means divides customers into a specified number of clusters based on their similarity.

The project evaluates different values of K and uses clustering metrics to help select a suitable number of customer groups.

### 2. Agglomerative Hierarchical Clustering

This algorithm starts with individual data points and progressively merges similar groups.

A dendrogram is used to visualize the hierarchical merging process.

## Data Preprocessing

### Feature Scaling

StandardScaler is applied to standardize numerical features before clustering.

Scaling is important because distance-based algorithms such as K-Means can be affected by differences in feature scales.

## Evaluation Methods

The project uses the following methods:

- **Elbow Method:** Examines how inertia changes as the number of clusters increases.
- **Silhouette Score:** Evaluates how well-separated and cohesive the clusters are.
- **Cluster Size Analysis:** Counts the customers assigned to each cluster.
- **Cluster Profile Analysis:** Compares average age, annual income, and spending score across clusters.
- **Dendrogram:** Visualizes the hierarchical clustering process.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## Project Structure

```text
Customer-Segmentation/
├── Untitled-1.py
├── Data.csv
├── README.md
└── requirements.txt
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Customer-Segmentation
```

Replace `YOUR_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python Untitled-1.py
```

Make sure that `Data.csv` is located in the same folder as `Untitled-1.py`.

## Results and Findings

The project compares different cluster counts and applies two clustering algorithms to explore customer similarities.

The final results should include:

- The selected number of clusters.
- The Silhouette Scores for the tested values of K.
- The number of customers in each cluster.
- The average age, annual income, and spending score of each cluster.
- A comparison of K-Means and Hierarchical Clustering.
- Potential marketing insights derived from the actual cluster profiles.

Actual numerical findings will be documented after running the project and examining its output.

## Limitations

- The analysis uses a limited number of customer features.
- The dataset may not represent all customer populations.
- Internal evaluation metrics do not guarantee business usefulness.
- The results depend on the selected features, preprocessing, and model parameters.

## Future Improvements

- Compare K-Means with DBSCAN.
- Explore additional customer features.
- Apply Principal Component Analysis (PCA) for dimensionality reduction and visualization.
- Investigate different clustering parameters.
- Develop marketing recommendations based on the observed customer profiles.

## Author

**Mohamed Mohsen Mohamed Soliman**

Artificial Intelligence Student | Machine Learning Enthusiast

## License

This project is intended for educational and portfolio purposes.
