"""
Helper functions for data visualization
"""
# src/visualization.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", context="talk")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def visualize_missing(data):
    """
    Visualize missing values in a DataFrame
    """
    plt.figure(figsize=(10,6))
    sns.heatmap(data.isna(), cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()

def plot_class_distribution(df, column="diagnosis", colors=None):
    """
    Plot distribution of a target class and return counts and percentages
    """
    # calculate counts and percentages
    counts = df[column].value_counts()
    pcts = df[column].value_counts(normalize=True) * 100

    # create barplot
    plt.bar(
        pcts.index,
        pcts.values,
        color=colors.values() if colors else None,
        edgecolor="black"
    )

    # titles and labels
    plt.title("Target Class Distribution", fontsize=16)
    plt.xlabel(column.capitalize(), fontsize=14)
    plt.ylabel("Percent", fontsize=14)
    plt.ylim(0, 105)

    # add percentages above bars
    for i, pct in enumerate(pcts):
        plt.text(i, pct + 3, f"{pct:.2f}%", ha="center")

    plt.grid(False)
    plt.show()

    return counts, pcts

def plot_correlation_heatmap(df: pd.DataFrame, title="Feature Correlation Heatmap"):
    """
    Plot correlation heatmap for all numeric features.
    """
    # calculate correlation matrix
    corr = df.drop(columns = "diagnosis").corr()
    high_corr = corr[(abs(corr) > 0.9) & (abs(corr) < 1.0)]

    mask = np.triu(np.ones_like(corr, dtype=bool)) # remove upper triangle

    plt.figure(figsize=(12,10))

    # plot correlation matrix
    sns.heatmap(
        corr, mask=mask, 
        annot=False, center=0,
        square=True, cmap="coolwarm", 
        fmt=".2f", cbar=True,
        linewidths = 0.5, 
        vmin = -0.5, vmax = 1
    )
    
    # add titles
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return corr, high_corr

def plot_pca(df, target="diagnosis", n_components=2, palette="Set2"):
    """
    Performs PCA on all features in df (excluding target) and plots 
    the first n_components (default=2).
    """
    # separate features and target
    X = df.drop(columns=target)
    y = df[target]

    # standardize features (entire dataset)
    X_scaled = StandardScaler().fit_transform(X)

    # perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # plot
    pca_df = pd.DataFrame({
        f"PC{i+1}": X_pca[:, i] for i in range(n_components)
    })
    pca_df[target] = y

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=target,
        palette=palette,
        alpha=0.8,
        s=70
    )

    plt.title("PCA Projection of Features")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.legend(title=target)
    
    sns.despine()
    plt.tight_layout()
    
    plt.show()


def plot_pairplot(df, features, target="diagnosis", corner=True, height=2.6, palette="Set2"):
    
    plt.figure(figsize=(8,6))
    sns.pairplot(
        df,
        vars=features,
        hue=target,
        diag_kind="kde",
        corner=corner,
        plot_kws={"alpha": 0.5},
        height=height,
        palette=palette
    )

    plt.grid(None)
    plt.suptitle("Pair Plot of Key Features", fontsize=26)
    plt.show()


def plot_eda_summary(df, target="diagnosis", pairplot_features=None, palette="Set2"):
    """
    Combined EDA summary using previously defined functions (above):
        1. Class distribution
        2. PCA projection
        3. Pairplot of selected features
    
    Parameters:
    -----------
        df (pd.DataFrame): dataset with features + target
        target (str): name of the target column
        pairplot_features (list): features to include in pairplot
        palette (str/list): color palette for classes
    """
    # 1. Class Distribution
    plot_class_distribution(df, column=target, colors=palette)

    # 2. PCA Projection
    plot_pca(df, target=target, palette=palette)

    # 3. Pairplot
    if pairplot_features is None:
        pairplot_features = df.drop(columns=target).columns[:4].tolist()
    plot_pairplot(df, features=pairplot_features, target=target, palette=palette)






#############################################################################











def plot_feature_importance(features, importances, top_n=10, colors=None, title="Feature Importance"):
    """
    Plot horizontal bar chart of top N feature importances.
    """
    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(top_n)
    plt.figure(figsize=(8,6))
    plt.barh(df["feature"], df["importance"], color=colors)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_scores, label=None):
    """
    Plot ROC curve.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], color="grey", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
