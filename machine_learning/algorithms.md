Machine Learning Algorithms
--------------------------------------------------------------------------------------------------------------
Machine Learning algorithms are methods or mathematical models that enable machines to learn from data. They are categorized based on the type of learning: Supervised, Unsupervised, and Reinforcement.
--------------------------------------------------------------------------------------------------------------
1. Supervised Learning Algorithms
--------------------------------------------------------------------------------------------------------------
    These algorithms learn from labeled data (input + output pairs).

    a) Linear Regression

        Purpose: Predicts a continuous value using a straight-line relationship between input features and output.

        Example: Predicting house prices from size and location.

        Formula: y=mx+c
        where 
        y = output, 
        x = input, 
        m = slope, 
        c = y intercept.
        https://drive.google.com/file/d/1YXYtabOWjrgdwSN3C2Uf-wQy0TYvqK97/view?usp=sharing

    b) Logistic Regression

        Purpose: Used for classification problems (binary or multiclass).

        Example: Predicting if an email is spam or not spam.

        How it works: Uses the sigmoid function to output probabilities between 0 and 1.
        https://drive.google.com/file/d/1O7I6QQaNDKOzsMuKpquJBIs2TMD0uey8/view?usp=sharing

    c) Decision Trees

        Purpose: Splits data into branches based on feature values to reach a decision.

        Example: Predicting whether a customer will buy a product based on age and income.

        Advantage: Easy to interpret and visualize.

        Disadvantage: Can overfit.
        https://drive.google.com/file/d/1p8MalKpEbNKL-p4EQ8yFaBe88cOq9IuA/view?usp=sharing

    d) Random Forest

        Purpose: Ensemble of many decision trees. The final prediction is based on majority voting (classification) or average (regression).

        Example: Loan approval prediction in banks.

        Advantage: Reduces overfitting compared to a single decision tree.
        https://drive.google.com/file/d/1QedTve9hk2JOoZ4y3PatM5oqUPRnE5TC/view?usp=sharing

    e) Support Vector Machines (SVM)

        Purpose: Finds the best boundary (hyperplane) that separates different classes.

        Example: Face recognition (separating images of Person A vs. Person B).

        Advantage: Works well in high-dimensional spaces.
        https://drive.google.com/file/d/1D0UJUAKh5q2RR8wIoPqAtsCokIOupnSi/view?usp=sharing

    f) k-Nearest Neighbors (k-NN)

        Purpose: Classifies data based on the majority class among its k nearest neighbors.

        Example: Movie recommendation based on ratings from similar users.

        Disadvantage: Slow for large datasets.
        https://drive.google.com/file/d/1DaQnOqaVA8xD-AtZn4Q-7Er1oSHfAb8Q/view?usp=sharing

    g) Naïve Bayes

        Purpose: Based on Bayes’ Theorem, assumes features are independent.

        Example: Text classification (spam filtering, sentiment analysis).

        Advantage: Works well with high-dimensional data (like text).
        https://drive.google.com/file/d/1ojESilTgVJRP5K0qL6L-ZVGrVQ22awlq/view?usp=sharing

2. Unsupervised Learning Algorithms

These algorithms work with unlabeled data to find hidden patterns.

    a) k-Means Clustering

        Purpose: Groups data into k clusters based on similarity.

        Example: Market segmentation (grouping customers with similar behavior).

        Steps:

        Choose number of clusters (k).

        Assign points to nearest cluster center.

        Update cluster centers until convergence.
        https://drive.google.com/file/d/1IJPCrc4rVPOsvg1kBvqMA3-WzJujt69P/view?usp=sharing

    b) Hierarchical Clustering

        Purpose: Builds a tree (dendrogram) of clusters.

        Example: Gene similarity in biology.

        Advantage: Does not require pre-defining k.
        https://drive.google.com/file/d/1aTvA2EIqjAXpJKDYoSjlaIy7g1e7S0Lu/view?usp=sharing

    c) Principal Component Analysis (PCA)

        Purpose: Dimensionality reduction – reduces number of features while keeping most variance.

        Example: Reducing image data for face recognition.

        Advantage: Improves speed and reduces noise.
        https://drive.google.com/file/d/1-OEjTNtfTD2s3Tn5XQIS5_IsW74r92-R/view?usp=sharing

    d) DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

        Purpose: Groups together dense regions and identifies outliers.

        Example: Fraud detection (finding unusual transactions).

        Advantage: Works well for irregularly shaped clusters.
        https://drive.google.com/file/d/1Ui9_z-QEVjkKiEHARUJZG2ml-QvzOpfe/view?usp=sharing

3. Reinforcement Learning Algorithms

    These learn through trial-and-error and feedback (rewards).

    a) Q-Learning

        Purpose: Model-free RL algorithm that learns the value of actions.

        Example: A robot learning how to navigate a maze.

        Core Idea: Updates Q-values (expected rewards) using Bellman Equation.
        https://drive.google.com/file/d/1tQH1u83nv_PKzECq8TVU5e9xRnjDPldM/view?usp=sharing

    b) Deep Q-Networks (DQN)

        Purpose: Combines Q-Learning with Deep Neural Networks.

        Example: Google DeepMind’s Atari game-playing AI.

        Advantage: Can handle large, complex state spaces.
        https://drive.google.com/file/d/1bQSUfhLeZySZ79eie1888SKlTFfJ4ru7/view?usp=sharing

    c) Policy Gradient Methods

        Purpose: Learn policies directly instead of value functions.

        Example: Training humanoid robots to walk.

        Advantage: Effective for continuous action spaces.
        https://drive.google.com/file/d/1xzkWP6aTE6nb4baYXowjSdUXU5Q7qdfj/view?usp=sharing