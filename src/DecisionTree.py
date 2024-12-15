import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Node:
    """
    Create a node for the Decision Tree
    """
    
    def __init__(self, feature_index: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None, info_gain: float = None, value: int = None):
        """
        Constructor for the Node class
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain  
        self.value = value

class DecisionTree(BaseEstimator, ClassifierMixin):
    """
    Decision Tree Classifier with scikit-learn compatible parameters.
    
    Parameters:
    -----------
    max_depth : int or None, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
        
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least min_samples_leaf training samples in each of the left and
        right branches.
        
    max_features : int or None, default=None
        The number of features to consider when looking for the best split:
        - If None, then consider all features.
        - If int, then consider max_features features at each split.
        
    mode : str, default="entropy"
        The function to measure the quality of a split:
        - 'entropy' for information gain with entropy
        - 'gini' for gini impurity
    """
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: int = None, mode: str = "entropy"):
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.max_features = max_features
        self.mode = mode
        self.root = None

    def _get_random_features(self, num_features: int) -> np.ndarray:
        """Select a random subset of features if max_features is specified."""
        if self.max_features is None or self.max_features >= num_features:
            return np.arange(num_features)
        else:
            return np.random.choice(num_features, self.max_features, replace=False)

    def split(self, dataset: np.ndarray, feature_index: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:     
        """Split the dataset ensuring min_samples_leaf constraint is met."""
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def entropy(self, y: np.ndarray) -> float:
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            entropy += -p * np.log2(p)
        return entropy
    
    def gini_index(self, y: np.ndarray) -> float:    
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p = len(y[y == cls]) / len(y)
            gini += p * (1 - p)
        return gini
    
    def calculate_leaf_value(self, y: np.ndarray) -> int:
        Y = list(y)
        return max(Y, key=Y.count)
        
    def information_gain(self, parent_node: np.ndarray, left_child: np.ndarray, 
                        right_child: np.ndarray, mode: str = "gini") -> float:
        left_weight = len(left_child) / len(parent_node)
        right_weight = len(right_child) / len(parent_node)

        if mode == "gini":
            gain = self.gini_index(parent_node) - (
                left_weight * self.gini_index(left_child) + 
                right_weight * self.gini_index(right_child)
            )
        else:
            gain = self.entropy(parent_node) - (
                left_weight * self.entropy(left_child) + 
                right_weight * self.entropy(right_child)
            )
        return gain

    def get_best_split(self, dataset: np.ndarray, num_features: int) -> dict:
        """Find the best split considering max_features and min_samples_leaf constraints."""
        best_split = {}
        max_info_gain = -float('inf')
        
        # Consider only a random subset of features if max_features is specified
        feature_indices = self._get_random_features(num_features)

        for feature_index in feature_indices:
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                
                # Check min_samples_leaf constraint
                if (len(dataset_left) >= self.min_samples_leaf and 
                    len(dataset_right) >= self.min_samples_leaf):
                    
                    y = dataset[:, -1]
                    y_left = dataset_left[:, -1]
                    y_right = dataset_right[:, -1]
                    
                    current_info_gain = self.information_gain(y, y_left, y_right, self.mode)

                    if current_info_gain > max_info_gain:
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'data_left': dataset_left,
                            'data_right': dataset_right,
                            'info_gain': current_info_gain
                        }
                        max_info_gain = current_info_gain

        return best_split

    def construct_tree(self, dataset: np.ndarray, current_depth: int = 0) -> Node:
        """
        Construct the decision tree recursively with scikit-learn compatible stopping criteria.
        """
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape
        
        # Check all stopping criteria
        should_split = (
            # Check if we have enough samples to split
            num_samples >= self.min_samples_split and
            # Check if we haven't reached max_depth (if specified)
            (self.max_depth is None or current_depth <= self.max_depth) and
            # Check if the node is not pure
            len(np.unique(Y)) > 1
        )

        if should_split:
            best_split = self.get_best_split(dataset, num_features)
            
            # Only split if we found a valid split that improves information gain
            if best_split and best_split.get('info_gain', 0) > 0:
                left_subtree = self.construct_tree(best_split['data_left'], current_depth + 1)
                right_subtree = self.construct_tree(best_split['data_right'], current_depth + 1)
                
                return Node(
                    best_split['feature_index'],
                    best_split['threshold'],
                    left_subtree,
                    right_subtree,
                    best_split['info_gain']
                )
        
        # Create a leaf node
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the decision tree to the training data."""
        Y = np.reshape(Y, (-1, 1))
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.construct_tree(dataset)
        return self

    def _predict(self, x: np.ndarray, tree: Node) -> int:
        """Make a prediction for a single sample."""
        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        return np.array([self._predict(x, self.root) for x in X])