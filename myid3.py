# Module file for implementation of ID3 algorithm.


import pickle, warnings
import numpy as np
import pandas as pd
import scipy.stats as sc
import sklearn.metrics as sm


class DecisionTree:
    def __init__(self, load_from=None):
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.trained_tree = pickle.load(load_from)
            load_from.close()
            return


    def train(self, X, y, attrs, prune=False):
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.

        self.trained_tree = self.build_tree(X, y, attrs, prune)


    def predict(self, instance, tree):
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.

        try:
            for att in instance.index:
                if att in tree.keys():
                    try:
                        pred = tree[att][instance.loc[att]]
                    except:
                        # If value not found, take the most common value in the parent node
                        return self.backoff(tree)
                    if isinstance(pred, dict):
                        return self.predict(instance, pred)
                    else:
                        return pred
        except:
            raise ValueError('ID3 untrained')


    def test(self, X, y, display=False):
        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.
        try:
            tree = self.trained_tree
        except:
            raise ValueError('ID3 untrained')

        predictions = []
        for i in range(len(X)):
            instance = X.iloc[i]
            prediction = self.predict(instance, tree)
            predictions.append(prediction)

        warnings.filterwarnings("ignore")
        accuracy = sm.accuracy_score(y, predictions)
        recall = sm.recall_score(y, predictions, average='weighted')
        precision = sm.precision_score(y, predictions, average='weighted')
        f1score = sm.f1_score(y, predictions, average='weighted')
        confusion_matrix = sm.confusion_matrix(y, predictions, labels=list(set(y)))
        confusion_matrix = pd.DataFrame(confusion_matrix, index=list(set(y)), columns=list(set(y)))

        result = {'precision':precision,
                  'recall':recall,
                  'accuracy':accuracy,
                  'F1':f1score,
                  'confusion-matrix':confusion_matrix}
        if display:
            print(result)
        return result


    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        try:
            return str(self.trained_tree)
        except:
            return "ID3 untrained"


    def save(self, output):
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        pcl_string = pickle.dumps(self.trained_tree)
        output.write(pcl_string)


    def entropy(self, data):
        '''Calculate entropy from DataFrame'''
        probdata = data.value_counts(normalize=True) # probabilities
        H = sc.entropy(probdata, base=2)  # entropy
        return H


    def most_informative(self, X, y, attrs):
        '''Finds the most informative attribute'''
        choices = {}
        entropy_pre = self.entropy(y)
        for att in attrs:
            val_h = 0
            for val in set(X[att]):
                indexlist = X.index[X[att]==val].tolist()
                h = self.entropy(y.loc[indexlist])
                weighted_h = h * len(indexlist) / len(X)
                val_h += weighted_h
            ig = entropy_pre - val_h
            choices[att] = ig
        mi = max(choices, key=choices.get)
        return mi


    def build_tree(self, X, y, attrs, prune):
        '''Builds a decision tree and saves it to a dictionary'''

        # Base cases
        # If all values are the same, create leaf with the homogeneous value
        if len(set(y)) == 1:
            return y.iloc[0]

        # If no attributes are left, create a leaf with the most common value
        elif len(attrs) == 0:
            return y.value_counts().idxmax()

        # Pre-pruning when only one attribute is left.
        # Because of the way predict handles not found values
        # (it takes the most common value in the parent node),
        # this should not affect the accuracy negatively, but will
        # create a less complex tree.
        if prune:
            if len(attrs) == 1:
                return y.value_counts().idxmax()

        # Recursion if none of the base cases hold
        mi = self.most_informative(X, y, attrs)
        tree_dict = {mi:{}}  # initiate the subtree

        # Iterate through all values
        for value in set(X[mi]):
            sub_X = X.loc[X.index[X[mi]==value].tolist()].drop(mi, axis=1)  # Sub set of attributes, drop the attribute just used
            sub_y = y.loc[X.index[X[mi]==value].tolist()]  # Subset of target values
            sub_attrs = [att for att in attrs if att != mi]  # Drop the attribute just used
            sub_tree = self.build_tree(sub_X, sub_y, sub_attrs, prune)  # Run the function recursively, with the new sub set
            tree_dict[mi][value] = sub_tree  # Fill the sub tree
        return tree_dict


    def backoff(self, dictionary):
        '''Returns the most common value in a nested dictionary'''
        vals = []
        for value in dictionary.values():
            if isinstance(value, dict):
                vals.append(self.backoff(value))
            else:
                vals.append(value)
        return pd.Series(vals).value_counts().idxmax()

