from collections import Counter
from evofuzzy import fuzzyclassifier
from sklearn.datasets import load_iris
import pandas as pd

"""Script for testing the classifier by running it on the iris dataset.
"""

classifier = fuzzyclassifier.FuzzyClassifier(
    population_size=20,
    hall_of_fame_size=1,
    max_generation=20,
    mutation_prob=0.9,
    crossover_prob=0.1,
)
data = load_iris()
cols = [c.replace(" ", "_").replace("_(cm)", "") for c in data.feature_names]
iris = pd.DataFrame(data.data, columns=cols)
y = pd.Series(data.target)

classes = {name: val for (name, val) in zip(data.target_names, range(3))}
classifier.fit(iris, y, classes)
print(f"Best Rule:  size = {len(classifier.best)}")
print("\n".join(classifier.best_strs))

print(
    "Final length of rules sets",
    dict(Counter(x.length for x in classifier.population_)),
)
