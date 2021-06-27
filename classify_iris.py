from evofuzzy import fuzzyclassifier
from sklearn.datasets import load_iris
import pandas as pd

# from random import sample

classifier = fuzzyclassifier.FuzzyClassifier(
    population_size=20, hall_of_fame_size=3, max_generation=20
)
data = load_iris()
cols = [c.replace(" ", "_").replace("_(cm)", "") for c in data.feature_names]
iris = pd.DataFrame(data.data, columns=cols)
y = pd.Series(data.target)
# idx = sample(range(len(y)), 10)

classes = {name: val for (name, val) in zip(data.target_names, range(3))}
classifier.fit(iris, y, classes)
print("\n".join(classifier.best_strs))