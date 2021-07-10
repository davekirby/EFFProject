from collections import Counter
from datetime import datetime
from pathlib import Path
from sklearn.datasets import load_iris
import pandas as pd
from evofuzzy import fuzzyclassifier
import tensorboardX

"""Script for testing the classifier by running it on the iris dataset.
"""

logdir = Path(f"tb_logs/iris/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
logdir.mkdir(parents=True, exist_ok=True)
tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))

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
classifier.fit(iris, y, classes, tensorboard_writer=tensorboard_writer)
tensorboard_writer.close()

print(f"Best Rule:  size = {len(classifier.best)}")
print("\n".join(classifier.best_strs))
print(
    "Final length of rules sets",
    dict(Counter(x.length for x in classifier.population_)),
)
