from datetime import datetime
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
from evofuzzy import FuzzyClassifier
import tensorboardX

"""Script for testing the classifier by running it on the iris dataset.
"""

TO_TENSORBOARD = True  # write results and stats to tensorboard?

data = load_iris()
cols = [c.replace(" ", "_").replace("_(cm)", "") for c in data.feature_names]
iris = pd.DataFrame(data.data, columns=cols)
y = pd.Series(data.target)

train_X, test_X, train_y, test_y = train_test_split(iris, y, test_size=50)


classes = {name: val for (name, val) in zip(data.target_names, range(3))}
antecendent_terms = {
    col: ["v.narrow", "narrow", "medium", "wide", "v.wide"]
    if "width" in col
    else ["v.short", "short", "medium", "long", "v.long"]
    for col in cols
}

if TO_TENSORBOARD:
    logdir = Path(f"tb_logs/iris/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logdir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
else:
    tensorboard_writer = None

classifier = FuzzyClassifier(
    population_size=20,
    elite_size=3,
    n_iter=20,
    mutation_prob=0.5,
    crossover_prob=0.5,
    min_tree_height=1,
    max_tree_height=3,
    min_rules=4,
    max_rules=6,
    whole_rule_prob=0.1,
    batch_size=20,
)
classifier.fit(
    train_X,
    train_y,
    classes,
    antecedent_terms=antecendent_terms,
    tensorboard_writer=tensorboard_writer,
)

print(f"Best Rule:  size = {len(classifier.best)}")
print(classifier.best_str)

predictions = classifier.predict(test_X)
confusion = pd.DataFrame(
    data=confusion_matrix(test_y, predictions),
    columns=data.target_names,
    index=data.target_names,
)
print(confusion)
if tensorboard_writer:
    tensorboard_writer.add_text("confusion", confusion.to_markdown())
    tensorboard_writer.close()
