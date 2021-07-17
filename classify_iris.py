from collections import Counter
from datetime import datetime
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import pandas as pd
from evofuzzy import fuzzyclassifier
import tensorboardX

"""Script for testing the classifier by running it on the iris dataset.
"""

TO_TENSORBOARD = True  # write results and stats to tensorboard?

data = load_iris()
cols = [c.replace(" ", "_").replace("_(cm)", "") for c in data.feature_names]
iris = pd.DataFrame(data.data, columns=cols)
y = pd.Series(data.target)

classes = {name: val for (name, val) in zip(data.target_names, range(3))}
antecendent_terms = {
    col: ["v.narrow", "narrow", "medium", "wide", "v.wide"]
    if "width" in col
    else ["v.short", "short", "medium", "long", "v.long"]
    for col in cols
}

for i in range(5):
    if TO_TENSORBOARD:
        logdir = Path(f"tb_logs/iris/{i}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        logdir.mkdir(parents=True, exist_ok=True)
        tensorboard_writer = tensorboardX.SummaryWriter(str(logdir))
    else:
        tensorboard_writer = None

    classifier = fuzzyclassifier.FuzzyClassifier(
        population_size=50,
        hall_of_fame_size=10,
        max_generation=20,
        mutation_prob=0.5,
        crossover_prob=0.5,
        min_tree_height=1,
        max_tree_height=3,
        max_rules=4,
        whole_rule_prob=0.2,
        tree_height_limit=5,
    )
    classifier.fit(
        iris,
        y,
        classes,
        antecedent_terms=antecendent_terms,
        tensorboard_writer=tensorboard_writer,
    )

    print(f"Best Rule:  size = {len(classifier.best)}")
    print("\n".join(classifier.best_strs))
    print(
        "Final length of rules sets",
        dict(Counter(x.length for x in classifier.population_)),
    )
    predictions = classifier.predict(iris)
    confusion = pd.DataFrame(
        data=confusion_matrix(y, predictions),
        columns=data.target_names,
        index=data.target_names,
    )
    print(confusion)
    if tensorboard_writer:
        tensorboard_writer.add_text("confusion", confusion.to_markdown())
        tensorboard_writer.close()
