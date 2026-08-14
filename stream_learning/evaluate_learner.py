from pathlib import Path
import pandas as pd
from capymoa.stream import CSVStream
from io import StringIO
from capymoa.classifier import *
from capymoa.evaluation import ClassificationEvaluator
import jpype
from sohot.sohot_ensemble_layer import SoftHoeffdingTreeLayer


def load_data():
    BASE = Path(__file__).resolve().parent
    data_path = BASE / "data" / "dataset_model_ready.csv"
    data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date").sort_index()
    X = data.drop(columns=["label_up"])
    # Rename the column names (remove the "F18 ")
    X.columns = X.columns.str.split(n=1).str[-1]
    dates = X.index.tolist()
    y = data["label_up"].astype(int)
    return X, y, dates


# =============== MOA ===============
def get_moa_stream():
    X, y, dates = load_data()
    # Create MOA stream
    combined_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    csv_file = StringIO()
    combined_df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    stream = CSVStream(file=csv_file,
                       target="label_up",
                       categories={"label_up": ["0", "1"]},
                       name="CO2Stream",
                       length=len(combined_df))

    return stream, dates


def evaluate_moa_learner(stream, dates, seed, confidence=1e-1, save_every_n=100):
    stream.restart()
    schema = stream.get_schema()
    model = HoeffdingAdaptiveTree(schema, grace_period=5, confidence=confidence, random_seed=seed, nb_threshold=20)
    # model = EFDT(schema, grace_period=5, confidence=confidence, random_seed=seed)
    evaluator = ClassificationEvaluator(schema=schema)
    BASE = Path(__file__).resolve().parent
    results_folder = BASE / "stream_learner_results"
    results_folder.mkdir(parents=True, exist_ok=True)
    file_path = (results_folder / f"HAT_confidence_{confidence}.txt")
    # file_path = (results_folder / f"EFDT_confidence_{confidence}.txt")

    # Empty file at beginning
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")
    StringBuilder = jpype.JClass("java.lang.StringBuilder")
    step = 0
    while stream.has_more_instances():
        instance = stream.next_instance()
        step += 1
        y_pred = model.predict(instance)
        model.train(instance)
        evaluator.update(y_target_index=instance.y_index, y_pred_index=y_pred)
        if step % save_every_n == 0:
            sb = StringBuilder()
            model.moa_learner.getModelDescription(sb, 0)
            date = dates[step]
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n========================================\n")
                f.write(f"Date: {date}\n")
                f.write("========================================\n\n")
                f.write(str(sb))
                f.write(f"\n\nAccuracy: {evaluator.accuracy()}\n")

    return evaluator.accuracy()


# def evaluate_SGD_learner(stream, dates, seed, save_every_n=100):
#     stream.restart()
#     schema = stream.get_schema()
#     model = SGDClassifier(schema, random_seed=seed, loss="modified_huber")
#     evaluator = ClassificationEvaluator(schema=schema)
#     step = 0
#     while stream.has_more_instances():
#         instance = stream.next_instance()
#         step += 1
#         y_pred = model.predict(instance)
#         model.train(instance)
#         evaluator.update(y_target_index=instance.y_index, y_pred_index=y_pred)
#     print(evaluator.accuracy())
#     return evaluator.accuracy()

# =============== SoHoT ===============
def evaluate_torch_learner(stream, dates, seed, confidence=1e-1, tie_threshold=0.08, save_every_n=100):
    stream.restart()
    schema = stream.get_schema()
    model = SoftHoeffdingTreeLayer(schema=schema, seed=seed, trees_num=1, lr=0.01, split_confidence=confidence,
                                   grace_period=5, tie_threshold=tie_threshold)
    evaluator = ClassificationEvaluator(schema=schema)

    BASE = Path(__file__).resolve().parent
    results_folder = BASE / "stream_learner_results"
    results_folder.mkdir(parents=True, exist_ok=True)
    file_path = results_folder / f"SoHoT_confidence_{confidence}_tie_{tie_threshold}.txt"
    file_path.write_text("", encoding="utf-8")
    step = 0
    while stream.has_more_instances():
        instance = stream.next_instance()
        step += 1
        y_pred = model.predict(instance)
        model.train(instance)
        evaluator.update(y_target_index=instance.y_index, y_pred_index=y_pred)

        if step % save_every_n == 0:
            tree_text = model.plot_tree(instance=instance, tree_idx=0, text_format=True)
            date = dates[step]
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(
                    f"========================================\n"
                    f"Date: {date}\n"
                    f"========================================\n\n"
                    f"{tree_text}\n\n"
                    f"Accuracy: {evaluator.accuracy()}\n\n"
                )
    return evaluator.accuracy()


if __name__ == '__main__':
    stream, dates = get_moa_stream()
    seed = 42
    save_every_n = 20

    # evaluate_SGD_learner(stream=stream, dates=dates, seed=seed)

    for confidence in [0.01, 0.05, 0.1, 0.2, 0.3]:
        evaluate_moa_learner(stream=stream, dates=dates, seed=seed, confidence=confidence, save_every_n=save_every_n)
    for confidence in [0.1, 0.2]:
        for tie_threshold in [0.05, 0.06, 0.07, 0.08]:
            evaluate_torch_learner(stream=stream, dates=dates, seed=seed, confidence=confidence,
                                   tie_threshold=tie_threshold, save_every_n=save_every_n)
