import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
from scipy.special import softmax
from matplotlib.axes import Axes
from argparse import ArgumentParser, Namespace

FONT_SIZE = 14
plt.rcParams["font.size"] = FONT_SIZE

RANDOM_BASELINE = "Random"
LABEL_SET = ["Yes", "No"]
YES_FORMS = ["Yes", " Yes"]
NO_FORMS = ["No", " No"]
METHODS = ["EAV", "EA", "WA", "DA", RANDOM_BASELINE]
# COLORS = ["lightgrey", "darkgrey", "dimgrey", "black"]
METHOD2NAME = {
    "DA": "Direct Ask",
    "WA": "Write then Ask",
    "EA": "Execute then Ask",
    "EAV": "Execute then Ask (Verb.)",
    RANDOM_BASELINE: RANDOM_BASELINE
}
NEGATIVE_INF = -100

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def calculate_yesprobs(row: dict) -> dict[str, np.ndarray]:
    """Extract token logits (in the LABEL_SET) and calculate "yes probability" for each METHOD."""
    method2yesprob = dict()
    for method in METHODS:
        if method == RANDOM_BASELINE:
            method2yesprob[method] = random.random()
        elif method in {"EAV"}:
            method2yesprob[method] = row[method]["need_hint"]
        else:
            logits = list()
            for label in LABEL_SET:
                try:
                    logit = row[method]["logprobs"][0].get(label, NEGATIVE_INF)
                except KeyError:
                    print("Logits not found. Use uniform distribution.")
                    logit = -0.7
                logits.append(logit)
            probs = softmax(np.array(logits))
            method2yesprob[method] = probs[0]
    return method2yesprob

def calc_method2yesprobs(data: list[dict]) -> dict[str, list]:
    method2yesprobs = {method: list() for method in METHODS}
    for row in data:
        yesprobs = calculate_yesprobs(row)
        for method in METHODS:
            method2yesprobs[method].append(yesprobs[method])
    return method2yesprobs

def get_method2triples(data: list[dict]) -> dict[str, list[tuple[int, int, float]]]:
    """The method2triples' format is as follows:
    
    {
        "method1": [(correct1, correct_w_hint1, yes_prob1), ...]
        "method2": [...]
    }
    """
    correct = [row["correct"] for row in data]
    correct_w_hint = [row["correct_w_hint"] for row in data]
    method2yesprobs = calc_method2yesprobs(data)
    method2triples = {method: sorted(list(zip(correct, correct_w_hint, method2yesprobs[method])), key=lambda t: t[2]) for method in METHODS}  # triple: (correct, correct_w_hint, yes_prob)    
    return method2triples

def plot_accuracy_curve(
    data: list[dict],
    method2triples: dict[str, list[tuple[int, int, float]]],
    ax: Axes
) -> None:
    sample_size = len(data)
    # Calculate correct count accumulation
    method2cnt = {method: {"cnt": list(), "cnt_w_hint": list()} for method in METHODS}
    for method in METHODS:
        cnt = 0
        cnt_w_hint = 0
        method2cnt[method]["cnt"].append(cnt)
        method2cnt[method]["cnt_w_hint"].append(cnt_w_hint)
        for i in range(sample_size):
            cnt += method2triples[method][i][0]
            method2cnt[method]["cnt"].append(cnt)
            cnt_w_hint += method2triples[method][-(i+1)][1]
            method2cnt[method]["cnt_w_hint"].append(cnt_w_hint)
        method2cnt[method]["cnt_w_hint"] = list(reversed(method2cnt[method]["cnt_w_hint"]))
    
    # Preprocessing for plotting
    method2deltaccs = {method: [((method2cnt[method]["cnt"][i] + method2cnt[method]["cnt_w_hint"][i] - method2cnt[method]["cnt"][-1]) / sample_size * 100) for i in range(sample_size + 1)] for method in METHODS}
    ask_ratios = [((sample_size - i) / sample_size * 100) for i in range(sample_size + 1)]

    # Ploting
    for i, method in enumerate(METHODS):
        if method == RANDOM_BASELINE:
            # Plot a diagonal dotted line
            total_cnt = method2cnt[method]["cnt"][-1]
            total_cnt_w_hint = method2cnt[method]["cnt_w_hint"][0]
            deltaaccs = [(total_cnt_w_hint - total_cnt) * ((sample_size - i) / sample_size) / sample_size * 100 for i in range(sample_size + 1)]
            ax.plot(ask_ratios, deltaaccs, label=METHOD2NAME[method], linestyle="--", color="black")
        else:
            ax.plot(ask_ratios, method2deltaccs[method], label=METHOD2NAME[method])  #, color=COLORS[i])
    ax.set_xlim(-5, 105)
    ax.set_xlabel("User Burden (%)", fontsize=FONT_SIZE + 2)
    ax.set_ylabel("ΔExecution Accuracy (%)", fontsize=FONT_SIZE + 2)
    ax.set_title("Delta-Burden Curve (DBC)", fontweight="bold")
    ax.legend()
    ax.grid(True)
    
    # Print accuracies / AUC
    acc = sum([row["correct"] for row in data]) / len(data) * 100
    acc_w_hint = sum([row["correct_w_hint"] for row in data]) / len(data) * 100
    print(f"Accuracy: {acc:.2f}%")
    print(f"Accuracy with Hint: {acc_w_hint:.2f}%")
    
    area_max = acc_w_hint - acc
    for method in METHODS:
        area_sum = 0
        for deltacc in method2deltaccs[method]:
            area_sum += deltacc
        auc = (area_sum / sample_size) / area_max
        if method == RANDOM_BASELINE:
            auc = 0.5
        print(f"{METHOD2NAME[method]}: AUC = {auc:.4f}")

def plot_pr_curve(
    data: list[dict],
    method2triples: dict[str, list[tuple[int, int, float]]],
    ax: Axes
) -> None:
    """Definition of precision and recall in the ActiveSQL context:
    
    Precision = #(NeedHint & Wrong) / #NeedHint
    Recall = #(NeedHint & Wrong) / #Wrong
    
    Abbreviation: nw_cnt = #(NeedHint & Wrong)
    """
    # Calculate statistics for precision and recall
    method2stats = {method: {"precision": [], "recall": []} for method in METHODS}  # from the highest threshold (not asking any hints) to the lowest
    nwrong = len(data) - sum([row["correct"] for row in data])  # originally wrong count
    for method in METHODS:
        nw_cnt = 0
        for i in range(-1, -len(data) - 1, -1):  # from the highest threshold to the lowest threshold
            nhint = -i
            if method2triples[method][i][0] == 0:  # wrong
                nw_cnt += 1
            method2stats[method]["precision"].append(nw_cnt / nhint * 100)
            method2stats[method]["recall"].append(nw_cnt / nwrong * 100)
    # Plotting
    for method in METHODS:
        if method == RANDOM_BASELINE:
            random_prec = nwrong / len(data) * 100
            ax.plot(method2stats[method]["recall"], [random_prec] * len(data), label=METHOD2NAME[method], linestyle="--", color="black")
        else:
            ax.plot(method2stats[method]["recall"], method2stats[method]["precision"], label=METHOD2NAME[method])
    ax.set_xlim(-5, 105)
    ax.set_xlabel("Recall of Asking for Support (%)", fontsize=FONT_SIZE + 2)
    ax.set_ylabel("Precision of Asking for Support (%)", fontsize=FONT_SIZE + 2)
    ax.set_title("PR Curve of Asking for Support", fontweight="bold")
    ax.legend()
    ax.grid(True)
    
    # Print AUPRC
    for method in METHODS:
        area_sum = 0
        for precision in method2stats[method]["precision"]:
            area_sum += precision
        auprc = area_sum / len(data)
        if method == RANDOM_BASELINE:
            auprc = random_prec
        print(f"{METHOD2NAME[method]}: AUPRC = {auprc / 100:.4f}")

def plot_fliprate_curve(
    data: list[dict],
    method2triples: dict[str, list[tuple[int, int, float]]],
    ax: Axes
) -> None:
    # Calculate flip rates for each method
    sample_size = len(data)
    method2fr = {method: list() for method in METHODS}  # fr == fliprate
    method2fr_cnt = {method: list() for method in METHODS}
    for method in METHODS:
        fr_cnt = 0  # flip count
        for i in range(-1, -sample_size - 1, -1):
            nhint = -i
            fr_cnt += (method2triples[method][i][1] - method2triples[method][i][0])
            method2fr[method].append(fr_cnt / nhint * 100)
            method2fr_cnt[method].append(fr_cnt)

    ask_ratios = [(i / sample_size * 100) for i in range(sample_size)]
    # Plotting
    for method in METHODS:
        if method == RANDOM_BASELINE:
            random_fliprate = method2fr[method][-1]
            ax.plot(ask_ratios, [random_fliprate] * len(data), label=METHOD2NAME[method], linestyle="--", color="black")
        else:
            ax.plot(ask_ratios, method2fr[method], label=METHOD2NAME[method])
    ax.set_xlim(-5, 105)
    ax.set_xlabel("User Burden (%)", fontsize=FONT_SIZE + 2)
    ax.set_ylabel("Flip Rate (%)", fontsize=FONT_SIZE + 2)
    ax.set_title("Flip Rate Curve (FRC)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True)

def plot_curves(data: list[dict], seed: int) -> None:
    random.seed(seed)
    N_PLOTS = 3
    _, axes = plt.subplots(nrows=1, ncols=N_PLOTS, figsize=(18, 5))
    method2triples = get_method2triples(data)
    plot_accuracy_curve(data, method2triples, axes[0])
    plot_pr_curve(data, method2triples, axes[1])
    plot_fliprate_curve(data, method2triples, axes[2])

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--methods",
        type=lambda x: x.split(),
        default="Random",
        help="Specify the methods to plot. If not specified, plot all methods."
    )
    parser.add_argument(
        "--length",
        type=int,
        default=int(1e8),
        help="The number of samples to plot. Default: 1e8 (all)."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./curves",
        help="The directory to save the plots."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the plot."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    for method in args.methods:
        assert method in set(METHODS)
    METHODS = args.methods  # for convenience, overwrite the global variable
    data = load_jsonl(args.jsonl)[:args.length]
    plot_curves(data, args.seed)
    plt.tight_layout()
    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.jsonl.stem}_curves.pdf"
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    plt.show()
