import matplotlib.pyplot as plt
import numpy as np

LABELS_USED = []


def make_pie_plot(label_to_score, add_other=False, multiply_by_100=True):
    LABEL_TO_COLOR = {
        "dog": "lightskyblue",
        "dalmatian": "lightgreen",
        "spotted dog": "plum",
        "other": "lightgray",
        "small dog": "salmon",
        "pug": "moccasin",
        "doggy": "gold",
        "black and white dog": "pink",
        "little dog": "cornflowerblue",
        
        "banana": "lightskyblue",
        "soda": "lightgreen",
        "pop": "plum",
        "fries": "gold",
        "chips": "moccasin",
        "crisps": "salmon",
        "toque": "pink",
        "hat": "cornflowerblue",
        "winter hat": "orange",
        "cap": "yellowgreen"
    }
    labels = sorted([label for label, score in label_to_score.items()
                     if score >= 0.01])
    scores = [label_to_score[label] for label in labels]
    if multiply_by_100:
        scores = [item * 100 for item in scores]
    labels = [" ".join(item.split("_")) for item in labels]
    if add_other:
        labels.append("other")
        scores.append(100 - np.sum(scores))
        
    global LABELS_USED
    LABELS_USED = []
        
    def generate_label(percentage):
        global LABELS_USED
        percentage_i, curr_percentage = None, None
        for i in range(0, len(scores)):
            if i in LABELS_USED:
                continue
            if (percentage_i is None) or (
                    abs(scores[i] - percentage) < abs(curr_percentage - percentage)):
                percentage_i, curr_percentage = i, scores[i]
        LABELS_USED.append(percentage_i)
        return f'{labels[percentage_i]}\n{percentage:.1f}%'
    
    plt.pie(
        scores,
        autopct=generate_label,
        colors=[LABEL_TO_COLOR[label] for label in labels])
    plt.show()



