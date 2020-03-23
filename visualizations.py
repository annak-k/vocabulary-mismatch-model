import matplotlib.pyplot as plt
import numpy as np


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
        "little dog": "cornflowerblue"
    }
    labels = sorted(list(label_to_score.keys()))
    scores = [label_to_score[label] for label in labels]
    if multiply_by_100:
        scores = [item * 100 for item in scores]
    labels = [" ".join(item.split("_")) for item in labels]
    if add_other:
        labels.append("other")
        scores.append(100 - np.sum(scores))
        
    def generate_label(percentage):
        percentage_i, curr_percentage = 0, scores[0]
        for i in range(1, len(scores)):
            if abs(scores[i] - percentage) < abs(curr_percentage - percentage):
                percentage_i, curr_percentage = i, scores[i]
        return f'{labels[percentage_i]}\n{percentage:.1f}%'
    
    plt.pie(
        scores,
        #labels=labels,
        autopct=generate_label,
        colors=[LABEL_TO_COLOR[label] for label in labels])
    plt.show()



