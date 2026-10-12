import pandas as pd
from ch04.id3 import id3, predict
from ch04.info_metrics import information_gains, continuous_information_gains
from ch04.draw_tree import print_ascii_tree

def sec4_4_2():
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_8.csv")
    data = df.drop(columns="ID")

    scores = continuous_information_gains(data, "ELEVATION")
    tree = id3(data, information_gains, continuous_features={"ELEVATION"})
    print_ascii_tree(tree)

    label = predict(tree, {"STREAM": True, "SLOPE": "steep", "ELEVATION": 4500})
    print(label)  # conifer

if __name__ == '__main__':
    sec4_4_2()