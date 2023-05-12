import pandas as pd

for i in range(1, 5):
    df = pd.read_csv("qna{}.csv".format(i), sep=",", engine="python", encoding="utf-8")

    df.to_csv("qna{}.tsv".format(i), sep="\t", encoding="utf-8")

    print("completed {i}th file conversion")
