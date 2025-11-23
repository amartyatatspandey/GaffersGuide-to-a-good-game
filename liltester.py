import pandas as pd
df = pd.read_csv("opendata/data/matches/1886347/1886347_dynamic_events.csv")

print("third_id_start:", df["third_id_start"].unique())
print("channel_id_start:", df["channel_id_start"].unique())
print("channel_start:", df["channel_start"].unique())
print("x_start min/max:", df["x_start"].min(), df["x_start"].max())
print("y_start min/max:", df["y_start"].min(), df["y_start"].max())