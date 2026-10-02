import pandas as pd

main_label = "30cm"
sub_dfs = []

for i in range(1, 5):
    input(f"Copy Submeasurement {i} in Excel, then press ENTER here...")
    df = pd.read_clipboard(sep=r"\s+", header=None, names=[f"Pos_{i}", f"Power_{i}"])
    sub_dfs.append(df)
    print(f"-> Captured Submeasurement {i}!")

combined_df = pd.concat(sub_dfs, axis=1)

combined_df.to_csv(f"data2_{main_label}.csv", index=False)
print(f"\nSaved successfully to data2_{main_label}.csv!")