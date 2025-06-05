import pandas as pd

csv_path = 'UNSW_NB15.csv'

df = pd.read_csv(csv_path)

print("Header (column names):")
print(df.columns.tolist())

start_row = 240
end_row = 260
print(f"Rows {start_row} to {end_row}:")
print(df.iloc[start_row:end_row + 1])
