import pandas as pd

print("--- Inspecting Excel File Headers ---")
file_path = 'original_data_for_week1.xlsx'

for i in range(5):
    try:
        print(f"\n--- Reading with header={i} ---")
        df = pd.read_excel(file_path, header=i)
        print("Columns:", list(df.columns))
        # print("First 5 rows:")
        # print(df.head())
    except Exception as e:
        print(f"Could not read with header={i}. Error: {e}")

