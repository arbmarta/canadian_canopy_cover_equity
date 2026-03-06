import pandas as pd
import glob
import os

# Subdirectory
sub_directory = 'road_network'

if sub_directory == 'road_network':
    # Directory containing the CSV files
    directory = "../data/road_network/"

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "canopy_cover_road_network_da_batch_*.csv"))

elif sub_directory == 'dissemination_areas':
    # Directory containing the CSV files
    directory = "../data/dissemination_areas/"

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "canopy_cover_dissemination_areas_batch_*.csv"))

if not csv_files:
    print("No CSV files found in the directory.")
else:
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")

    # Read and concatenate all CSVs
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    print(f"\nCombined {len(csv_files)} files → {len(df):,} rows | Columns: {list(df.columns)}")

    # Load the csduid relational table and left join onto dissemination areas
    relational_path = "../data/csduid_dauid_relational_datatable.csv"
    relational_df = pd.read_csv(relational_path)

    print(f"\nRelational table: {len(relational_df):,} rows | Columns: {list(relational_df.columns)}")

    # Left join — keeping all rows in dissemination_areas
    # Update the join key below if your column names differ
    df = df.merge(relational_df, on="DAUID", how="left")

    print(f"After join: {len(df):,} rows | Columns: {list(df.columns)}")

    # Save the result
    if sub_directory == 'road_network':
        output_path = "../data/road_network/road_network.csv"
    elif sub_directory == 'dissemination_areas':
        output_path = "../data/dissemination_areas/dissemination_areas.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDone! Saved → {output_path}")