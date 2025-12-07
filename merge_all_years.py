import pandas as pd
import glob
import os

def merge_all_files():
    """Merge all historical occupancy files from 2009 to 2025 chronologically"""

    print("="*70)
    print("MERGING HISTORICAL OCCUPANCY DATA (2009-2025)")
    print("="*70)

    data_dir = r'C:\Users\reservations\Desktop\Data Science'

    # List of years to process in order
    years = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
             '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']

    all_data = []
    total_records = 0
    revenue_by_year = {}

    for year in years:
        if year == '2025':
            # Special handling for 2025 file with different name
            file_path = os.path.join(data_dir, 'occupancy_2025.csv')
        elif year == '2022':
            file_path = os.path.join(data_dir, 'historical_occupancy_2022_040906.csv')
        elif year == '2023':
            file_path = os.path.join(data_dir, 'historical_occupancy_2023_040911.csv')
        else:
            file_path = os.path.join(data_dir, f'historical_occupancy_{year}.csv')

        if os.path.exists(file_path):
            print(f"\nProcessing {year}...")
            df = pd.read_csv(file_path)

            # Standardize column names
            # Expected columns: Date, DOW, Rm Sold, Revenue, ADR, Occupancy_Pct, RevPar
            if year == '2025':
                # Rename 2025 columns to match standard format
                df = df.rename(columns={
                    'Occ%': 'Occupancy_Pct'
                })
            elif year in ['2022', '2023']:
                # 2022 and 2023 have columns in different order but same names
                # Just reorder them to standard format
                pass

            # Ensure standard column order
            standard_columns = ['Date', 'DOW', 'Rm Sold', 'Revenue', 'ADR', 'Occupancy_Pct', 'RevPar']
            df = df[standard_columns]

            # Round all numeric columns to 2 decimal places
            numeric_cols = ['Rm Sold', 'Revenue', 'ADR', 'Occupancy_Pct', 'RevPar']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

            # Calculate total revenue for this year
            year_revenue = df['Revenue'].sum()
            revenue_by_year[year] = year_revenue

            print(f"  Records: {len(df)}")
            print(f"  Total Revenue: {year_revenue:,.2f}")

            all_data.append(df)
            total_records += len(df)
        else:
            print(f"\nWarning: File not found for {year}: {file_path}")

    # Combine all data
    if all_data:
        print("\n" + "="*70)
        print("COMBINING DATA...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by date to ensure chronological order
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df = combined_df.sort_values('Date')

        # Convert date back to string format
        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')

        # Save merged file
        output_file = os.path.join(data_dir, 'historical_occupancy_2009_2025_merged.csv')
        combined_df.to_csv(output_file, index=False)

        print(f"\nSuccessfully created: {output_file}")
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

        # Print revenue summary
        print("\n" + "="*70)
        print("REVENUE SUMMARY BY YEAR (AED)")
        print("="*70)
        total_revenue = 0
        for year in years:
            if year in revenue_by_year:
                revenue = revenue_by_year[year]
                total_revenue += revenue
                print(f"  {year}: {revenue:,.2f}")
        print("-"*70)
        print(f"  TOTAL (All Years): {total_revenue:,.2f}")
        print("="*70)

    else:
        print("\nNo data to merge!")

if __name__ == "__main__":
    merge_all_files()
