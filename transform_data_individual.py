import pandas as pd
import glob
import os
from datetime import datetime

# Mapping from extraction file
# AM- Date (CHAR_CONSIDERED_DATE)
# AN-DOW (extracted from date)
# Y-REVENUE (REVENUE column)
# AV-ROOM SOLD (NO_ROOMS)
# AX-ADR (CF_AVERAGE_ROOM_RATE)
# AY-Occupancy_Pct (CF_OCCUPANCY)
# REV PAR- OCC_PCT * ADR (calculated)

def transform_file(input_file, output_file, year):
    """Transform a single history_forecast CSV file"""
    print(f"Processing {input_file}...")

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Filter out rows with empty CHAR_CONSIDERED_DATE
    df = df[df['CHAR_CONSIDERED_DATE'].notna() & (df['CHAR_CONSIDERED_DATE'] != '')]

    # Create the transformed dataframe
    transformed = pd.DataFrame()

    # Parse the date from CHAR_CONSIDERED_DATE
    df['CONSIDERED_DATE'] = pd.to_datetime(df['CHAR_CONSIDERED_DATE'], format='%d.%m.%y %a')

    # Map columns according to extraction logic
    transformed['Date'] = df['CONSIDERED_DATE'].dt.strftime('%Y-%m-%d')
    transformed['DOW'] = df['CONSIDERED_DATE'].dt.strftime('%a')
    transformed['Rm Sold'] = df['NO_ROOMS'].astype(int)

    # Format amounts as numeric values with 2 decimal places
    transformed['Revenue'] = df['REVENUE'].round(2)
    transformed['ADR'] = df['CF_AVERAGE_ROOM_RATE'].round(2)
    transformed['Occupancy_Pct'] = df['CF_OCCUPANCY'].round(2)

    # Calculate RevPar = Occupancy_Pct * ADR
    revpar = df['CF_OCCUPANCY'] * df['CF_AVERAGE_ROOM_RATE'] / 100
    transformed['RevPar'] = revpar.round(2)

    # Save to individual file
    transformed.to_csv(output_file, index=False)
    print(f"  Created: {output_file}")
    print(f"  Records: {len(transformed)}")

    # Calculate total revenue for summary (extract numeric value)
    total_revenue = df['REVENUE'].sum()

    return len(transformed), total_revenue

def main():
    # Get all history_forecast CSV files
    input_files = sorted(glob.glob(r'C:\Users\reservations\Desktop\Data Science\history_forecast*.csv'))

    print(f"Found {len(input_files)} files to process\n")
    print("="*70)

    revenue_by_year = {}
    output_dir = r'C:\Users\reservations\Desktop\Data Science'

    for file in input_files:
        try:
            # Extract year from filename
            year = file.split('history_forecast')[1].split('.csv')[0]

            # Create output filename
            output_file = os.path.join(output_dir, f'historical_occupancy_{year}.csv')

            # Transform and save
            record_count, total_revenue = transform_file(file, output_file, year)
            revenue_by_year[year] = total_revenue
            print()

        except Exception as e:
            print(f"Error processing {file}: {e}")
            print()

    # Print revenue summary by year
    print("="*70)
    print("REVENUE SUMMARY BY YEAR (AED)")
    print("="*70)
    total_all_years = 0
    for year in sorted(revenue_by_year.keys()):
        revenue = revenue_by_year[year]
        total_all_years += revenue
        print(f"  {year}: AED {revenue:,.2f}")
    print("-"*70)
    print(f"  TOTAL (All Years): AED {total_all_years:,.2f}")
    print("="*70)

    print(f"\nAll files saved to: {output_dir}")
    print(f"File naming pattern: historical_occupancy_YYYY.csv")

if __name__ == "__main__":
    main()
