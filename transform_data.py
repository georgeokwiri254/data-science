import pandas as pd
import glob
from datetime import datetime

# Mapping from extraction file
# AM- Date (CHAR_CONSIDERED_DATE)
# AN-DOW (extracted from date)
# Y-REVENUE (REVENUE column)
# AV-ROOM SOLD (NO_ROOMS)
# AX-ADR (CF_AVERAGE_ROOM_RATE)
# AY-Occupancy_Pct (CF_OCCUPANCY)
# REV PAR- OCC_PCT * ADR (calculated)

def transform_file(input_file):
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
    transformed['Rm Sold'] = df['NO_ROOMS']
    transformed['Revenue'] = df['REVENUE']
    transformed['ADR'] = df['CF_AVERAGE_ROOM_RATE']
    transformed['Occupancy_Pct'] = df['CF_OCCUPANCY']

    # Calculate RevPar = Occupancy_Pct * ADR
    transformed['RevPar'] = transformed['Occupancy_Pct'] * transformed['ADR'] / 100

    return transformed

def main():
    # Get all history_forecast CSV files
    input_files = sorted(glob.glob(r'C:\Users\reservations\Desktop\Data Science\history_forecast*.csv'))

    print(f"Found {len(input_files)} files to process")

    # Process all files and concatenate
    all_data = []
    revenue_by_year = {}

    for file in input_files:
        try:
            df = transform_file(file)
            all_data.append(df)

            # Extract year from filename for revenue summary
            year = file.split('history_forecast')[1].split('.csv')[0]
            total_revenue = df['Revenue'].sum()
            revenue_by_year[year] = total_revenue
            print(f"  Year {year}: {len(df)} records, Total Revenue: ${total_revenue:,.2f}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by date
        combined_df = combined_df.sort_values('Date')

        # Save to output file
        output_file = r'C:\Users\reservations\Desktop\Data Science\historical_occupancy_2024_040917.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully created {output_file}")
        print(f"  Total records: {len(combined_df)}")
        print(f"  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

        # Print revenue summary by year
        print("\n" + "="*60)
        print("REVENUE SUMMARY BY YEAR")
        print("="*60)
        total_all_years = 0
        for year in sorted(revenue_by_year.keys()):
            revenue = revenue_by_year[year]
            total_all_years += revenue
            print(f"  {year}: ${revenue:,.2f}")
        print("-"*60)
        print(f"  TOTAL (All Years): ${total_all_years:,.2f}")
        print("="*60)
    else:
        print("No data to process")

if __name__ == "__main__":
    main()
