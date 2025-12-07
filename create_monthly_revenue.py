import pandas as pd
import os

def create_monthly_revenue_files():
    """Create monthly revenue summary files for each year"""

    print("="*70)
    print("CREATING MONTHLY REVENUE SUMMARIES BY YEAR")
    print("="*70)

    data_dir = r'C:\Users\reservations\Desktop\Data Science'
    input_file = os.path.join(data_dir, 'historical_occupancy_2009_2025_merged.csv')

    # Hotel configuration
    TOTAL_ROOMS = 339

    # Read the merged file
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file)

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')

    # Get unique years
    years = sorted(df['Year'].unique())

    print(f"Processing {len(years)} years: {min(years)} to {max(years)}\n")

    # Create monthly summaries for each year
    for year in years:
        year_data = df[df['Year'] == year].copy()

        # Group by month and aggregate
        monthly = year_data.groupby(['Month', 'Month_Name']).agg({
            'Revenue': 'sum',
            'Rm Sold': 'sum',
            'Date': 'count'  # Number of days
        }).reset_index()

        # Rename columns
        monthly = monthly.rename(columns={
            'Date': 'Days',
            'Revenue': 'Total_Revenue',
            'Rm Sold': 'Total_Rooms_Sold'
        })

        # Calculate correct ADR: Total Revenue / Total Rooms Sold
        monthly['ADR'] = (monthly['Total_Revenue'] / monthly['Total_Rooms_Sold']).round(2)

        # Calculate correct Occupancy %: (Rooms Sold / (339 * Days in Month)) * 100
        monthly['Occupancy_Pct'] = ((monthly['Total_Rooms_Sold'] / (TOTAL_ROOMS * monthly['Days'])) * 100).round(2)

        # Calculate correct RevPar: ADR * (Occupancy % / 100)
        monthly['RevPar'] = (monthly['ADR'] * (monthly['Occupancy_Pct'] / 100)).round(2)

        # Format values
        monthly['Total_Revenue'] = monthly['Total_Revenue'].round(2)
        monthly['Total_Rooms_Sold'] = monthly['Total_Rooms_Sold'].astype(int)

        # Reorder columns
        monthly = monthly[['Month', 'Month_Name', 'Days', 'Total_Revenue',
                          'Total_Rooms_Sold', 'ADR', 'Occupancy_Pct', 'RevPar']]

        # Save to file
        output_file = os.path.join(data_dir, f'monthly_revenue_{year}.csv')
        monthly.to_csv(output_file, index=False)

        total_revenue = monthly['Total_Revenue'].sum()
        print(f"  {year}: {output_file}")
        print(f"         Total Revenue: AED {total_revenue:,.2f}")
        print(f"         Months: {len(monthly)}")

    # Create a combined summary file with all years
    print("\n" + "="*70)
    print("CREATING COMBINED MONTHLY SUMMARY FILE")
    print("="*70)

    # Group by Year and Month
    monthly_all = df.groupby(['Year', 'Month', 'Month_Name']).agg({
        'Revenue': 'sum',
        'Rm Sold': 'sum',
        'Date': 'count'
    }).reset_index()

    # Rename columns
    monthly_all = monthly_all.rename(columns={
        'Date': 'Days',
        'Revenue': 'Total_Revenue',
        'Rm Sold': 'Total_Rooms_Sold'
    })

    # Calculate correct ADR: Total Revenue / Total Rooms Sold
    monthly_all['ADR'] = (monthly_all['Total_Revenue'] / monthly_all['Total_Rooms_Sold']).round(2)

    # Calculate correct Occupancy %: (Rooms Sold / (339 * Days in Month)) * 100
    monthly_all['Occupancy_Pct'] = ((monthly_all['Total_Rooms_Sold'] / (TOTAL_ROOMS * monthly_all['Days'])) * 100).round(2)

    # Calculate correct RevPar: ADR * (Occupancy % / 100)
    monthly_all['RevPar'] = (monthly_all['ADR'] * (monthly_all['Occupancy_Pct'] / 100)).round(2)

    # Format values
    monthly_all['Total_Revenue'] = monthly_all['Total_Revenue'].round(2)
    monthly_all['Total_Rooms_Sold'] = monthly_all['Total_Rooms_Sold'].astype(int)

    # Reorder columns
    monthly_all = monthly_all[['Year', 'Month', 'Month_Name', 'Days', 'Total_Revenue',
                               'Total_Rooms_Sold', 'ADR', 'Occupancy_Pct', 'RevPar']]

    # Save combined file
    combined_file = os.path.join(data_dir, 'monthly_revenue_2009_2025_all.csv')
    monthly_all.to_csv(combined_file, index=False)

    print(f"\nCombined file created: {combined_file}")
    print(f"Total records: {len(monthly_all)} (months across all years)")

    # Print summary statistics
    print("\n" + "="*70)
    print("YEARLY REVENUE SUMMARY")
    print("="*70)
    yearly_summary = monthly_all.groupby('Year')['Total_Revenue'].sum()
    total_all = 0
    for year, revenue in yearly_summary.items():
        total_all += revenue
        print(f"  {year}: AED {revenue:,.2f}")
    print("-"*70)
    print(f"  TOTAL: AED {total_all:,.2f}")
    print("="*70)

    print(f"\nAll files saved to: {data_dir}")

if __name__ == "__main__":
    create_monthly_revenue_files()
