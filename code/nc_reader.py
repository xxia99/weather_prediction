import xarray as xr
import pandas as pd

# Load NetCDF file
ds = xr.open_dataset('data/test_24h_demo.nc')

# Convert to DataFrame (flattening if needed)
df = ds.to_dataframe().reset_index()

# Save to Excel
df.to_excel('output.xlsx', index=False)