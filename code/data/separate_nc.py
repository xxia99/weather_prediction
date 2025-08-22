# Re-run after kernel reset
import xarray as xr

# Reload full dataset from uploaded file
ds = xr.open_dataset("train.nc")

# Take first 10 days = 240 hours
# subset_ds = ds.isel(timestamp=slice(0, 24*50))
# subset_ds = ds.isel(timestamp=slice(24*50, 24*50+24))
subset_ds = ds.isel(timestamp=slice(24*50, 24*50+24*2))

# Save to new .nc file
# output_path = "train_50days.nc"
# output_path = "valid_1day.nc"
output_path = "valid_2days.nc"
subset_ds.to_netcdf(output_path)