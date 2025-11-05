# Copyright 2017 National Computational Infrastructure(NCI).
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import numpy as np
from osgeo import gdal
import scipy.optimize as opt
import scipy.ndimage
# netCDF4 is no longer needed for output, but kept if other parts of original code used it.
# import netCDF4
import json
import os
import datetime
import argparse
import time
import re # For regular expressions to parse GeoTIFF filenames
import glob # For finding files in a directory

print("Libraries imported successfully.")

# --- Configuration and Paths (UPDATE THESE FOR YOUR DATA) ---
# This should be the root directory where you downloaded all your COMBINED GEE MODIS GeoTIFFs.
# Assumes files are named like 'MODIS_Daily_Mosaic_YYYY_MM_DD.tif'.
# Example for HPC: '/sandisk1/u8022291/MODIS_FC/GEE_MODIS_Exports/'
COMBINED_MODIS_DATA_DIR = "/sandisk1/u8022291/MODIS_FC/DATA/" # <--- UPDATED for HPC!

# Path to the endmember file (endmembers_v6_20170831.txt)
# Ensure this file is in the same directory as this script, or provide its full path.
ENDMEMBERS_FILE = "/sandisk1/u8022291/MODIS_FC/endmembers_v6_20170831.txt" # <--- UPDATED for HPC!

# Path to the NetCDF metadata JSON file (nc_metadata.json)
# This file is still referenced for metadata, but GeoTIFFs have limited global metadata capacity.
# Some metadata might not transfer directly as with NetCDF.
METADATA_FILE = "/sandisk1/u8022291/MODIS_FC/nc_metadata.json" # <--- UPDATED for HPC!

# Directory where the fractional cover GeoTIFF outputs will be saved
OUTPUT_DIR = "/sandisk1/u8022291/MODIS_FC/fractional_cover_output/" # <--- UPDATED for HPC!

# Product version (e.g., '310' for version 3.1.0 as in the original metadata)
PRODUCT_VERSION = "310"

# Output NoData Value for GeoTIFFs
OUTPUT_NODATA_VALUE = 255 # Consistent with original code's fill value

print("\nConfiguration paths and parameters defined.")
print(f"Input Data Directory: {COMBINED_MODIS_DATA_DIR}")
print(f"Output Data Directory: {OUTPUT_DIR}")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- Helper Functions Definition ---

def get_geotiff_name(combined_geotiff_path, dest, ver, suffix=""):
    """
    Constructs the output GeoTIFF filename and ensures the destination directory exists.
    Assumes combined GeoTIFF filename format like 'MODIS_Daily_Mosaic_YYYY_MM_DD.tif'.
    """
    file_name = os.path.basename(combined_geotiff_path) # e.g., MODIS_Daily_Mosaic_2015_10_02.tif
    
    # Extract date from filename (e.g., 2015_10_02)
    match = re.search(r'(\d{4}_\d{2}_\d{2})', file_name)
    if not match:
        raise ValueError(f"Could not parse date from GeoTIFF filename: {file_name}")
    date_str = match.group(1).replace('_', '') # YYYYMMDD format

    # Construct the final GeoTIFF filename
    # Example output: MODISFC_20151002.tif
    base_filename = f"MODISFC_{suffix}.{date_str}_DEU.tif" 
    
    # The output path is now directly within the destination directory,
    # without creating a subfolder for each date.
    return os.path.join(dest, base_filename)


def input_mask(ds_combined, rows, cols):
    """
    Reads the MODIS MCD43A2 quality bands from the combined GeoTIFF.
    Expected: 'BRDF_Albedo_LandWaterType' as Band 8 and 'Snow_BRDF_Albedo' as Band 9.
    Pixels with water, snow, or low quality are excluded.
    """
    print(f"  Loading quality mask bands (Band 8, 9) from combined GeoTIFF.")
    try:
        # Read LandWaterType (expected to be Band 8 in combined GeoTIFF)
        pq_raw = ds_combined.GetRasterBand(8).ReadAsArray()
        # Read Snow_BRDF_Albedo (expected to be Band 9 in combined GeoTIFF)
        snow_pq_raw = ds_combined.GetRasterBand(9).ReadAsArray()

        # Bitwise AND with mask (0b1111) for LandWaterType (bits 0-3)
        mask_bits = 0b0000000000001111 # Corresponds to bits 0-3 for LandWaterType
        pq = pq_raw & mask_bits

        # Snow mask: 0 indicates no snow/ice, non-zero indicates snow/ice
        snow_mask = np.equal(snow_pq_raw, 0)

        # Combined mask: True for valid land/vegetation pixels (1, 2, or 4 for LandWaterType) AND no snow
        valid_land_types = np.logical_or(np.equal(pq, 1), np.logical_or(np.equal(pq, 2), np.equal(pq, 4)))
        
        final_mask = np.logical_and(snow_mask, valid_land_types)
        
        return final_mask.reshape(rows * cols) # Reshape to a 1D array for pixel-wise processing
    except Exception as e:
        print(f"  Error reading quality bands from combined GeoTIFF: {e}")
        print("  Returning a dummy mask (all True). PLEASE RESOLVE GDAL/BAND ISSUES FOR ACCURATE RESULTS.")
        return np.ones(rows * cols, dtype=bool)


def input_stack(ds_combined, rows, cols):
    """
    Reads the 7 MODIS Nadir BRDF-Adjusted Reflectance (NBAR) bands from the combined GeoTIFF
    (expected as Bands 1-7) and calculates 85 derivative features for each pixel.
    These derivatives include log transforms and band interactions.
    """
    print(f"  Loading spectral bands (Band 1-7) and computing features from combined GeoTIFF.")
    
    bands_data = []
    try:
        # Load each of the 7 reflectance bands (expected as Band 1 to 7 in combined GeoTIFF)
        for b in range(1, 8):
            # MODIS reflectance values are scaled by 0.0001. Apply this scaling.
            band_array = np.nan_to_num(ds_combined.GetRasterBand(b).ReadAsArray().astype(np.float32) * .0001)
            bands_data.append(band_array)
    except Exception as e:
        print(f"  Error reading reflectance bands from combined GeoTIFF: {e}")
        print("  Returning dummy band data. PLEASE RESOLVE GDAL/BAND ISSUES FOR ACCURATE RESULTS.")
        bands_data = [np.random.rand(rows, cols) * 0.5 for _ in range(7)] # Dummy reflectance 0-0.5

    # Initialize the 85-feature stack (rows*cols pixels x 85 features)
    bands_stack = np.empty((rows * cols, 85), dtype=np.float32)
    
    # 1. Add original 7 band values
    for b_idx in range(7):
        bands_stack[:, b_idx] = bands_data[b_idx].reshape(rows * cols)

    # Use a small epsilon to prevent log(0) or division by zero issues
    epsilon = 1e-9 
    
    # 2. Add log(band) for each of the 7 bands
    log_bands = np.log(bands_stack[:, :7] + epsilon)
    bands_stack[:, 7:14] = np.nan_to_num(log_bands)
    
    # 3. Add band * log(band) for each of the 7 bands
    bands_stack[:, 14:21] = np.nan_to_num(bands_stack[:, :7] * bands_stack[:, 7:14])

    ii = 21 # Starting index for interaction terms
    
    # 4. Add band * next_band for all unique pairs (21 pairs)
    for b1_idx in range(7):
        for b2_idx in range(b1_idx + 1, 7):
            bands_stack[:, ii] = np.nan_to_num(bands_stack[:, b1_idx] * bands_stack[:, b2_idx])
            ii += 1
    
    # 5. Add log(band) * log(next_band) for all unique pairs (21 pairs)
    for b1_idx in range(7):
        for b2_idx in range(b1_idx + 1, 7):
            bands_stack[:, ii] = np.nan_to_num(bands_stack[:, b1_idx + 7] * bands_stack[:, b2_idx + 7])
            ii += 1
    
    # 6. Add (next_band - band) / (next_band + band) for all unique pairs (21 pairs)
    for b1_idx in range(7):
        for b2_idx in range(b1_idx + 1, 7):
            numerator = bands_stack[:, b2_idx] - bands_stack[:, b1_idx]
            denominator = bands_stack[:, b2_idx] + bands_stack[:, b1_idx] + epsilon # Add epsilon to avoid div by zero
            bands_stack[:, ii] = np.nan_to_num(numerator / denominator)
            ii += 1
    
    # 7. Add a column of ones (intercept / sum-to-one constraint)
    bands_stack[:, -1] = np.ones(bands_stack.shape[0], dtype=bands_stack.dtype)
    
    return bands_stack


def members():
    """
    Loads the endmember matrix from the specified text file.
    These are the reference spectral signatures for PV, NPV, and Bare Soil.
    A sum-to-one constraint row is appended to the matrix.
    """
    print(f"  Loading endmembers from: {ENDMEMBERS_FILE}")
    try:
        # Load the 84x3 endmember matrix
        A = np.loadtxt(ENDMEMBERS_FILE)
    except IOError:
        print(f"  Error: Endmember file '{ENDMEMBERS_FILE}' not found.")
        print("  Please ensure this file is in the same directory or provide its full path.")
        print("  Using a dummy endmember matrix for demonstration. THIS WILL LEAD TO INCORRECT RESULTS.")
        # Create a dummy 84x3 matrix if the file is not found (for demonstration only)
        A = np.random.rand(84, 3)
        
    # As per original paper/code, add a row for sum-to-one constraint
    SumToOneWeight = 0.02
    ones = np.ones(A.shape[1]) * SumToOneWeight
    ones = ones.reshape(1, A.shape[1])
    
    A = np.concatenate((A, ones), axis=0).astype(np.float32)
    return A


def jp_superfunc(A, arr, mask):
    """
    Applies the Non-Negative Least Squares (NNLS) algorithm to unmix the spectral features
    into fractional cover percentages (Photosynthetic Vegetation, Non-Photosynthetic Vegetation, Bare Soil).
    """
    print("  Applying Non-Negative Least Squares (NNLS) unmixing...")
    
    # Initialize output array for PV, NPV, BS (3 bands)
    res = np.zeros((arr.shape[0], A.shape[1]), dtype=np.uint8) 
    
    for i in range(arr.shape[0]):
        # Apply the mask and validate reflectance values (0 to 1)
        if mask[i] and np.all(arr[i, :7] >= 0) and np.all(arr[i, :7] <= 1):
            try:
                fractions = opt.nnls(A, arr[i, :])[0]
                res[i, :] = (fractions.clip(0, 2.54) * 100).astype(np.uint8)
            except Exception as e:
                print(f"  Warning: NNLS failed for pixel {i}: {e}. Setting to fill value.")
                res[i, :] = np.ones(A.shape[1], dtype=np.uint8) * OUTPUT_NODATA_VALUE # Use global NoData
        else:
            res[i, :] = np.ones(A.shape[1], dtype=np.uint8) * OUTPUT_NODATA_VALUE # Use global NoData
            
    return res


def pack_data(source_geotiff_path, arr, dest, ver):
    """
    Saves the calculated fractional cover data (PV, NPV, BS) to a GeoTIFF file,
    including georeferencing derived from the source GeoTIFF.
    The output bands are ordered as: Band 1 (Red) = Bare Soil, Band 2 (Green) = PV, Band 3 (Blue) = NPV.
    """
    output_geotiff_path = get_geotiff_name(source_geotiff_path, dest, ver)
    print(f"  Saving fractional cover to GeoTIFF: {output_geotiff_path}")
    
    # Get georeferencing information from the original GeoTIFF file
    try:
        ds_source = gdal.Open(source_geotiff_path)
        if not ds_source:
            raise Exception(f"Could not open GeoTIFF file for georeferencing: {source_geotiff_path}")
        proj_wkt = ds_source.GetProjection() # WKT projection string
        geot = ds_source.GetGeoTransform()   # GeoTransform tuple
        x_size = ds_source.RasterXSize       # Raster X size (columns)
        y_size = ds_source.RasterYSize       # Raster Y size (rows)
        ds_source = None # Close the GDAL dataset after getting info
    except Exception as e:
        print(f"  Error reading georeferencing from GeoTIFF file {source_geotiff_path}: {e}")
        print("  Using dummy georeferencing. Output GeoTIFF may not be spatially referenced correctly.")
        proj_wkt = 'PROJCS["Sinusoidal",GEOGCS["GCS_Sphere",DATUM["D_Sphere",SPHEROID["Sphere",6371007.181,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.017453292519943295]],PROJECTION["Sinusoidal"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Semi_Major_Axis",6371007.181],UNIT["Meter",1.0]]'
        geot = [0.0, 463.3127165, 0.0, 1111950.519667, 0.0, -463.3127165]
        x_size, y_size = 2400, 2400
            
    # Reshape the 1D output array back to 2D (rows x cols x 3)
    # arr[:, :, 0] is PV, arr[:, :, 1] is NPV, arr[:, :, 2] is BS
    reshaped_arr = arr.reshape(y_size, x_size, 3) 

    # Create the output GeoTIFF dataset
    driver = gdal.GetDriverByName("GTiff")
    options = ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'BIGTIFF=IF_NEEDED']
    
    # Create the GeoTIFF with 3 bands (PV, NPV, BS)
    ds_output = driver.Create(output_geotiff_path, x_size, y_size, 3, gdal.GDT_Byte, options)
    
    if ds_output is None:
        raise Exception(f"Could not create output GeoTIFF: {output_geotiff_path}")

    # Set georeferencing
    ds_output.SetGeoTransform(geot)
    ds_output.SetProjection(proj_wkt)

    # --- Write fractional cover bands in the desired order ---
    # Band 1 (Red channel in viewer) = Bare Soil (BS)
    out_band_bs = ds_output.GetRasterBand(1)
    out_band_bs.WriteArray(reshaped_arr[:, :, 2]) # BS is at index 2 in reshaped_arr
    out_band_bs.SetNoDataValue(OUTPUT_NODATA_VALUE)
    out_band_bs.SetDescription("Bare Soil (%)")

    # Band 2 (Green channel in viewer) = Photosynthetic Vegetation (PV)
    out_band_pv = ds_output.GetRasterBand(2)
    out_band_pv.WriteArray(reshaped_arr[:, :, 0]) # PV is at index 0 in reshaped_arr
    out_band_pv.SetNoDataValue(OUTPUT_NODATA_VALUE)
    out_band_pv.SetDescription("Photosynthetic Vegetation (%)")

    # Band 3 (Blue channel in viewer) = Non-Photosynthetic Vegetation (NPV)
    out_band_npv = ds_output.GetRasterBand(3)
    out_band_npv.WriteArray(reshaped_arr[:, :, 1]) # NPV is at index 1 in reshaped_arr
    out_band_npv.SetNoDataValue(OUTPUT_NODATA_VALUE)
    out_band_npv.SetDescription("Non-Photosynthetic Vegetation (%)")

    # Flush data to disk and close the dataset
    ds_output.FlushCache()
    ds_output = None # Explicitly close the dataset

    print(f"  Successfully saved {output_geotiff_path}")


# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting Fractional Cover modeling process for multiple COMBINED GeoTIFFs...")
    total_start_time = time.time()

    # 1. Load Endmember Matrix (A) - This only needs to be done once
    A_matrix = members()
    print("Endmember matrix loaded.")

    # 2. Discover all COMBINED GeoTIFF files in the root directory
    # Assumes files are named like 'MODIS_Daily_Mosaic_YYYY_MM_DD.tif'.
    combined_geotiff_files = glob.glob(os.path.join(COMBINED_MODIS_DATA_DIR, "MODIS_Daily_Mosaic_*.tif")) # <--- UPDATED FILE DISCOVERY PATTERN
    combined_geotiff_files.sort() # Process files in chronological order

    if not combined_geotiff_files:
        print(f"No combined MODIS GeoTIFF files found in {COMBINED_MODIS_DATA_DIR}. Please check the path and file names.")
    else:
        print(f"Found {len(combined_geotiff_files)} combined GeoTIFF files to process.")

    processed_count = 0
    for combined_geotiff_path in combined_geotiff_files:
        file_processing_start_time = time.time()
        combined_geotiff_filename = os.path.basename(combined_geotiff_path)
        
        print(f"\n--- Processing {combined_geotiff_filename} ---")

        try:
            # Open the combined GeoTIFF once for both reflectance and mask data
            ds_combined = gdal.Open(combined_geotiff_path)
            if not ds_combined:
                raise Exception(f"Could not open combined GeoTIFF file: {combined_geotiff_path}")
            
            rows, cols = ds_combined.RasterYSize, ds_combined.RasterXSize

            # 1. Prepare Input Mask using the combined GeoTIFF
            # Assumes QA_LandWaterType is Band 8 and Snow_BRDF_Albedo is Band 9
            pixel_mask = input_mask(ds_combined, rows, cols)

            # 2. Prepare Input Spectral Stack using the combined GeoTIFF
            # Assumes Nadir_Reflectance_Band1-7 are Bands 1-7
            spectral_stack = input_stack(ds_combined, rows, cols)

            # Close the GDAL dataset after reading bands to release file handle
            ds_combined = None 

            # 3. Apply the Fractional Cover Algorithm
            fractional_cover_output = jp_superfunc(A_matrix, spectral_stack, pixel_mask)

            # 4. Save Output to GeoTIFF
            pack_data(combined_geotiff_path, fractional_cover_output, OUTPUT_DIR, PRODUCT_VERSION)
            processed_count += 1
            print(f"Finished processing {combined_geotiff_filename} in {time.time() - file_processing_start_time:.2f} seconds.")

        except Exception as e:
            print(f"An error occurred while processing {combined_geotiff_filename}: {e}")
            print(f"Skipping to the next file.")
            # Ensure ds_combined is closed if an error occurs after opening
            if 'ds_combined' in locals() and ds_combined is not None:
                ds_combined = None
            continue

    total_end_time = time.time()
    print(f"\n--- Batch Processing Summary ---")
    print(f"Successfully processed {processed_count} out of {len(combined_geotiff_files)} combined GeoTIFF files found.")
    print(f"Total Fractional Cover modeling completed in {total_end_time - total_start_time:.2f} seconds.")
    print(f"Outputs saved to: {OUTPUT_DIR}")

