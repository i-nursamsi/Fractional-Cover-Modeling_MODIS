````markdown
# Fractional Cover (FC) Analysis Pipeline with MODIS

This document outlines the **end-to-end process** for conducting Fractional Cover (FC) analysis on **MODIS satellite imagery** using a High-Performance Computing (HPC) environment.  
The pipeline automates the calculation of fractional cover — the proportion of **exposed soil**, **photosynthetic vegetation (PV)**, and **non-photosynthetic vegetation (NPV)** within each satellite image pixel.

---

## Table of Contents

- [1. Pipeline Overview](#1-pipeline-overview)
- [2. Prerequisites](#2-prerequisites)
  - [2.1 HPC Access](#21-hpc-access)
  - [2.2 Miniconda Installation & Environment](#22-miniconda-installation--environment)
  - [2.3 rclone Installation & Configuration](#23-rclone-installation--configuration)
- [3. GEE Imagery Export (Local Python)](#3-gee-imagery-export-local-python)
- [4. Data Retrieval & FC Prediction on HPC](#4-data-retrieval--fc-prediction-on-hpc)
  - [4.1 PBS Script: run_modis_fc.sh](#41-pbs-script-run_modis_fcsh)
  - [4.2 How It Works](#42-how-it-works)
- [5. Final Output](#5-final-output)

---

## 1. Pipeline Overview

This workflow is broken down into **four main stages**, connecting cloud-based data acquisition with HPC processing:

1. **GEE Imagery Export:** Python script in Jupyter Notebook exports GeoTIFFs from Google Earth Engine (GEE) to Google Drive.  
2. **Data Retrieval:** GeoTIFF files are downloaded from Google Drive to the HPC using `rclone`.  
3. **FC Prediction:** Python script on HPC calculates fractional cover for each GeoTIFF.  
4. **Output Management:** Final fractional cover GeoTIFFs are stored in a designated HPC directory.

---

## 2. Prerequisites

Before running the pipeline, ensure the following are set up on your HPC environment.  
These steps must be performed on the **HPC login node**.

### 2.1 HPC Access

- Active HPC account with SSH access.

### 2.2 Miniconda Installation & Environment

**Install Miniconda**:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
````

> **Note:** Install to your home directory, e.g., `/home/u8022291/miniconda3`.

**Create Conda Environment (`fc_env`)**:

```bash
source ~/miniconda3/bin/activate
conda env create -f /home/u8022291/Automating_Workflow/environment.yml
conda activate fc_env
```

### 2.3 rclone Installation & Configuration

* Install `rclone` on HPC.
* Configure a **Google Drive remote** (e.g., `gdrive`).
* Run `rclone config` and follow interactive authentication.

---

## 3. GEE Imagery Export (Local Python)

Performed in a **local Python environment** with Google Earth Engine access.

* **Tool Used:** Python script using `earthengine-api`.
* **Input:** MODIS MCD43A4 (reflectance) and MCD43A2 (quality) imagery.
* **Process:** Filter data by ROI and date range; combine **7 reflectance bands + 2 quality bands** per acquisition date.
* **Output:** Combined GeoTIFFs, `MODIS_Combined_YYYY_MM_DD.tif`, saved to Google Drive.

> **Note:** Each export creates a separate task in the GEE Code Editor "Tasks" tab. You must manually start each export.

---

## 4. Data Retrieval & FC Prediction on HPC

Executed via a **PBS job script**, combining data transfer and analysis.

### 4.1 PBS Script: `run_modis_fc.sh`

Save in your `WORKFLOW_DIR`:

```bash
#!/bin/bash

#PBS -A hpcusers
#PBS -N fc_pipeline_MODIS
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -l mem=64gb
#PBS -j oe
#PBS -o pipelineMODIS.log

WORKFLOW_DIR="/sandisk1/u8022291/MODIS_FC"

# Proxy Configuration
export ftp_proxy="http://139.86.9.82:8080/"
export http_proxy="http://139.86.9.82:8080/"
export https_proxy="http://139.86.9.82:8080/"
export no_proxy="localhost,127.0.0.1"
echo "Proxy configured."

# Rclone Data Retrieval
RCLONE_CONFIG_NAME="gdrive"
RCLONE_SOURCE_PATH="GEE_MODIS_Exports_DEU_Daily"
HPC_DESTINATION_PATH="${WORKFLOW_DIR}/MODIS_Daily_Composites"

mkdir -p "${HPC_DESTINATION_PATH}"

rclone copy "$RCLONE_CONFIG_NAME:$RCLONE_SOURCE_PATH" "$HPC_DESTINATION_PATH" --progress --drive-skip-gdocs --copy-links

if [ $? -ne 0 ]; then
    echo "ERROR: MODIS data retrieval failed."
    exit 1
fi
echo "Data retrieval completed."

# Activate Conda Environment
source /home/u8022291/miniconda3/etc/profile.d/conda.sh
conda activate fc_env

cd "$WORKFLOW_DIR" || { echo "ERROR: Cannot change to WORKFLOW_DIR."; exit 1; }

echo "Starting FC prediction..."
/home/u8022291/miniconda3/envs/fc_env/bin/python MODIS_FC_Model.py

if [ $? -ne 0 ]; then
    echo "ERROR: FC prediction failed."
    exit 1
fi

echo "FC prediction completed."
conda deactivate
```

### 4.2 How It Works

1. Copies GeoTIFFs from Google Drive to HPC using `rclone`.
2. Activates Conda environment and runs `MODIS_FC_Model.py`.
3. Applies **quality mask** and **Non-Negative Least Squares (NNLS)** unmixing.
4. Outputs **three-band GeoTIFFs** (Bare Soil, PV, NPV) to HPC output directory.

---

## 5. Final Output

* Three-band GeoTIFFs are located in **OUTPUT_DIR**, e.g., `/sandisk1/u8022291/MODIS_FC/fractional_cover_output/`.
* Can be opened in any standard GIS software.

```
