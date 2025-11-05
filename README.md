# 🛰️ Fractional Cover (FC) Analysis Pipeline with MODIS

This document outlines the **end-to-end process** for conducting Fractional Cover (FC) analysis on **MODIS satellite imagery** using a High-Performance Computing (HPC) environment.  
The pipeline automates the calculation of fractional cover — the proportion of **exposed soil**, **photosynthetic vegetation (PV)**, and **non-photosynthetic vegetation (NPV)** within each satellite image pixel.

---

## 1. Pipeline Overview

This workflow is broken down into **four main stages**, seamlessly connecting cloud-based data acquisition with local HPC processing:

1. **GEE Imagery Export:** A Python script run in a Jupyter Notebook exports composite GeoTIFFs from Google Earth Engine (GEE) to a specified Google Drive folder.  
2. **Data Retrieval:** These GeoTIFF files are downloaded from Google Drive to the HPC using `rclone`.  
3. **FC Prediction:** A Python script runs on the HPC to process each individual GeoTIFF file, calculating its fractional cover.  
4. **Output Management:** The final fractional cover GeoTIFFs are stored in a designated directory on the HPC.

---

## 2. Prerequisites

Before running the pipeline, ensure the following are set up on your HPC environment.  
These steps must be performed on the **HPC's login node** before submitting any jobs.

### HPC Access
You have an active user account on the HPC with SSH access.

### Miniconda Installation & Environment

**Install Miniconda** (if not already installed) by running the following commands on the login node:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
