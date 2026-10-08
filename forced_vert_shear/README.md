# Zarr download instructions for vertical shear data archived on Constellation

**The simplest method for visualizing and downloading subvolumes of this dataset is by visiting the [interactive download portal: https://strata-turbulence.ca/shear/](https://strata-turbulence.ca/shear/)**

## Structure of archive

**Location:** Archive stored under [doi.org/10.13139/OLCF/3409014](https://doi.ccs.ornl.gov/dataset/d6fa913a-184d-5aef-a81a-b7cfc47f5112).

**Format:**
- Thirteen folders labelled `R#P#.zarr`, corresponding to different simulations (see table below). Simulations are size $N_x \times N_x /2 \times N_x /4$ (see further details under main Constellation README [doi.org/10.13139/OLCF/3409014](https://doi.ccs.ornl.gov/dataset/d6fa913a-184d-5aef-a81a-b7cfc47f5112)).
- Each `R#P#.zarr` contains six variables (`u,v,w,r,ee,chi`)
- Each variable contains a folder labelled `0` (and potentially further folders `1`, `2`, ...). These numbers represent sparsing levels. `0` corresponds to full resolution data, `1` corresponds to every 2nd point, `2` corresponds to every 4th point etc. Sparsing levels are chosen such that the maximum sparsing level per simulation gives a resolution of roughly $N_x<2000$. 
- Each variable, including at every sparsed level, is stored in Zarr (v3) format, detailed in the zarr.json file associated with each variable. Each field is split into separate subvolume (chunks) of size $128\times128\times128$ gridpoints (8 MiB each). To reduce the number of files, $6\times6\times6$ groups of chunks are then combined together in larger "shards" roughly of size 1.7 GiB.
- The scripts below are used to exploit the Zarr format, meaning that subvolumes can easily be extracted without needing to download the whole file. The smallest downloadable unit is a $128\times128\times128$ gridpoint chunk (8 MiB).


| Case  |     Full resolution Nx | Sparsing levels | Coarsest Nx |
|-------|-------:|-----------------|---------------:|
| R1P1  |   1536 | 0             | 1536 |
| R1P7  |   3072 | 0,1           | 1536 |
| R1P50 |   8000 | 0,1,2         | 2000 |
| R4P1  |   2048 | 0             | 2048 |
| R4P7  |   5120 | 0,1,2         | 1280 |
| R4P50 |  16000 | 0,1,2,3       | 2000 |
| R6P1  |   3072 | 0,1           | 1536 |
| R6P7  |  12288 | 0,1,2,3       | 1536 |
| R6P50 |  24000 | 0,1,2,3,4     | 1500 |
| R8P1  |   8736 | 0,1,2,3       | 1092 |
| R8P7  |  23040 | 0,1,2,3,4     | 1440 |
| R10P1 |  12288 | 0,1,2,3       | 1536 |
| R10P7 |  31680 | 0,1,2,3,4     | 1980 |



## Accessing the data


### Option 1 (recommended): Interactive subcube download (via live-stream HTTPS or Globus command line interface)

Visit **[interactive download portal: https://strata-turbulence.ca/shear/](https://strata-turbulence.ca/shear/)**.

User can specify x,y,z ranges from any simulation, variable, sparsing level, and download requested subvolume on demand to local computer.


### Option 2: Locally-run Jupyter notebook to download subvolumes

**Instructions.**
On local machine:
1. Ensure you are using Python version >= 3.11. Check with `python3 --version`.
2. Create Python environment using `requirements.txt` (see Appendix B).
3. Run `zarr_download.ipynb` (using IDE such as VS code, or `jupyter notebook zarr_download.ipynb` etc), and follow instructions within.


### Option 3: Download full variable files using Globus GUI 

Note: cannot download subchunks this way (see options 1 or 2 instead)

**Step 1: Download**
- Install Globus Connect Personal on your local machine (see Appendix A), or use Globus endpoint already set up on your cluster.
- [Login to Globus](https://app.globus.org/dashboard). Under File Manager, navigate to Collection: `OLCF DOI-DOWNLOADS` and Path `/gen101/world-shared/doi-data/OLCF/202609/10.13139_OLCF_3409014/`
- Choose folder to download (folder contains all the chunks required to reconstruct full field). For example: `zarr/R4P50.zarr/r/3` (R4P50 simulation, r variable, sparse level 3 (every 2^3 = 8th point))
- Download that folder, which must include both a `c` folder (contains all chunks) and `zarr.json` file (description of chunking), to your desired machine. If you get confused about what variable you are downloading after it's on your local machine, check the `attributes` field of the associated zarr.json file, which will list simulation, variable and level details.


**Step 2: Reconstruct**
- Activate specified Python environment (see Appendix B)
- Navigate to folder where Globus downloaded folder (in above example, you will have a folder called `3` on your local machine). 
- Run the `rebuild.py` script to reconstruct the file from the zarr components and save as .npy file. For option 2, you downloaded the full array (all chunks) and so the relevant command is:
`python rebuild.py 3` where `3` is the name of downloaded folder. The output file will be automatically named `sim_var_level.npy` based on attributes in zarr.json file.




## Appendix A: Globus Connect Personal 
A Globus Connect Personal (GCP) endpoint can be setup for free on your local machine (many research clusters already have an established Globus endpoint). This allows for the easy transfer of large amounts of data, including automatic checkpointing and restarts if a download is inturrputed. 

You can [install the Globus Connect Personal client here](https://www.globus.org/globus-connect-personal).

You will need to allow GCP to access the folder you wish to download to on your local machine. Once installed, go to GCP preferences and then info to obtain your Endpoint ID.


## Appendix B: Python Environment 
Follow these steps to install the required Python environment:

1. Navigate to the folder where you want the environment:
`cd /path/to/your/env/folder`

2. Create the virtual environment (creates a folder named "venv" here): `python3 -m venv my_env_name`

3. Activate the env:
   - `source my_env_name/bin/activate` # macOS/Linux
   - `my_env_name\Scripts\activate`    # Windows

4. Install dependencies from requirements.txt: `pip install -r requirements.txt`
