# 3D_mapping

### Installation
1. Open3D installation

Make sure you have Python 3.8
```
pip install open3d        # or
pip install open3d-cpu    # Smaller CPU only wheel on x86_64 Linux (since v0.17+)
```
Upgrade pip to a version >=20.3 to install Open3D in Linux
```
pip install -U pip>=20.3
```
Note: In general, we recommend using a virtual environment or conda environment. Otherwise, depending on the configurations, you may need pip3 for Python 3, or the --user option to avoid permission issues.
```
pip3 install open3d
# or
pip install --user open3d
# or
python3 -m pip install --user open3d
```
For more information, refer http://www.open3d.org/docs/release/getting_started.html

### Dataset
Note: Store the input.pcd (or custom dataset) into main directory outside the script folder
Link to download the dataset: 
https://shorturl.at/rwyAI

### Run Demo
```
git clone https://github.com/...
python3 main.py
```
