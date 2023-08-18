# Building a 2D Vector Map of a garage for Autonomous Vehicle Parking

**Problem Statement**: To build a 3D map from the point cloud data of parking garage collected using RGB-D camera and translating it into a robust and reliable 2D vector map for autonomous vehicle parking.

```
**Download our Repository:**
```
git clone https://github.com/sr-bang/2D_vectormap.git
cd Script
```


## Installation
**1. Open3D**

Make sure you have Python 3.8 version.
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
For more information, refer [Open3D Documentation](http://www.open3d.org/docs/release/getting_started.html).

**2. DeepLSD Model**

### DeepLSD
Implementation of the paper [DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients](https://arxiv.org/abs/2212.07766), accepted at CVPR 2023. DeepLSD is a generic line detector that combines the robustness of deep learning with the accuracy of handcrafted detectors. It can be used to extract generic line segments from images in-the-wild, and is suitable for any task requiring high precision, such as homography estimation, visual localization, and 3D reconstruction. By predicting a line distance and angle fields, it can furthermore refine any existing line segments through an optimization.

#### Installation
First clone the repository and its submodules in Script folder:
```
git clone --recurse-submodules https://github.com/cvg/DeepLSD.git
cd DeepLSD
```

#### Quickstart install (for inference only)

To test the pre-trained model on your images, without the final line refinement, the following installation is sufficient:
```
bash quickstart_install.sh
```

#### Full install

Follow these instructions if you wish to re-train DeepLSD, evaluate it, or use the final step of line refinement.

Dependencies that need to be installed on your system:
- [OpenCV](https://opencv.org/)
- [GFlags](https://github.com/gflags/gflags)
- [GLog](https://github.com/google/glog)
- [Ceres 2.0.0](http://ceres-solver.org/)
- DeepLSD was successfully tested with GCC 9, Python 3.7, and CUDA 11. Other combinations may work as well.

Once these libraries are installed, you can proceed with the installation of the necessary requirements and third party libraries:
```
bash install.sh
```

This repo uses a base experiment folder (EXPER_PATH) containing the output of all trainings, and a base dataset path (DATA_PATH) containing all the evaluation and training datasets. You can set the path to these two folders in the file `deeplsd/settings.py`.

#### Usage
We provide two pre-trained models for DeepLSD: [deeplsd_wireframe.tar](https://www.polybox.ethz.ch/index.php/s/FQWGkH57UNTqlJZ) and [deeplsd_md.tar](https://www.polybox.ethz.ch/index.php/s/XVb30sUyuJttFys), trained respectively on the Wireframe and MegaDepth datasets. The former can be used for easy indoor datasets, while the latter is more generic and works outdoors and on more challenging scenes.

Download and extract the weights in <name_of_your_direc>/Script/Deeplsd/weights/

## Dependencies
```
cd ../../
pip install -r requirements.txt
```

## Dataset
Note: Store the input.pcd (or custom dataset) into main directory outside the script folder.

Download dataset from [here](https://shorturl.at/rwyAI). 

## Run Demo
```
cd Script/
python3 main.py
```

main.py gives you an output image. You can tweek the output image you obtained by changing arguments of main.py

Once you obtain desired image of cloud file
run the following python file 
```
python3 DLSD.py
```

You can observe the final output in the form of png and svg in output_img folder. You can also tweek the final output of parking lines detected by tweeking the arguments in DLSD.py


## References
1. https://github.com/cvg/DeepLSD
```bibtex
@InProceedings{Pautrat_2023_DeepLSD,
    author = {Pautrat, Rémi and Barath, Daniel and Larsson, Viktor and Oswald, Martin R. and Pollefeys, Marc},
    title = {DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    year = {2023},
}
```
2. https://github.com/salykovaa/ransac
