## Human Fall Detection On Jetson Xavier NX - Jetpack 5.1

### Prerequisite
+ Jetpack 5.1 (prefer if run with cuda)
+ Python 3.8

### Installation
#### **1. Clone the repo**
```
git clone git@github.com:tonible14012002/fall_detector.git
```

#### **2. Install Dependencies**
**For cpu devices**
```
pip install -r requirements.txt
```

**For jetson devices**

Install other dependencies
```
pip install -r requirements.txt
```
Uninstall `pytorch` and `torchvision` because previous command will install cpu only version of these packages.
```
pip uninstall torch torchvision
```
Installation for cuda enable devices required `pytorch`, `torchvision` to be compiled using cuda enabled.
Follow this [instruction](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to install compatible pytorch version .

If you are using jetpack, there might be preinstalled `pytorch` version.

### Run
See [Makefile](./makefile) for quick commands.

**Quick start**
```
python main.py -C '<path_to_your_video>'
```

Start with cpu device
```
python main.py -C '<path_to_your_video>' --device cpu 
```

### Contribution
...