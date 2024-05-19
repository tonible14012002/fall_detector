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

### Jetson Xavier NX setup
**Maximize Power Model**
Jetpack Version
```
> show nvidia-jetpack
# ---- Results -----
Package: nvidia-jetpack
Version: 5.0.2-b231
Priority: standard
Section: metapackages
Maintainer: NVIDIA Corporation
Installed-Size: 199 kB
Depends: nvidia-jetpack-runtime (= 5.0.2-b231), nvidia-jetpack-dev (= 5.0.2-b231)
Homepage: http://developer.nvidia.com/jetson
Download-Size: 29,3 kB
APT-Sources: https://repo.download.nvidia.com/jetson/common r35.1/main arm64 Packages
Description: NVIDIA Jetpack Meta Package
```
Checking supported power models 
```
> nvpmodel -p --verbose | grep POWER_MODEL
# ---- Result ----
NVPM VERB: POWER_MODEL: ID=0 NAME=MODE_15W_2CORE
NVPM VERB: POWER_MODEL: ID=1 NAME=MODE_15W_4CORE
NVPM VERB: POWER_MODEL: ID=2 NAME=MODE_15W_6CORE
NVPM VERB: POWER_MODEL: ID=3 NAME=MODE_10W_2CORE
NVPM VERB: POWER_MODEL: ID=4 NAME=MODE_10W_4CORE
NVPM VERB: POWER_MODEL: ID=5 NAME=MODE_10W_DESKTOP
NVPM VERB: POWER_MODEL: ID=6 NAME=MODE_20W_2CORE
NVPM VERB: POWER_MODEL: ID=7 NAME=MODE_20W_4CORE
NVPM VERB: POWER_MODEL: ID=8 NAME=MODE_20W_6CORE
```

Activate max speed fan 
```
> sudo vim /etc/nvfancontrol.conf

# --- Modify ----
	FAN_PROFILE quiet {
...
	}
	FAN_PROFILE full { # Add this
		#TEMP 	HYST	PWM	RPM
		0	8 	255 	6000
		108	0 	255 	6000
	}
	FAN_PROFILE cool {
...
	FAN_DEFAULT_CONTROL open_loop
	FAN_DEFAULT_PROFILE full # activate to full
	FAN_DEFAULT_GOVERNOR pid

# ---- Saved file ----
> sudo systemctl stop nvfancontrol
> sudo rm /var/lib/nvfancontrol/status
> sudo systemctl start nvfancontrol
```

**Install TRTPose Model Api**
Install `torch2trt` for conversion form PyTorch model to TensorRT
```
> pip3 install tqdm cython pycocotools matplotlib # required packages
> git clone https://github.com/NVIDIA-AI-IOT/torch2trt
> cd torch2trt
> python3 setup.py install
```

Install TRTPose
```
> cd ../
> git clone https://github.com/NVIDIA-AI-IOT/trt_pose
> cd trt_pose/
> python3 setup.py install
```

```python
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
WIDTH = 256
HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

# ------- Results --------
- Converting ....
- Success
```

Clone Repo
```bash
> git clone https://github.com/tonible14012002/fall_detector
> cd fall_detector/
> python3 app.py --model trtpose --cam './sample-video'
# ---- Logs ----
Connected to server https://*************/
```

### Contribution
...