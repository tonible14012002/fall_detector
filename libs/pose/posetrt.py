import json
import trt_pose.coco
import trt_pose.models
import torch

# import torch2trt
import time
import cv2
import PIL.ImageDraw, PIL.ImageFont, PIL.Image  # noqa

import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects
import os.path
from libs.fall_detector.pose_predictor import (
    BasePosePredictor,
    BasedPoseEstimator,
)
from libs.fall_detector.preprocessor import BasePreprocessor

TRTModule = None
torch2trt = None

DETECTION_SIZE = (256, 256)


class TrtPreprocessor(BasePreprocessor):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    def __init__(self) -> None:
        super().__init__()
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        self.device = torch.device("cuda")

    ## Need resizer
    def preprocess(self, image):
        device = torch.device(self.device)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]


class TrtPosePredictor(BasePosePredictor):
    model = None
    torch_device = None
    parse_objects = None
    draw_objects = None

    def setup(self):
        with open("./human_pose.json", "r") as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts = len(human_pose["keypoints"])
        num_links = len(human_pose["skeleton"])

        MODEL_WEIGHTS = "./densenet121_baseline_att_256x256_B_epoch_160.pth"
        OPTIMIZED_MODEL = (
            "./densenet121_baseline_att_256x256_B_epoch_160_trt.pth"
        )

        model = (
            trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links)
            .cuda()
            .eval()
        )
        WIDTH, HEIGHT = DETECTION_SIZE

        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        if os.path.exists(OPTIMIZED_MODEL) is False:
            print(
                "-- Converting TensorRT models. This may takes several minutes..."
            )
            model.load_state_dict(torch.load(MODEL_WEIGHTS))
            model_trt = torch2trt.torch2trt(
                model, [data], fp16_mode=True, max_workspace_size=1 << 25
            )
            torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

        t0 = time.time()
        torch.cuda.current_stream().synchronize()
        for i in range(50):
            y = model_trt(data)  # noqa
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        print(50.0 / (t1 - t0))

        self.model = model_trt
        self.torch_device = torch.device("cuda")
        self.parse_objects = ParseObjects(topology)

    def set_device(self, device="cpu"):
        raise Exception(
            "set device not allowed in trt pose, available for cuda only"
        )

    def get_keypoint(humans, hnum, peaks):
        # check invalid human index
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
                # print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
            else:
                peak = (j, None, None)
                kpoint.append(peak)
                # print('index:%d : None %d'%(j, k) )
        return kpoint

    def predict(self, image):
        start = time.time()
        cmap, paf = self.model(image)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        end = time.time()
        counts, objects, peaks = self.parse_objects(
            cmap, paf
        )  # , cmap_threshold=0.15, link_threshold=0.15)
        poses = []
        for i in range(counts[0]):
            _kpoint = self.get_keypoint(objects, i, peaks)  # noqa
            poses.append(_kpoint)

        netfps = 1 / (end - start)
        print("===== NET FPS :%f =====" % (netfps))
        return _kpoint

    def postprocess(self, keypoints):

        return self.PoseResults(poses=[], bboxes_xyxy=[], scores=[])


class TrtPoseEstimator(BasedPoseEstimator):
    preprocessor = None
    predictor = None
    config = None

    @classmethod
    def new(cls, preprocessor, predictor, config):
        n = cls()
        n.config = config
        n.preprocessor = preprocessor
        n.predictor = predictor

    def set_predictor_device(self, device):
        raise Exception(
            "set device not allowed in trt pose, available for cuda with tensorRT only"
        )
