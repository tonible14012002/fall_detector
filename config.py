SIGNALING_SERVER = "https://signaling-server-pfm2.onrender.com/"
ICE_SERVERS = [
    "stun:stun1.l.google.com:19302",
    "stun:stun2.l.google.com:19302",
]
DEFAULT_DETECTION_SIZE = (640, 640)
VIDEO_SRC = "./scripts/samples/fall-vid.mp4"
DEVICE_ID = "1"


class AppConfig:
    signaling_server_url = SIGNALING_SERVER
    ice_server_urls = ICE_SERVERS
    video_src = VIDEO_SRC
    detection_size = (640, 640)
    device = "cpu"
    device_id = "1"

    @classmethod
    def new(
        cls,
        signaling_server_url: str = None,
        ice_server_urls: list = None,
        video_src: str = None,
        detection_size: "tuple[int, int]" = None,
        device="cpu",
        device_id="1",
    ):
        if signaling_server_url is None:
            raise Exception("signaling_server_url is required")
        if ice_server_urls is None:
            raise Exception("ice_server_urls is required")
        if video_src is None:
            raise Exception("video_src is required")

        config = cls()
        config.signaling_server = signaling_server_url
        config.ice_servers = ice_server_urls
        config.video_src = video_src
        config.detection_size = detection_size
        config.device = device
        config.device_id = device_id
        return config


def init_default_config(device="cpu"):
    return AppConfig.new(
        signaling_server_url=SIGNALING_SERVER,
        ice_server_urls=ICE_SERVERS,
        video_src=VIDEO_SRC,
        detection_size=DEFAULT_DETECTION_SIZE,
        device=device,
        device_id=DEVICE_ID,
    )
