import asyncio
import socketio
import entities as app_entities
import logging
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
    RTCConfiguration,
)
import os
import signal
from video_track import VideoTransformTrack
from aiortc.mediastreams import MediaStreamTrack
from utils.cam import CamLoader_Q
from utils.resize_preproc import detection_preproc


class Streamer:
    entities: app_entities.Entities = None
    sio = socketio.AsyncClient()
    pcs = set()

    id: str = None
    username: str = None

    media_track: MediaStreamTrack = None

    def set_stream_id(self, id: str):
        self.id = id

    def set_username(self, username: str):
        self.username = username

    @classmethod
    def new(
        cls,
        entities: app_entities.Entities,
        media_track: MediaStreamTrack = None,
        id: str = None,
        username: str = None,
    ):
        s = cls()
        s.entities = entities
        s.id = id
        s.username = username
        s.media_track = media_track
        return s

    def set_stream_track(self, stream_track: MediaStreamTrack):
        """
        copy of this stream_track
        """
        self.media_track = stream_track

    async def on_shutdown(self):
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    async def join(self, id, username):
        await self.sio.emit("join", {"username": username, "room": id})

    async def stream(self):
        self.register_connect()
        self.register_offer()
        try:
            await self.sio.connect(self.entities.config.signaling_server)
            await self.sio.wait()
        except Exception as e:
            print("Error in start method: ", e)
            os.kill(os.getpid(), signal.SIGINT)

    def register_connect(self):
        @self.sio.event
        async def connect() -> None:
            try:
                print(
                    "Connected to server %s"
                    % self.entities.config.signaling_server
                )
                # await send_ack()
                await self.join(self.id, self.username)
            except Exception as e:
                print("Error in connect event handler: ", e)

    def register_offer(self):
        @self.sio.event
        async def offer(data) -> None:
            params = data
            offer = RTCSessionDescription(
                sdp=params["sdp"], type=params["type"]
            )
            ice_servers = [
                RTCIceServer(
                    urls=[
                        "stun:stun1.l.google.com:19302",
                        "stun:stun2.l.google.com:19302",
                    ]
                ),
            ]
            config = RTCConfiguration(iceServers=ice_servers)
            pc = RTCPeerConnection(config)
            self.pcs.add(pc)

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                print("ICE connection state is %s" % pc.iceConnectionState)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print("Connection state is %s" % pc.connectionState)
                if pc.connectionState == "failed":
                    await pc.close()
                    self.pcs.discard(pc)

            pc.addTrack(self.media_track)

            # handle offer
            await pc.setRemoteDescription(offer)

            # send answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            await self.sio.emit(
                "answer",
                {
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                    "username": self.username,
                    "room": self.id,
                },
            )
            pass

    def setup_logging(self):
        if self.entities.config.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def start(self):
        asyncio.run(self.stream())


streamer = None


def init_streamer(
    entities: app_entities.Entities, username: str = None, id: str = None
):
    global streamer
    if streamer is not None and isinstance(streamer, Streamer):
        return streamer

    streamer = Streamer.new(
        entities,
        id=id,
        username=username,
    )
    return streamer


__all__ = [
    "init_streamer",
]

# TESTING
if __name__ == "__main__":

    class EmptyFallDetection:
        pass

    entities = app_entities.Entities.new(
        fall_detector=EmptyFallDetection(),
        config=app_entities.app_config.AppConfig.new(
            signaling_server_url="https://signaling-server-pfm2.onrender.com/",
            ice_server_urls=[
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302",
            ],
            video_src="./scripts/samples/fall-vid.mp4",
        ),
    )
    cam = CamLoader_Q(
        entities.config.video_src,
        queue_size=10000,
        preprocess=detection_preproc,
    ).start()
    streamer = Streamer.new(
        entities,
        media_track=VideoTransformTrack(cam=cam),
        id="1",
        username="1",
    )
    streamer.set_stream_id("1")
    streamer.set_username("1")
    streamer.start()
