run-demo-cpu:
	@python main.py -C './scripts/samples/fall-vid.mp4' --device cpu

run-camera-cpu:
	@python main.py -C 0 --device cpu

run-demo-cuda:
	@python main.py -C './scripts/samples/fall-vid.mp4' --device cuda
	
run-camera-cuda:
	@python main.py -C 0 --device cuda

test_streawm:
	@python stream.py

test-streamer:
	@python streamer.py

test-cv-cam:
	@python test_cam.py -C 0

test-cv-video:
	@python test_cam.py -C './scripts/samples/fall-vid.mp4'

start:
	@python app.py

test-falldetection:
	@python fall_detection.py
