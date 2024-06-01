from pynput import keyboard

cur = 0


def on_press(key):
    global cur
    try:
        if key.char == "b":
            print("FALL DETECTED")
        if key.char == "a":
            print(LOGS[cur])
            cur += 1
    except AttributeError:
        pass


LOGS = [
    """2024-05-26 13:39:54.826352: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.""",
    """2024-05-26 13:39:55.484920: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:39:55.485924: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:39:55.490511: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:39:55.491281: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:39:55.491892: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:40:28.976311: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:40:28.977367: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:40:28.977551: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1721] Could not identify NUMA node of platform GPU id 0, defaulting to 0.
Your kernel may not have been built with NUMA support.
2024-05-26 13:40:28.978044: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] could not open file to read NUMA node: /sys/bus/pci/devices/0000:00:00.0/numa_node
Your kernel may have been built without NUMA support.
2024-05-26 13:40:28.978324: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1637] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2162 MB memory:  -> device: 0, name: Xavier, pci bus id: 0000:00:00.0, compute capability: 7.2""",
    # 3s
    """Connected to server https://signaling-server-pfm2.onrender.com/""",
    """ICE connection state is checking""",
    """Connection state is connecting""",
    """Connection state is connected""",
    # Until finish
    """Connection state is closed""",
]
# Setting up the listener

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
