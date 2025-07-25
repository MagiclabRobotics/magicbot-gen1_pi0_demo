from openpi_client import websocket_client_policy
from openpi.policies import droid_policy
from openpi.policies import aloha_policy
import time
import numpy as np

# create the input
example = droid_policy.make_droid_example()
# example = aloha_policy.make_aloha_example()
print(example)

# ===========================================================================================
# example = {
#         "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/joint_position": np.random.rand(7),
#         "observation/gripper_position": np.random.rand(1),
#         "prompt": "help me do something",
#     }
# ===========================================================================================
start_time = time.time()

policy_client = websocket_client_policy.WebsocketClientPolicy(host="10.10.37.63", port=8001)
action_chunk = policy_client.infer(example)["actions"]

end_time = time.time()
print('spend time:', end_time - start_time)
print('action_chunk=', action_chunk.shape)
print('action_chunk=', action_chunk)
# ===========================================================================================