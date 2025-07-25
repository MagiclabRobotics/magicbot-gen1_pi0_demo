# MODIFIED BY MagicLab, 2025
"""
Script to convert Magiclab humanoid hdf5 data to the LeRobot dataset v2.0 format.
"""
import os
import numpy as np
import shutil
import sys
import tyro
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import h5py

INPUT_PATH = '/data3/Pi0_humaniod_data/hdf5_data/pt_test/'
OUTPUT_PATH = '/data3/Pi0_humaniod_data/lerobot_data/pt_test/'
PROMPT = "grasp the pitaya, and place it in basket"  

CAM_LIST = ["head", "right_hand"]

def main(data_dir, output_dir, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    print(output_dir)
    output_path = Path(output_dir)


    if output_path.exists():
        print("file exist ！")

        sys.exit()

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=output_dir,
        robot_type="humanrobot",
        fps=30,
        features={
            "head_image": {
                "dtype": "image",
                "shape": (480, 640, 3), #image.size[0], image.size(1)
                "names": ["height", "width", "channel"],
            },
            "right_hand_image": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (30,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (30,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    data_dir = Path(data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))

    def load_images(ep: h5py.File) -> dict[str, np.ndarray]:
        imgs_per_cam = {}
        for camera in CAM_LIST:  
            imgs_array = ep[f"/observations/images/{camera}"][:]
            if imgs_array.shape[-1] == 3:
                imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))  # CHW
            imgs_per_cam[camera] = imgs_array
        return imgs_per_cam

    for ep_path in tqdm(hdf5_files, desc="Converting HDF5 files to parquet", unit="file"):
        with h5py.File(ep_path, "r") as ep:
            task_description = ep["/task"][()].decode('utf-8') if "/task" in ep else "unknown_task"
            state = ep["/action"][:]  
            action = ep["/action"][:] 
            imgs_per_cam = load_images(ep)

        num_steps = state.shape[0]
        for step_idx in range(2, num_steps):
            current_state = state[step_idx]
            prev_state = state[step_idx - 2]
            dataset.add_frame({
                "head_image": imgs_per_cam["head"][step_idx].transpose(1, 2, 0),  # HWC
                "right_hand_image": imgs_per_cam["right_hand"][step_idx].transpose(1, 2, 0),  
                "state": prev_state[0:30],
                "actions": current_state[0:30],
            })

        task_description = PROMPT

        dataset.save_episode(task=task_description)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)
    # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
        # )
    print(f"✅ Dataset saved to: {output_path.resolve()}")


if __name__ == '__main__':
    # tyro.cli(main)
    main(data_dir=INPUT_PATH, output_dir =OUTPUT_PATH)