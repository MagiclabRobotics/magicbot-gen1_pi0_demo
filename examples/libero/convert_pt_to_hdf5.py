import torch
import h5py
import numpy as np
from PIL import Image
import io
import os
import glob
from tqdm import tqdm

input_folder = '/data3/Pi0_humaniod_data/pt_data/pt_test'  
output_folder = '/data3/Pi0_humaniod_data/hdf5_data/pt_test'  

def convert_pt_to_hdf5(pt_file_path, hdf5_file_path):

    data = torch.load(pt_file_path, weights_only=False)

    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        for key, value in data.items():

            if 'images' in key and isinstance(value, list):
                images = []
                for idx, img_bytes in enumerate(value):
                    if isinstance(img_bytes, bytes):
                        try:

                            with Image.open(io.BytesIO(img_bytes)) as img:
                                img = img.convert('RGB')
   
                                image_data = np.array(img)
                                images.append(image_data)
                        except Exception as e:
                            print(f"Error processing image at index {idx} for key {key}: {e}")

                if images:
                    images_np = np.array(images)
                    hdf5_file.create_dataset(key, data=images_np)

            elif isinstance(value, torch.Tensor):
                hdf5_file.create_dataset(key, data=value.numpy())
            else:
                try:
                    hdf5_file.create_dataset(key, data=value)
                except TypeError as te:
                    print(f"Skipping unsupported type for {key}: {te}")


def convert_all_pt_in_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    pt_files = glob.glob(os.path.join(input_folder, "*.pt"))

    if not pt_files:
        print(f"No .pt files found in {input_folder}")
        return

    for pt_file in tqdm(pt_files, desc="Converting .pt files to HDF5", unit="file"):

        file_name = os.path.basename(pt_file).replace('.pt', '.hdf5')
        hdf5_file_path = os.path.join(output_folder, file_name)

        convert_pt_to_hdf5(pt_file, hdf5_file_path)

convert_all_pt_in_folder(input_folder, output_folder)