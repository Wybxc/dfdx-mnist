"""
Fetch mnist dataset
"""

import numpy as np
import os
import requests
from tqdm import tqdm
import gzip

# 定义下载MNIST数据集的函数
def download_mnist(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for file in files:
        file_url = f"{url}/{file}"
        file_path = os.path.join(save_path, file)

        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            response = requests.get(file_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='B', unit_scale=True)
            
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            
            t.close()
            if total_size != 0 and t.n != total_size:
                print("Download failed.")
            

# 下载和保存MNIST数据集
mnist_url = 'http://yann.lecun.com/exdb/mnist'
save_dir = 'mnist_data'
download_mnist(mnist_url, save_dir)

print("MNIST数据集已下载。")
