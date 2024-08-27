import os
from datetime import datetime

def create_output_dir(base_path):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(base_path, current_time)
    os.makedirs(path, exist_ok=True)
    return path