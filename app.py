import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import cv2
import numpy as np
import time
import open3d as o3d
#import multiprocessing as mp
import torch.multiprocessing as mp
from queue import Empty
from collections import deque
from utils.app import *


def run_estimator(model_path, device, camera_idx, image_size, window_scale, data_q, stop):
    depth_pipeline = get_depth_pipeline(model_path, device)

    window_size = int(image_size * window_scale)

    cv2.namedWindow('2D & Depth')
    vc = cv2.VideoCapture(camera_idx)

    frame_times = deque(maxlen=20)
    mean_fps = 0
    while True:
        start_time = time.time()

        read_ok, image_frame = vc.read()
        key = cv2.waitKey(1)
        if key == 27 or not read_ok or stop.is_set(): # exit on ESC
            stop.set()
            break

        image_frame = center_crop(image_frame)
        image_frame = downscale(image_frame, image_size)

        depth = depth_pipeline(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
        #depth = np.log(depth)
       
        depth_norm = normalize_depth(depth)
        depth_frame = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        data_q.put((image_frame, depth))

        cv2.putText(image_frame, f'{mean_fps:.0f}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 222, 0), 2, cv2.LINE_AA)
        cv2.imshow('2D & Depth', np.concatenate((
            upscale(image_frame, window_size), upscale(depth_frame, window_size)
        ), axis=1))

        end_time = time.time()
        frame_times.append(end_time - start_time)
        mean_fps = 1 / (sum(frame_times) / len(frame_times))
     
    cv2.destroyWindow('2D & Depth')
    vc.release()


def run_reconstructor(depth_scale, image_size, window_scale, data_q, stop):
    vis = o3d.visualization.Visualizer()
    window_size = int(image_size * window_scale * 2)
    vis.create_window('3D', width=window_size, height=window_size, left=window_size * 2)

    cloud = o3d.geometry.PointCloud()
    setup = True

    while vis.poll_events():
        if stop.is_set():
            break
        try:
            image, depth = data_q.get_nowait()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) / 255
            depth = depth - depth.min()
            points = depth2xyz(depth, depth_scale)

            cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3) * np.array([1, -1, 1]))
            cloud.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3))
            if setup:
                vis.add_geometry(cloud, True)
                setup = False
            vis.update_geometry(cloud)
            vis.update_renderer()
        except Empty:
            pass 

    vis.destroy_window()
    stop.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('2D to 3D demo')
    parser.add_argument('-m', '--model', dest='model_path', type=str, default=None)
    parser.add_argument('-c', '--camera', dest='camera_idx', type=int, default=0)
    parser.add_argument('-s', '--imgsize', dest='image_size', type=int, default=224)
    parser.add_argument('-w', '--wscale', dest='window_scale', type=float, default=2.0)
    parser.add_argument('-d', '--dscale', dest='depth_scale', type=float, default=10.0)
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
 
    ctx = mp.get_context('spawn')
    m = ctx.Manager()
    data_q = m.Queue()
    stop = m.Event()

    estimator_process = ctx.Process(target=run_estimator, args=(args.model_path, device, args.camera_idx, args.image_size, args.window_scale, data_q, stop))
    reconstructor_process = ctx.Process(target=run_reconstructor, args=(args.depth_scale, args.image_size, args.window_scale, data_q, stop))

    estimator_process.start()
    reconstructor_process.start()

    estimator_process.join()
    reconstructor_process.join()

