#!/usr/bin/env python3
"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from argparse import ArgumentParser, SUPPRESS
import json
import os
import sys

import cv2
import numpy as np

from modules.input_reader import InputReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
from modules import InferenceManager
from utils import PerfMonitor
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))
import monitors


class HumanPoseEstimation3d(InferenceManager):
    def __init__(self, args):
        self.perf_monitor = PerfMonitor(sinks=[self.displayCallback])
        InferenceManager.__init__(self, args.model, args.device, 0, 0, sinks=[self.perf_monitor])
        self.canvas_3d = np.zeros((200, 300, 3), dtype=np.uint8)
        #self.canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.plotter = Plotter3d(self.canvas_3d.shape[:2])
       
        file_path = args.extrinsics_path
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), 'data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        self.R = np.array(extrinsics['R'], dtype=np.float32)
        self.t = np.array(extrinsics['t'], dtype=np.float32)

        self.frame_provider = InputReader(args.input)
        self.base_height = args.height_size
        self.fx = args.fx
        self.stride = 8

        self.presenter = monitors.Presenter(args.utilization_monitors, 0)
        self.running = False

    def run(self):
        self.start()
        self.perf_monitor.start()
        self.running = True
        for frame in self.frame_provider:
            current_time = cv2.getTickCount()
            input_scale = self.base_height / frame.shape[0]
            scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
            if self.fx < 0:  # Focal length is unknown
                self.fx = np.float32(0.8 * frame.shape[1])

            scaled_img = scaled_img[0:scaled_img.shape[0] - (scaled_img.shape[0] % self.stride),
                      0:scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]
            input = dict(frames=[scaled_img],  original_frame=frame)
            self.put(input)
            if self.running == False:
                break
                
    def post_process(self, outputs, frame, index, input):
        result = (outputs['features'][0], outputs['heatmaps'][0], outputs['pafs'][0])
        frame = input['original_frame']
        input_scale = self.base_height / frame.shape[0]
        poses_3d, poses_2d = parse_poses(result, input_scale, self.stride, self.fx, self.frame_provider.is_video)
        
        self.presenter.drawGraphs(frame)
        draw_poses(frame, poses_2d)
        return input

    def displayCallback(self, frame, output):
        print(self.get_performances())
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        frame_s = cv2.resize(frame, (600, 400))
        cv2.imshow('image', frame_s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False

    def rotate_poses(self, poses_3d, R, t):
        R_inv = np.linalg.inv(R)
        for pose_id in range(poses_3d.shape[0]):
            pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
            pose_3d[0:3] = np.dot(R_inv, pose_3d[0:3] - t)
            poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

        return poses_3d


if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.',
                            add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                      help='Required. Path to an .xml file with a trained model.',
                      type=str, required=True)
    args.add_argument('-i', '--input',
                      help='Required. Path to input image, images, video file or camera id.',
                      nargs='+', default='')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU).',
                      type=str, default='CPU')
    args.add_argument('--height_size', help='Optional. Network input layer height size.', type=int, default=256)
    args.add_argument('--extrinsics_path',
                      help='Optional. Path to file with camera extrinsics.',
                      type=str, default=None)
    args.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args.add_argument('--no_show', help='Optional. Do not display output.', action='store_true')
    args.add_argument("-u", "--utilization_monitors", default='', type=str,
                      help="Optional. List of monitors to show initially.")
    args.add_argument('--output_dir', help='Optional. Path to directory to save results.', default='')
    args = parser.parse_args()

    if args.input == '':
        raise ValueError('Please, provide input data.')
    human_pose_estimation_3d = HumanPoseEstimation3d(args)
    e_start = cv2.getTickCount()
    human_pose_estimation_3d.run()
    t_elapsed = (cv2.getTickCount() - e_start) / cv2.getTickFrequency()
    print('time elapsed : {}'.format(t_elapsed))
   
   
   
