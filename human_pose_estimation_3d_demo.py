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
import h5py
import re

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
        InferenceManager.__init__(self, args.model, args.device, 6, 0, sinks=[self.perf_monitor])
        #self.canvas_3d = np.zeros((200, 300, 3), dtype=np.uint8)
        self.canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.plotter = Plotter3d(self.canvas_3d.shape[:2])
       
        file_path = args.extrinsics_path
        if file_path is None:
            file_path = os.path.join(os.path.dirname(__file__), 'data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        self.R = np.array(extrinsics['R'], dtype=np.float32)
        self.t = np.array(extrinsics['t'], dtype=np.float32)

        self.output_dir_path = args.output_dir
        self.save_results, out_hdf5_filepath = self.create_output_file()
        self.create_videos_file(args.save_video, out_hdf5_filepath)
        self.i_frame = 0
        self.frame_provider = InputReader(args.input)
        self.base_height = args.height_size
        self.fx = args.fx
        self.stride = 8
        self.t_start = cv2.getTickCount() / cv2.getTickFrequency()

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
                if self.save_results:
                    self.f_res.close()
                break

    def create_videos_file(self, save_video, out_hdf5_filepath):
        if save_video:
            if self.save_results:q
                out_vid_filepath = out_hdf5_filepath[:-5]+'.avi'
            else:
                os.mkdir('out_videos')
                out_vid_filepath = os.path.join('out_videos', 'out_video.avi')
            self.save_video = True
            self.vid_writer = cv2.VideoWriter(out_vid_filepath, cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (600, 400))
            out_vid_dirpath, out_vid_filename = os.path.dirname(out_vid_filepath), os.path.basename(out_vid_filepath)[:-4]
            self.vid_frames_dir = os.path.join(out_vid_dirpath, out_vid_filename)
            os.mkdir(self.vid_frames_dir)
        else:
            self.save_video = False
            self.vid_writer = []

    def create_output_file(self):
        if not self.output_dir_path:
            return False, ''
        else:
            if not os.path.exists(self.output_dir_path):
                try:
                    os.mkdir(self.output_dir_path)
                except:
                    print('Could not create directory {}'.format(self.output_dir_path))
                    return False, ''
            in_filepath = args.input[0]
            in_filename = in_filepath[in_filepath.rfind(os.path.sep) + 1:]
            if in_filename.rfind('.') is not -1:
                in_filename = in_filename[:in_filename.rfind('.')]
            # Create unique output filepath
            temp_path = os.path.join(self.output_dir_path, 'human_pose_estimation_3d_{}'.format(in_filename))
            i = 0
            while os.path.exists('{}_{}.hdf5'.format(temp_path, i)):
                i += 1
            out_res_filepath = '{}_{}.hdf5'.format(temp_path, i)
            out_res_filepath_light = '{}_{}_light.hdf5'.format(temp_path, i)
            self.f_res = h5py.File(out_res_filepath, 'w')
            self.f_res_light = h5py.File(out_res_filepath_light, 'w')
            res_3d_coords = self.f_res.create_group("3d_coords")
            res_3d_coords.create_group("x")
            res_3d_coords.create_group("y")
            res_3d_coords.create_group("z")
            self.f_res.create_group("kpt_heatmaps")
            self.f_res.create_group("kpt_pairwise_rel")
            self.f_res.create_group("time")
            res_3d_coords_l = self.f_res_light.create_group("3d_coords")
            res_3d_coords_l.create_group("x")
            res_3d_coords_l.create_group("y")
            res_3d_coords_l.create_group("z")
            self.f_res_light.create_group("time")
            print("Saving results in {}".format(out_res_filepath))
            return True, out_res_filepath

    def post_process(self, outputs, frame, index, input):
        result = (outputs['features'][0], outputs['heatmaps'][0], outputs['pafs'][0])
        frame = input['original_frame']
        input_scale = self.base_height / frame.shape[0]
        poses_3d, poses_2d = parse_poses(result, input_scale, self.stride, self.fx, self.frame_provider.is_video)
        print('IN POST PROCESS : len_poses_3d : {}'.format(len(poses_3d)))
        if self.save_results and len(poses_3d) > 0:
            time_frame = cv2.getTickCount() / cv2.getTickFrequency() - self.t_start
            self.f_res.require_group("3d_coords/x").create_dataset(str(self.i_frame), data=poses_3d[:, 0::4])
            self.f_res.require_group("3d_coords/y").create_dataset(str(self.i_frame), data=poses_3d[:, 1::4])
            self.f_res.require_group("3d_coords/z").create_dataset(str(self.i_frame), data=poses_3d[:, 2::4])
            self.f_res.require_group("kpt_heatmaps").create_dataset(str(self.i_frame), data=outputs['heatmaps'][0])
            self.f_res.require_group("kpt_pairwise_rel").create_dataset(str(self.i_frame), data=outputs['pafs'][0])
            self.f_res.require_group("time").create_dataset(str(self.i_frame), data=time_frame)
            self.f_res_light.require_group("3d_coords/x").create_dataset(str(self.i_frame), data=poses_3d[:, 0::4])
            self.f_res_light.require_group("3d_coords/y").create_dataset(str(self.i_frame), data=poses_3d[:, 1::4])
            self.f_res_light.require_group("3d_coords/z").create_dataset(str(self.i_frame), data=poses_3d[:, 2::4])
            self.i_frame += 1
        self.presenter.drawGraphs(frame)
        draw_poses(frame, poses_2d)
        return input

    def displayCallback(self, frame, output):
        print('Frame {} : {}'.format(self.i_frame, self.get_performances()))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        frame_s = cv2.resize(frame, (600, 400))
        cv2.imshow('image', frame_s)
        if self.save_video:
            self.vid_writer.write(frame_s)
            cv2.imwrite(os.path.join(self.vid_frames_dir, '{}.png'.format(str(self.i_frame))), frame_s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if self.save_results:
                self.f_res.close()
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
    args.add_argument('--output_dir', help='Optional. Path to directory to save results.', type=str, default='')
  #  args.add_argument('--save_heatmaps', help='Optional. If True and output_dir is set, will save heatmaps in addition'
  #                                            ' to the 3D coordinates', type=bool, default=False)
    args.add_argument('--save_video', help='Optional. If save the output video', default='')
    args = parser.parse_args()

    if args.input == '':
        raise ValueError('Please, provide input data.')
    human_pose_estimation_3d = HumanPoseEstimation3d(args)
    e_start = cv2.getTickCount()
    human_pose_estimation_3d.run()
    t_elapsed = (cv2.getTickCount() - e_start) / cv2.getTickFrequency()
    print('time elapsed : {}'.format(t_elapsed))
