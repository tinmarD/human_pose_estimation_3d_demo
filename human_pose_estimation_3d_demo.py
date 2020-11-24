#!/usr/bin/env python
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

import cv2
import numpy as np
import h5py

from modules.inference_engine import InferenceEngine
from modules.input_reader import InputReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(poses_3d.shape[0]):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3] = np.dot(R_inv, pose_3d[0:3] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


def create_output_file(output_dir_path):
    if not os.path.exists(output_dir_path):
        try:
            os.mkdir(output_dir_path)
        except:
            print('Could not create directory {}'.format(output_dir_path))
            return '', [], []
    in_filepath = args.input[0]
    in_filename = in_filepath[in_filepath.rfind(os.path.sep) + 1:]
    if in_filename.rfind('.') is not -1:
        in_filename = in_filename[:in_filename.rfind('.')]
    # Create unique output filepath
    temp_path = os.path.join(output_dir_path, 'human_pose_estimation_3d_{}'.format(in_filename))
    i = 0
    while os.path.exists('{}_{}.hdf5'.format(temp_path, i)):
        i += 1
    out_res_filepath = '{}_{}.hdf5'.format(temp_path, i)
    out_res_filepath_light = '{}_{}_light.hdf5'.format(temp_path, i)
    f_res = h5py.File(out_res_filepath, 'w')
    f_res_light = h5py.File(out_res_filepath_light, 'w')
    res_3d_coords = f_res.create_group("3d_coords")
    res_3d_coords.create_group("x")
    res_3d_coords.create_group("y")
    res_3d_coords.create_group("z")
    f_res.create_group("kpt_heatmaps")
    f_res.create_group("kpt_pairwise_rel")
    f_res.create_group("time")
    res_3d_coords_l = f_res_light.create_group("3d_coords")
    res_3d_coords_l.create_group("x")
    res_3d_coords_l.create_group("y")
    res_3d_coords_l.create_group("z")
    f_res_light.create_group("time")
    print("Saving results in {}".format(out_res_filepath))
    return out_res_filepath, f_res, f_res_light


def create_videos_file(save_results, out_hdf5_filepath, output_video_size):
    if save_results:
        out_vid_filepath = out_hdf5_filepath[:-5]+'.avi'
    else:
        os.mkdir('out_videos')
        out_vid_filepath = os.path.join('out_videos', 'out_video.avi')
    vid_writer = cv2.VideoWriter(out_vid_filepath, cv2.VideoWriter_fourcc('M','J','P','G'), 9.0, output_video_size)
    out_vid_dirpath, out_vid_filename = os.path.dirname(out_vid_filepath), os.path.basename(out_vid_filepath)[:-4]
    vid_frames_dir = os.path.join(out_vid_dirpath, out_vid_filename)
    try:
        os.mkdir(vid_frames_dir)
        save_video = True
    except:
        save_video = False
    return save_video, out_vid_filepath, vid_frames_dir, vid_writer


def save_hdf5_results(f_res, f_res_light, i_frame, t_frame, poses_3d, heatmaps, pafs):
    i_frame_str = str(i_frame)
    f_res.require_group("3d_coords/x").create_dataset(i_frame_str, data=poses_3d[:, 0::4])
    f_res.require_group("3d_coords/y").create_dataset(i_frame_str, data=poses_3d[:, 1::4])
    f_res.require_group("3d_coords/z").create_dataset(i_frame_str, data=poses_3d[:, 2::4])
    f_res.require_group("kpt_heatmaps").create_dataset(i_frame_str, data=heatmaps)
    f_res.require_group("kpt_pairwise_rel").create_dataset(i_frame_str, data=pafs)
    f_res.require_group("time").create_dataset(i_frame_str, data=t_frame)
    f_res_light.require_group("3d_coords/x").create_dataset(i_frame_str, data=poses_3d[:, 0::4])
    f_res_light.require_group("3d_coords/y").create_dataset(i_frame_str, data=poses_3d[:, 1::4])
    f_res_light.require_group("3d_coords/z").create_dataset(i_frame_str, data=poses_3d[:, 2::4])
    f_res_light.require_group("time").create_dataset(i_frame_str, data=t_frame)



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
    args.add_argument('--output_dir', help='Optional. Path to directory to save results.', type=str, default='')
    #  args.add_argument('--save_heatmaps', help='Optional. If True and output_dir is set, will save heatmaps in addition'
    #                                            ' to the 3D coordinates', type=bool, default=False)
    args.add_argument('--save_video', help='Optional. If save the output video', default='')
    args = parser.parse_args()

    if args.input == '':
        raise ValueError('Please, provide input data.')

    stride = 8
    inference_engine = InferenceEngine(args.model, args.device, stride)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    output_video_size = (640, 360)
    # If output_dir is given, create output file to save results
    save_results, save_video = False, False
    if args.output_dir:
        out_res_filepath, f_res, f_res_light = create_output_file(args.output_dir)
        if out_res_filepath:
            save_results = True
    # If save_video is set, create output directory for saving video and images
    if args.save_video:
        save_video, out_vid_filepath, vid_frames_dir, vid_writer = create_videos_file(save_results, out_res_filepath, output_video_size)

    frame_provider = InputReader(args.input)
    is_video = frame_provider.is_video
    base_height = args.height_size
    fx = args.fx
    i_frame = 0
    t_start = cv2.getTickCount() / cv2.getTickFrequency()

    mean_time = 0
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    for frame in frame_provider:
        current_time = cv2.getTickCount()
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        print('frame shape : {}'.format(frame.shape))
        print(scaled_img.shape)
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        # inference_results[0] -> 3D coords
        # inference_results[1] -> Keypoint heatmaps
        # inference_results[2] -> Pairwise Relations
        inference_result = inference_engine.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        edges = []
        if len(poses_3d) > 0:
            x, y, z = poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4]
            if save_results:
                t_frame = cv2.getTickCount() / cv2.getTickFrequency() - t_start
                save_hdf5_results(f_res, f_res_light, i_frame, t_frame, poses_3d, inference_result[1], inference_result[2])
            i_frame += 1

        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        fps = int(1 / mean_time * 10) / 10
        cv2.putText(frame, 'FPS: {}'.format(fps), (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        if save_video:
            frame_s = cv2.resize(frame, output_video_size)
            vid_writer.write(frame_s)
            cv2.imwrite(os.path.join(vid_frames_dir, '{}.png'.format(str(i_frame))), frame)
        # Print mean fps on console
        print('Frame {} - FPS : {}'.format(i_frame, fps))

        if args.no_show:
            continue

        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    f_res.close()
    f_res_light.close()
    vid_writer.release()


