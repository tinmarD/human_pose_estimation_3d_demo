#!/usr/bin/env python

import os
import sys
import math 
import multiprocessing
import threading
from threading import Thread, Condition
import queue
from queue import Queue
import cv2
from statistics import median

from openvino.inference_engine import IECore, get_version

from utils import logger, Sink

from .device_manager import Device
from .network import Network
from .requests_wrapper import InferRequest, InferRequestsQueue

#----------------------------------------------------------------------------------------------------------------------
# Inference Manager 
#----------------------------------------------------------------------------------------------------------------------
class InferenceManager(Sink):
	def __init__(self, model, device_name="CPU", num_reqs=0, threshold=0.5, sinks=[]):
		Sink.__init__(self, name=self.__class__.__name__, order=True)
		self.ie = IECore()
		self.threshold = threshold
		self.sinks = sinks	
		self.enable(False)
		self.flush()
		logger.info('--------------------------------------------------------------')	 
		logger.info("InferenceEngine:\n{: <9}{}".format("",get_version()))
		version_string = "Device is {}".format(device_name)
		logger.info(version_string)
		logger.info("Threshold {}".format(threshold))

		self.device = Device.create(device_name, self.ie, num_reqs);
		self.device.config()

		self.net = Network(ie=self.ie, model=model, device=self.device, batch_size=1)		
		self.num_reqs = len(self.net.infer_requests)

		logger.info("Requests number: {}".format(self.num_reqs));
		self.queue = [None]*self.num_reqs
		self.requests_queue = InferRequestsQueue(self.net.infer_requests, self.net, self.process_output)

		self.request_idx = 0
		self.queue = [None]*self.num_reqs
		self.enable(True)

	#--------------------------------------------------------------------------------
	def process_output(self, inferRequest):

		self.lock()

		request_index, batch_index, user_data = inferRequest.user_data		
		output = self.post_process(inferRequest.request.outputs, inferRequest.frame, batch_index, user_data)
		if output is not None:
			self.queue[request_index % self.num_reqs] = output
			self.push()
		
		self.unlock()
	
	def handler(self):
		self.lock()
		self.request_idx = 0;
		request_index = 0;
		while self.isRunning():
			self.unlock()
			input = self.get(0.1)
			self.lock()
			if input is not None:
				batch_index = 0;
				frames, user_data, input = self.pre_process(input)
				
				if len(frames) > 0:
					self.batch_size = len(frames)
					for frame in frames:
						self.unlock()
						infer_request = self.requests_queue.getIdleRequest()
						if not infer_request:
							raise Exception("No idle Infer Requests!")
						self.lock()

						while request_index >= (self.request_idx + self.num_reqs):
							self.wait(0.1)

						infer_request.startAsync(frame, (request_index,batch_index,user_data))

						batch_index += 1
					request_index += 1
				else:
					self.queue[request_index % self.num_reqs] = input
					self.push()
					request_index += 1	
		
		if self.requests_queue:
			self.requests_queue.waitAll()
			self.requests_queue = None

		self.unlock()

	def push(self):
		while self.queue[self.request_idx % self.num_reqs] is not None:
			output = self.queue[self.request_idx % self.num_reqs]
			for sink in self.sinks:
				if callable(sink):
					sink(output['original_frame'], output)
				else:
					sink.put(output)
			self.queue[self.request_idx % self.num_reqs] = None
			self.request_idx += 1
			self.notify()

	def pre_process(self, input):
		return (input['frames'], input, input);

	def post_process(self, outputs):
		return None

	def get_performances(self):
		times = self.requests_queue.times
		latency_ms = None
		if len(times) > 0:
			times.sort()
			latency_ms = int(median(times))
		return (self.requests_queue.getFPS(),latency_ms)

	def draw(self, frame, fps=None, latency=None, message=None, margin = 10):
		fontFace =  cv2.FONT_HERSHEY_SIMPLEX
		frame_size = frame.shape[:-1]
		fontScale = 0.4
		thickness = 1
		bcolor = (0,255,255)
		fcolor = (0,0,255)

		if fps is not None:
			fps = "{:.1f}".format(fps)
			textsize = cv2.getTextSize(fps, fontFace, fontScale, thickness)[0]
			center = (int(margin + textsize[0] / 2), int(margin + textsize[0] / 2))
			radius = int(textsize[0]/2 + 6)
			cv2.circle(frame, center, radius, bcolor, 1, cv2.LINE_AA)
			textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
			cv2.putText(frame, fps, textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)
			textsize = cv2.getTextSize("FPS", fontFace, fontScale, thickness)[0]
			textPos = (int(center[0]  + radius + 5), int(center[1] + textsize[1]/2))
			cv2.putText(frame, "FPS", textPos, fontFace, fontScale, bcolor, thickness, cv2.LINE_AA)

		if latency is not None:
			textsize = cv2.getTextSize(str(latency), fontFace, fontScale, thickness)[0]
			center = (int(frame_size[1] - margin - textsize[0] / 2), int(margin + textsize[0] / 2))
			radius = int(textsize[0]/2 + 6)
			cv2.circle(frame, center, radius, bcolor, 1, cv2.LINE_AA)
			textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
			cv2.putText(frame, str(latency), textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)
			textsize = cv2.getTextSize("Latency", fontFace, fontScale, thickness)[0]
			textPos = (int(center[0] - textsize[0] - radius - 5), int(center[1] + textsize[1]/2))
			cv2.putText(frame, "Latency", textPos, fontFace, fontScale, bcolor, thickness, cv2.LINE_AA)

		if message:
			fontScale = 1
			thickness = 2
			textsize = cv2.getTextSize(message, fontFace, fontScale, thickness)[0]
			textPos = (int((frame_size[1] - textsize[0])/2), int(frame_size[0] - margin))
			cv2.putText(frame, message, textPos, fontFace, fontScale, (0,0,255), thickness)


