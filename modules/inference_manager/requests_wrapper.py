#!/usr/bin/env python

import os
import sys
import time
from datetime import datetime
import numpy as np

from threading import Condition

from openvino.inference_engine import IENetwork, IECore
import cv2

from utils import logger

class InferRequest:
	def __init__(self, request, id, net, callbackQueue):
		self.id = id
		self.request = request
		self.net = net
		self.request.set_completion_callback(self.callback, self.id)
		self.callbackQueue = callbackQueue
		self.frame = None
		self.updateFrames = True

	def callback(self, statusCode, userdata):
		if (userdata != self.id):
			logger.info("Request ID {} does not correspond to user data {}".format(self.id, userdata))
		elif statusCode != 0:
			logger.info("Request {} failed with status code {}".format(self.id, statusCode))
		self.callbackQueue(self.id, self.request.latency, self.user_data)

	def startAsync(self, frame, user_data=None):
		self.user_data = user_data
		self.frame = frame.copy()
		
		height, width = frame.shape[:-1]

		if frame.shape[:-1] != (self.net.height, self.net.width):
			frame = cv2.resize(frame, (self.net.width, self.net.height), interpolation = cv2.INTER_NEAREST)
			if frame.shape[2] == 4:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

		frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
		frame = frame.reshape(1, self.net.c, self.net.height, self.net.width)

		self.request.async_infer({self.net.input_blob:[frame]})

	def infer(self, input_data):
		self.request.infer(input_data)
		self.callbackQueue(self.id, self.request.latency);


class InferRequestsQueue:
	def __init__(self, requests, net, callback):
		self.idleIds = []
		self.requests = []
		self.times = []
		self.callback = callback
		for id in range(0, len(requests)):
				self.requests.append(InferRequest(requests[id], id, net, self.putIdleRequest))
				self.idleIds.append(id)
		self.startTime = datetime.max
		self.endTime = datetime.min
		self.framesNumber = 0
		self.cv = Condition()

	def resetTimes(self):
		self.startTime = datetime.max
		self.endTime = datetime.min
		self.framesNumber = 0
		self.times.clear()

	def getDurationInSeconds(self):
		return (self.endTime - self.startTime).total_seconds()

	def getFPS(self):
		fps = round(self.framesNumber / self.getDurationInSeconds(),1)
		return fps
		
	def putIdleRequest(self, id, latency, user_data):
		self.cv.acquire()
		self.framesNumber += 1 
		self.endTime = max(self.endTime, datetime.now())
		self.callback(self.requests[id])
		self.requests[id].frame = None
		self.times.append(latency)
		self.idleIds.append(id)
		self.cv.notify()
		self.cv.release()

	def getIdleRequest(self):
		self.cv.acquire()
		while len(self.idleIds) == 0:
				self.cv.wait()
		id = self.idleIds.pop();
		self.startTime = min(datetime.now(), self.startTime);
		self.cv.release()
		return self.requests[id]

	def waitAll(self):
		self.cv.acquire()

		while len(self.idleIds) != len(self.requests):
				self.cv.wait()

		self.cv.release()
