

import os
import sys
import time
from datetime import datetime
from statistics import mean
import cv2

from utils import Sink

class L(list):
		def __init__(self, maxlen=100):
			self.maxlen = maxlen
		def add(self, item):
			list.append(self, item)
			if len(self) > self.maxlen: del self[0]
			return mean(self)

class PerfMonitor(Sink):
	def __init__(self, sinks=[]):
		self.sinks = sinks
		self.startTime = time.time()
		self.framesNumber = 0
		self.times = {} 
		super().__init__("PerfMonitor")

	def handler(self):
		while self.running:
			input = self.get(timeout=0.5)
			if input is not None:
				if self.startTime is None:
					self.startTime = input['ts']
				self.framesNumber += 1 
				duration = (time.time()-self.startTime)
				fps = self.framesNumber/duration
				self.draw(input['original_frame'], fps=fps, times=input['times'])
				for sink in self.sinks:
					if callable(sink):
						sink(input['original_frame'], input)
					else:
						sink.put(input)

	def draw(self, frame, fps=None, latency=None, times=None):
		fontFace =  cv2.FONT_HERSHEY_SIMPLEX
		frame_size = frame.shape[:-1]
		fontScale = 0.5
		thickness = 1
		bcolor = (0,255,255)
		fcolor = (0,0,255)
		margin = 50

		def drawPerf(label, value, center):
			value = str(value)
			textsize = cv2.getTextSize(value, fontFace, fontScale, thickness)[0]
			radius = int(textsize[0]/2 + 6)
			cv2.circle(frame, center, radius, bcolor, 1, cv2.LINE_AA)
			textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
			cv2.putText(frame, value, textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)
			textsize = cv2.getTextSize(label, fontFace, fontScale, thickness)[0]
			textPos = (int(center[0]  - textsize[0]/2), int(center[1] + radius + textsize[1] + 5))
			cv2.putText(frame, label, textPos, fontFace, fontScale, bcolor, thickness, cv2.LINE_AA)

		def drawTime(time, center, margin, sepLen, textColor=fcolor): 
			textsize = cv2.getTextSize(time.name, fontFace, fontScale, thickness)[0]
			rectP1 = (int(center[0]-margin-textsize[0]/2), int(center[1]-margin-textsize[1]/2))
			rectP2 = (int(center[0]+margin+textsize[0]/2), int(center[1]+margin+textsize[1]/2))
			cv2.rectangle(frame, rectP1, rectP2, bcolor, 1)
			textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
			cv2.putText(frame, time.name, textPos, fontFace, fontScale, textColor, thickness, cv2.LINE_AA)
			textsize = cv2.getTextSize(str(time.duration), fontFace, fontScale, thickness)[0]
			textPos = (int(center[0] - textsize[0]/2), int(rectP2[1] + 5 +textsize[1]))
			cv2.putText(frame, str(time.duration), textPos, fontFace, fontScale, textColor, thickness, cv2.LINE_AA)
			if sepLen > 0:
				cv2.line(frame, (rectP2[0], center[1]), (rectP2[0]+sepLen,center[1]), bcolor, 1)

		if fps is not None:
			fps = f'{fps:04.1f}'
			drawPerf("fps", fps, (margin, frame_size[0]-margin))


		if times is not None:
			latency = (times[-1].ts - times[0].ts)*1000
			latency = '{:04d}'.format(int(latency))
			drawPerf("latency (ms)", latency, (frame_size[1] - margin, frame_size[0]-margin))

		if times is not None:
			sepLen=50
			padding=10
			textLen=0
			nbTimes = len(times)-1
			for t in range(nbTimes):
				if times[t].name not in self.times:
					self.times[times[t].name] = L()
				times[t].duration = int(self.times[times[t].name].add((times[t+1].ts - times[t].ts)*1000))
				textsize = cv2.getTextSize(times[t].name, fontFace, fontScale, thickness)[0]
				textLen += textsize[0]
			dLen = textLen + (nbTimes)*padding*2 + (nbTimes-1)*sepLen
			startX = (frame_size[1] - dLen )/2
			for t in range(nbTimes):
				textsize = cv2.getTextSize(times[t].name, fontFace, fontScale, thickness)[0]
				startX += padding + int(textsize[0]/2)
				if t == nbTimes - 1:
					sepLen = 0
				drawTime(times[t], (startX, frame_size[0]-margin), padding, sepLen, (0,255,0) )
				startX += padding + int(textsize[0]/2) + sepLen

		

