import os
import sys
import time
import queue
from queue import Queue, Empty

from .logger import logger
from .worker import Worker
from .worker import Worker

class Time:
	def __init__(self,name):
		self.name = name
		self.ts = time.time()

class Sink(Worker):
	def __init__(self, name=None, maxsize=1000, order=False):
		Worker.__init__(self, name=name)
		self.__queue = Queue(maxsize=maxsize)
		self.enabled = True
		self.index = 0
	
	def put(self, obj):
		if self.enabled:
			if 'index' not in obj:
				obj['index'] = self.index
				self.index += 1
			self.__queue.put(obj)

	def get(self, timeout=None):
		if self.enabled:
			try:
				obj = self.__queue.get(timeout=timeout)
				if obj is not None:
					if 'times' not in obj:
						obj['times'] = []
					obj['times'].append(Time(self.name))
				return obj;
			except Exception as error:
				return None
				
	def empty(self):
		return self.__queue.empty()

	def size(self):
		return self.__queue.qsize()

	def enable(self, flag):
		self.lock()
		self.enabled = flag
		self.notify()
		self.unlock()
		if flag == False:
			self.flush()

	def flush(self):
		self.lock()
		try:
			while True:
				self.__queue.get_nowait()
		except Empty:
			pass
		self.unlock()
	#		
	def handle(self, dict):
		raise NotImplementedError()

	def handler(self):
		self.lock()
		while self.running:
			self.unlock()
			data=self.get(0.5)
			while data is None and self.running:
				self.lock()
				self.wait(0.1)
				self.unlock()
				data=self.get(0.5)

			self.lock()

			if data and self.running:
				self.unlock()
				self.handle(data)
				self.lock()

		self.unlock()
