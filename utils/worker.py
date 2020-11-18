#!/usr/bin/env python


import os
import sys

from threading import Thread, Condition

from .logger import logger

class Worker:
	def __init__(self, name=None, handler=None, debug=True):
		self.cv = Condition()
		self.running = False; 
		self.name = name
		self.debug = debug
		self._handler = handler

	def lock(self):
		self.cv.acquire()

	def unlock(self):
		self.cv.release()

	def wait(self, timeout=None):
		self.cv.wait(timeout)
	
	def notify(self):
		self.cv.notify()
			
	def start(self):

		self.lock()
		
		if self.running == False:
			target = self.__handler
			if self._handler:
				target = self._handler
		
			self.proc = Thread(target=target)
			self.proc.daemon = True
			self.running = True;
			self.onStart()
			self.proc.start()

		self.unlock()

	def stop(self):
		self.lock()
		if self.running:
			self.running = False;
			self.cv.notify()
			self.unlock()
			self.proc.join()
			self.lock()
			self.onStop()
		self.unlock()

	def onStart(self):
		pass

	def onStop(self):
		pass	
	
	def handler(self):
		raise NotImplementedError()

	def __handler(self):
		if self.debug and self.name:
			logger.info("{} thread started".format(self.name))
		self.handler()
		if self.debug and self.name:
			logger.info("{} thread ended".format(self.name))
		self.running = False;
		
	def isRunning(self, flag=True):
		return (self.running and (flag))
