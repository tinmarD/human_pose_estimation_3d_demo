#!/usr/bin/env python

import sys
import os
import multiprocessing

from openvino.inference_engine import IECore
from utils import logger

class Device:
	subs = []

	def __init__(self, ie=None, num_requests=None):
		sub = Device.getSub(self.name)
		if sub:
			self.ie = ie
			if isinstance(num_requests, float) and num_requests > 0 and num_requests < 1.0:
				num_requests = max(1, int(sub.maxRequests*num_requests))
			available_requests = sub.maxRequests - sub.totalRequests
			if num_requests == 0:
				num_requests = available_requests;
			self.requestsNumber = min(available_requests, num_requests)
			sub.totalRequests += self.requestsNumber

	@classmethod
	def addSub(cls, scls):
		cls.subs.append(scls)

	@staticmethod
	def create(device_string, ie, num_requests):
		name = device_string
		if ':' in device_string:
			name = device_string.split(':')[0].strip()

		sub = Device.getSub(name)
		if sub:
			return sub(device_string, ie, num_requests);
		
		return None

	@staticmethod
	def getSub(name):
		for sub in Device.subs:
			if sub.name == name:
				return sub

		return None

	def getName(self):
		if hasattr(self, '_name'):
			return self._name
		else:
			return self.name

	def config(self, extension_path=None):
		logger.info("Configuring {}".format(self.getName()))

	def reset(self):
		sub = Device.getSub(self._name)
		if sub:
			sub.totalRequests = 0;
		
		return None
	
	@staticmethod
	def parse(device_string):
	    devices = device_string
	    if ':' in devices:
	        devices = devices.partition(':')[2]
	    return [d[:d.index('(')] if '(' in d else d for d in devices.split(',')]

@Device.addSub
class DeviceCPU(Device):
	name = "CPU"
	maxRequests = multiprocessing.cpu_count()
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		super().__init__(ie, num_requests)

	def config(self, extension_path=None):
		logger.info("Configuring {}".format(self.getName()))
		if extension_path is None:
			extension_path = os.environ['INTEL_OPENVINO_DIR'] + \
				"/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
				
		#self.ie.add_extension(extension_path, self.name)
		self.ie.set_config({'CPU_THROUGHPUT_STREAMS': str(self.requestsNumber) 
							if self.requestsNumber > 0 else 'CPU_THROUGHPUT_AUTO' }, self.getName())

		#self.ie.set_config({'CPU_THREADS_NUM': str(4)}, self.getName())

		#self.ie.set_config({'CPU_BIND_THREAD': 'YES'}, self.getName())

@Device.addSub
class DeviceGPU(Device):
	name = "GPU"
	maxRequests = 0
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		super().__init__(ie, num_requests)

	def config(self, extension_path=None):
		logger.info("Configuring {}".format(self.getName()))
		#self.ie.set_config({'GPU_THROUGHPUT_STREAMS': 'GPU_THROUGHPUT_AUTO'}, self.getName())
		#self.ie.set_config({'CLDNN_PLUGIN_THROTTLE': '1'},  self.getName())

@Device.addSub
class DeviceMYRIAD(Device):
	name = "MYRIAD"
	maxRequests = 0
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		super().__init__(ie, num_requests)

	def config(self, extension_path=None):
		logger.info("Configuring {}".format(self.getName()))
		pass

@Device.addSub
class DeviceFPGA(Device):
	name = "FPGA"
	maxRequests = 0
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		self._name = device_string
		super().__init__(ie, num_requests)

@Device.addSub
class DeviceHDDL(Device):
	name = "HDDL"
	maxRequests = 0
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		super().__init__(ie, num_requests)

	def config(self, extension_path=None):
		logger.info("Configuring {}".format(self.getName()))
		pass

@Device.addSub
class DeviceMULTI(Device):
	name = "MULTI"
	maxRequests = 0
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		self._name = device_string
		super().__init__(ie, num_requests)
		self.devices = []
		for d in Device.parse(device_string):
			self.devices.append(Device.create(d.split('.')[0],ie,num_requests))

	def config(self, extension_path=None):
		for device in self.devices:
			device.config(extension_path=extension_path)
		

@Device.addSub
class DeviceHETERO(Device):
	name = "HETERO"
	maxRequests = 0
	totalRequests = 0

	def __init__(self, device_string, ie, num_requests):
		self._name = device_string
		super().__init__(ie, num_requests)
		self.devices = []
		for d in Device.parse(device_string):
			self.devices.append(Device.create(d.split('.')[0],ie,num_requests))

	def config(self, extension_path=None):
		for device in self.devices:
			device.config(extension_path=extension_path)
		pass
	
