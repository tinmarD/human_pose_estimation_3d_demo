#!/usr/bin/env python

import sys
import os

from openvino.inference_engine import IECore

from utils import logger


class Network:
	def __init__(self, ie, model, device, batch_size=1):
		self.ie = ie
		self.batch_size = batch_size
		self.model_xml = model
		self.model_bin = os.path.splitext(model)[0] + ".bin"
		labels = os.path.splitext(model)[0] + ".labels"
		self.labels_map = None
		if os.path.exists(labels):
			with open(labels, 'r') as f:
				self.labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
	
		logger.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, self.model_bin))
		self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
		
		self.inputs = self.net.inputs
		if len(self.inputs) == 0:
			raise AttributeError('No inputs info is provided')

		self.net.batch_size = batch_size

		logger.info("Network batch size: {}".format(self.net.batch_size))

		for key in self.inputs.keys():
				# Should be called before load of the network to the plugin
			self.inputs[key].precision = 'U8'

		logger.info("Loading network for device {} with {} requests number".format(device.getName(), device.requestsNumber))

		self.exec_net = self.ie.load_network(self.net, device.getName(), num_requests=device.requestsNumber)
		
		self.input_blob = next(iter(self.net.inputs))
		self.n, self.c, self.height, self.width = self.net.inputs[self.input_blob].shape

		self.outputs = self.net.outputs
		self.out_blob = next(iter(self.outputs))

		self.layers = self.net.layers

		self.infer_requests = self.exec_net.requests
