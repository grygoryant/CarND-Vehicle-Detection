import os
import sys

def get_img_file_paths(root):
	files_paths = []
	for root, subdirs, files in os.walk(root):
		for filename in files:
			file_path = os.path.join(root, filename)
			if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
				files_paths.append(file_path)
	return files_paths

def get_data_info(vehicles_path, non_vehicles_path):
	print('===Dataset statistics===')
	print('Vehicles path: ' + vehicles_path)
	print('Non vehicles path: ' + non_vehicles_path)
	if not os.path.exists(vehicles_path):
		print('ERROR: Vehicles dir not exists!')
		return
	if not os.path.exists(non_vehicles_path):
		print('ERROR: Non-vehicles dir not exists!')
		return
	vehicles_file_paths = get_img_file_paths(vehicles_path)
	non_vehicles_file_paths = get_img_file_paths(non_vehicles_path)

	print('Vehicles set size: {}'.format(len(vehicles_file_paths)))
	print('Non-vehicles set size: {}'.format(len(non_vehicles_file_paths)))
	print('========================')

	data_info = {'vehicles': vehicles_file_paths, 
					'non-vehicles': non_vehicles_file_paths}
	return data_info
	

class VehicleDetector:

	def __init__(self):
		print('Initializing detector...')

	def train(self, vehicles_path, non_vehicles_path):
		print('Training...')
		dataset_info = get_data_info(vehicles_path, non_vehicles_path)

	def save_classifier(self, path):
		print('Saving classifier...')

	def load_classifier(self, path):
		print('Loading classifier...')

	def process(self):
		print('Processing...')


detector = VehicleDetector()
detector.train('./training_data/subset/vehicles_smallset/', 
				'./training_data/subset/non-vehicles_smallset/')