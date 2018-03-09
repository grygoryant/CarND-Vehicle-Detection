import os
import sys
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from FeatureExtractor import FeatureExtractor
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from matplotlib import cm

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
	
def load_and_extract_features(img_paths, feature_extractor):
	features = []
	for path in img_paths:
		image = mpimg.imread(path)
		#image = image.astype(np.float32)/255
		image = (image*255).astype(np.uint8)
		features.append(feature_extractor.process(image))
	return features

def get_sliding_windows(frame_shape, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = frame_shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = frame_shape[0] 
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
	window_list = []
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			window_list.append(((startx, starty), (endx, endy)))
	return window_list#np.array(window_list).astype(np.int64)

def draw_windows(img, bboxes, color=(0, 0, 255), thick=2):
	imcopy = np.copy(img)
	for bbox in bboxes:
		cv2.rectangle(imcopy, tuple(bbox[0]), tuple(bbox[1]), color, thick)
	return imcopy

def add_heat(heatmap, bbox_list):
	for box in bbox_list:
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	return heatmap
    
def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap

def draw_labeled_bboxes(img, labels):
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	return img

def get_colors(inp, colormap, vmin=None, vmax=None):
	norm = plt.Normalize(vmin, vmax)
	m = cm.ScalarMappable(norm=norm, cmap=colormap)
	return m.to_rgba(inp)#colormap(norm(inp))

class VehicleDetector:

	def __init__(self):
		print('Initializing detector...')
		self.classifier = LinearSVC(verbose=True)
		self.X_scaler = StandardScaler()
		self.feature_extractor = FeatureExtractor(color_space='YCrCb', 
			orient=9, hog_channel='ALL')
		self.last_detections = deque(maxlen=20)

	def detect_vehicles(self, img, windows):
		on_windows = []
		for window in windows:
			test_img = cv2.resize(img[window[0][1]:window[1][1], 
				window[0][0]:window[1][0]], (64, 64))
			#plt.imshow(test_img)
			#plt.show()
			features = self.feature_extractor.process(test_img)
			test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
			prediction = self.classifier.predict(test_features)[0].astype(np.int64)
			#print(prediction)
			if prediction == 1:
				on_windows.append(window)
		return on_windows#np.array(on_windows).astype(np.int64)

	def train(self, vehicles_path, non_vehicles_path):
		print('Training...')
		dataset_info = get_data_info(vehicles_path, non_vehicles_path)
		
		car_features = load_and_extract_features(dataset_info['vehicles'], 
			self.feature_extractor)

		non_car_features = load_and_extract_features(dataset_info['non-vehicles'], 
			self.feature_extractor)

		X = np.vstack((car_features, non_car_features)).astype(np.float64)
		y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

		rand_state = np.random.randint(0, 100)
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=rand_state)

		self.X_scaler.fit(X_train)
		X_train = self.X_scaler.transform(X_train)
		X_test = self.X_scaler.transform(X_test)

		print('Feature vector length:', len(X_train[0]))

		self.classifier.fit(X_train, y_train)
		print('Test Accuracy = ', round(self.classifier.score(X_test, y_test), 4))

	def save(self, clf_path, sclr_path):
		print('Saving classifier...')
		joblib.dump(self.classifier, clf_path) 
		joblib.dump(self.X_scaler, sclr_path) 

	def load(self, clf_path, sclr_path):
		print('Loading classifier...')
		self.classifier = joblib.load(clf_path) 
		self.X_scaler = joblib.load(sclr_path)

	def get_merged_detections(self, frame_detections, img, threshold=1):
		heat = np.zeros_like(img[:,:,0]).astype(np.float)
		heat = add_heat(heat, frame_detections)
		heat = apply_threshold(heat,threshold)
		heatmap = np.clip(heat, 0, 255)
		labels = label(heatmap)

		cars = []
		for car_number in range(1, labels[1]+1):
			nonzero = (labels[0] == car_number).nonzero()
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			bbox = ((np.min(nonzerox), np.min(nonzeroy)),
				(np.max(nonzerox), np.max(nonzeroy)))
			cars.append(bbox)

		return cars, heatmap

	def get_avg_detections(self, img):
		last_detections_conc = np.concatenate(np.array(self.last_detections))
		detections, _ = self.get_merged_detections(last_detections_conc, img,
			threshold=min(len(self.last_detections)-1, 15))
		return detections

	def process(self, img):
		window_size = [220, 146, 117, 100]
		y_start_stop = [[440, 660], [414, 560], [400, 517], [390, 490]]
		window_overlap = [0.8, 0.8, 0.8, 0.8]
		wnd_color = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 255, 0)]

		bboxes_overlay = np.zeros_like(img)
		heatmap_overlay = np.zeros_like(img)

		frame_detections = []
		for y_ss, wnd_sz, wnd_olp, color in zip(y_start_stop, window_size, 
									window_overlap, wnd_color):
			windows = get_sliding_windows(img.shape, x_start_stop=[None, None], y_start_stop=y_ss, 
                   xy_window=(wnd_sz, wnd_sz), xy_overlap=(wnd_olp, wnd_olp))
			detections = self.detect_vehicles(img, windows)
			if detections:
				frame_detections.append(detections)

		if frame_detections:
			frame_detections = np.concatenate(frame_detections)
			merged, heatmap = self.get_merged_detections(frame_detections, img, 1)

			if merged:
				self.last_detections.append(merged)

			if self.last_detections:
				detections = self.get_avg_detections(img)
				bboxes_overlay = draw_windows(bboxes_overlay, 
					detections, color=(255,255,0))

			heatmap_overlay = get_colors(heatmap, cm.hot)
			heatmap_overlay = heatmap_overlay[:,:,:3]*255
		else:
			print('No detections in frame!')

		return bboxes_overlay, heatmap_overlay
		

def video():
	detector = VehicleDetector()
	detector.load('./classifier_YCrCb_lin.pkl', './scaler_YCrCb_lin.pkl')

	def process_image(image):
		res = np.copy(image)
		bboxes_overlay, heatmap = detector.process(image)
		small_heatmap = cv2.resize(heatmap, (0,0), fx=0.25, fy=0.25)

		res = cv2.addWeighted(res, 1., bboxes_overlay, 1., 0.)

		x_offset = image.shape[1] - small_heatmap.shape[1] - 10
		y_offset = 10
		res[y_offset:y_offset + small_heatmap.shape[0], 
			x_offset:x_offset + small_heatmap.shape[1]] = small_heatmap
		return res

	output = './project_video_annotated.mp4'
	clip1 = VideoFileClip('./project_video.mp4')#.subclip(10,11)

	#output = './test_video_annotated.mp4'
	#clip1 = VideoFileClip('./test_video.mp4')#.subclip(45,46)

	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(output, audio=False)

def train():
	detector = VehicleDetector()
	#detector.train('./training_data/subset/vehicles_smallset/', 
	#			'./training_data/subset/non-vehicles_smallset/')
	detector.train('./training_data/full/vehicles/', 
				'./training_data/full/non-vehicles/')

	image = mpimg.imread('./test_images/test6.jpg')
	
	detector.process(image)
	
	detector.save('./classifier_YCrCb_lin.pkl', './scaler_YCrCb_lin.pkl')


def test():
	for i in range(1,2):
		img_name = 'test' + str(i) + '.jpg'
		image = mpimg.imread('./test_images/' + img_name)
		detector = VehicleDetector()
		detector.load('./classifier_YCrCb_lin.pkl', './scaler_YCrCb_lin.pkl')
		detector.process(image)

video()



