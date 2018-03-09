from skimage.feature import hog
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
 
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

class FeatureExtractor:

	def __init__(self, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
		self.color_space = color_space
		self.spatial_size = spatial_size
		self.hist_bins = hist_bins
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.hog_channel = hog_channel
		self.spatial_feat = spatial_feat
		self.hist_feat = hist_feat
		self.hog_feat = hog_feat

	def process(self, img):
		features = []
		if self.color_space != 'RGB':
			if self.color_space == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			elif self.color_space == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
			elif self.color_space == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			elif self.color_space == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			elif self.color_space == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(img)  
		if self.spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=self.spatial_size)
			features.append(spatial_features)
		if self.hist_feat == True:
			hist_features = color_hist(feature_image, nbins=self.hist_bins)
			features.append(hist_features)
		if self.hog_feat == True:
			if self.hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					feat = get_hog_features(feature_image[:,:,channel], 
						self.orient, self.pix_per_cell, self.cell_per_block, 
						vis=False, feature_vec=True)
					hog_features.extend(feat) 
			else:
				hog_features = get_hog_features(feature_image[:,:,self.hog_channel], self.orient, 
					self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
			features.append(hog_features)

		return np.concatenate(features)
















