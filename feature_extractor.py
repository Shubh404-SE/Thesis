import os
import numpy as np
import cv2
from tqdm import tqdm
import logging
from typing import Optional, Tuple, List
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)

class GLCMExtractor:
    """Extracts GLCM features from preprocessed images."""
    
    def __init__(self, preprocessed_path: str, output_file_path: str, 
                 image_size: Tuple[int, int] = (128, 128),
                 distances: List[int] = [1], 
                 angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 levels: int = 256):
        self.preprocessed_path = preprocessed_path
        self.output_file_path = output_file_path
        self.image_size = image_size
        self.mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.properties = ['energy', 'contrast', 'correlation', 'homogeneity']

    def compute_glcm(self, image: np.ndarray) -> np.ndarray:
        """Computes GLCM for an image."""
        gray_img = img_as_ubyte(image)
        glcm = graycomatrix(gray_img, distances=self.distances, angles=self.angles,
                           levels=self.levels, symmetric=True, normed=True)
        return glcm

    def compute_features(self, glcm: np.ndarray) -> np.ndarray:
        """Computes Haralick-like features from GLCM."""
        features = []
        for prop in self.properties:
            feat = graycoprops(glcm, prop)
            features.append(feat.ravel())
        return np.concatenate(features)

    def process_patient(self, patient_path: str, label: int) -> Optional[dict]:
        """Processes a single patient and returns GLCM features."""
        try:
            patient_features = []
            for mri_type in self.mri_types:
                img_path = os.path.join(patient_path, mri_type, "preprocessed.png")
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img.shape != self.image_size:
                    img = cv2.resize(img, self.image_size)
                
                glcm = self.compute_glcm(img)
                features = self.compute_features(glcm)
                patient_features.append(features)
            
            if not patient_features:
                return None
                
            combined_features = np.concatenate(patient_features)
            patient_id = os.path.basename(patient_path)
            return {'patient_id': patient_id, 'features': combined_features, 'label': label}
        except Exception as e:
            logging.error(f"Error processing {patient_path}: {e}")
            return None

    def run_extraction(self) -> None:
        """Runs GLCM feature extraction for all patients."""
        all_data = []
        for mgmt_status in ["MGMT-", "MGMT+"]:
            mgmt_path = os.path.join(self.preprocessed_path, mgmt_status)
            label = 0 if mgmt_status == "MGMT-" else 1

            if not os.path.exists(mgmt_path):
                logging.warning(f"Skipping {mgmt_status}: directory not found.")
                continue

            patient_folders = sorted(os.listdir(mgmt_path))
            for patient in tqdm(patient_folders, desc=f"Extracting GLCM for {mgmt_status}"):
                patient_path = os.path.join(mgmt_path, patient)
                patient_data = self.process_patient(patient_path, label)
                if patient_data is not None:
                    all_data.append(patient_data)

        if all_data:
            patient_ids = np.array([data['patient_id'] for data in all_data])
            features = np.array([data['features'] for data in all_data])
            labels = np.array([data['label'] for data in all_data])
            np.savez(self.output_file_path, patient_ids=patient_ids, features=features, labels=labels)
            logging.info(f"GLCM feature extraction completed! Saved to {self.output_file_path}")
        else:
            logging.warning("No features were extracted.")

class HOGExtractor:
    """Extracts HOG features from preprocessed images."""
    
    def __init__(self, preprocessed_path: str, output_file_path: str, 
                 image_size: Tuple[int, int] = (128, 128),
                 cell_size: Tuple[int, int] = (8, 8),
                 block_size: Tuple[int, int] = (2, 2),
                 bins: int = 9,
                 epsilon: float = 1e-8):
        self.preprocessed_path = preprocessed_path
        self.output_file_path = output_file_path
        self.image_size = image_size
        self.mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins
        self.epsilon = epsilon

    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes gradient magnitude and orientation."""
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = image.astype(np.float32)
        
        Ex = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Ey = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(Ex ** 2 + Ey ** 2)
        angle = np.arctan2(Ey, Ex) * (180 / np.pi)
        angle[angle < 0] += 180
        
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude /= max_mag
        else:
            logging.warning("No gradients detected (max magnitude = 0)")
        
        return magnitude, angle

    def compute_histograms(self, magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """Computes gradient histograms for cells."""
        h, w = magnitude.shape
        cell_h, cell_w = self.cell_size
        cells_x, cells_y = w // cell_w, h // cell_h
        histograms = np.zeros((cells_y, cells_x, self.bins))

        for i in range(cells_y):
            for j in range(cells_x):
                cell_mag = magnitude[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                cell_ang = angle[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                hist, _ = np.histogram(cell_ang, bins=self.bins, range=(0, 180), weights=cell_mag)
                histograms[i, j] = hist
        
        return histograms

    def normalize_blocks(self, histograms: np.ndarray) -> np.ndarray:
        """Normalizes histograms in blocks."""
        cells_y, cells_x, bins = histograms.shape
        block_h, block_w = self.block_size
        normalized_hog = []

        for i in range(cells_y - block_h + 1):
            for j in range(cells_x - block_w + 1):
                block = histograms[i:i + block_h, j:j + block_w].flatten()
                norm_factor = np.sum(block) + self.epsilon
                normalized_block = np.sqrt(block / norm_factor)
                normalized_hog.extend(normalized_block)

        return np.array(normalized_hog)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extracts HOG features from an image."""
        magnitude, angle = self.compute_gradients(image)
        histograms = self.compute_histograms(magnitude, angle)
        hog_descriptor = self.normalize_blocks(histograms)
        if np.all(hog_descriptor == 0):
            logging.warning("HOG features are all zeros")
        return hog_descriptor

    def process_patient(self, patient_path: str, label: int) -> Optional[dict]:
        """Processes a single patient and returns HOG features."""
        try:
            patient_features = []
            for mri_type in self.mri_types:
                img_path = os.path.join(patient_path, mri_type, "preprocessed.png")
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img.shape != self.image_size:
                    img = cv2.resize(img, self.image_size)
                
                if img.max() == 0:
                    logging.warning(f"Skipping {img_path}: Image is all zeros")
                    continue
                
                hog_feats = self.extract_features(img)
                patient_features.append(hog_feats)
            
            if not patient_features:
                return None
                
            combined_features = np.concatenate(patient_features)
            patient_id = os.path.basename(patient_path)
            return {'patient_id': patient_id, 'features': combined_features, 'label': label}
        except Exception as e:
            logging.error(f"Error processing {patient_path}: {e}")
            return None

    def run_extraction(self) -> None:
        """Runs HOG feature extraction for all patients."""
        all_data = []
        for mgmt_status in ["MGMT-", "MGMT+"]:
            mgmt_path = os.path.join(self.preprocessed_path, mgmt_status)
            label = 0 if mgmt_status == "MGMT-" else 1

            if not os.path.exists(mgmt_path):
                logging.warning(f"Skipping {mgmt_status}: directory not found.")
                continue

            patient_folders = sorted(os.listdir(mgmt_path))
            for patient in tqdm(patient_folders, desc=f"Extracting HOG for {mgmt_status}"):
                patient_path = os.path.join(mgmt_path, patient)
                patient_data = self.process_patient(patient_path, label)
                if patient_data is not None:
                    all_data.append(patient_data)

        if all_data:
            patient_ids = np.array([data['patient_id'] for data in all_data])
            features = np.array([data['features'] for data in all_data])
            labels = np.array([data['label'] for data in all_data])
            np.savez(self.output_file_path, patient_ids=patient_ids, features=features, labels=labels)
            logging.info(f"HOG feature extraction completed! Saved to {self.output_file_path}")
        else:
            logging.warning("No features were extracted.")

class LBPExtractor:
    """Extracts LBP features from preprocessed images."""
    
    def __init__(self, preprocessed_path: str, output_file_path: str, 
                 image_size: Tuple[int, int] = (128, 128),
                 radius: int = 1,
                 n_points: int = 8):
        self.preprocessed_path = preprocessed_path
        self.output_file_path = output_file_path
        self.image_size = image_size
        self.mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.radius = radius
        self.n_points = n_points
        self.n_bins = n_points + 2  # Uniform LBP bins: n_points + 2 (uniform + non-uniform)

    def compute_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Computes LBP features (histogram) for a single image."""
        gray_img = img_as_ubyte(image)
        lbp = local_binary_pattern(gray_img, P=self.n_points, R=self.radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=self.n_bins, range=(0, self.n_bins), density=True)
        return hist

    def process_patient(self, patient_path: str, label: int) -> Optional[dict]:
        """Processes a single patient and returns LBP features."""
        try:
            patient_features = []
            for mri_type in self.mri_types:
                img_path = os.path.join(patient_path, mri_type, "preprocessed.png")
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img.shape != self.image_size:
                    img = cv2.resize(img, self.image_size)
                
                if img.max() == 0:
                    logging.warning(f"Skipping {img_path}: Image is all zeros")
                    continue
                
                lbp_feats = self.compute_lbp_features(img)
                patient_features.append(lbp_feats)
            
            if not patient_features:
                return None
                
            combined_features = np.concatenate(patient_features)
            patient_id = os.path.basename(patient_path)
            return {'patient_id': patient_id, 'features': combined_features, 'label': label}
        except Exception as e:
            logging.error(f"Error processing {patient_path}: {e}")
            return None

    def run_extraction(self) -> None:
        """Runs LBP feature extraction for all patients."""
        all_data = []
        for mgmt_status in ["MGMT-", "MGMT+"]:
            mgmt_path = os.path.join(self.preprocessed_path, mgmt_status)
            label = 0 if mgmt_status == "MGMT-" else 1

            if not os.path.exists(mgmt_path):
                logging.warning(f"Skipping {mgmt_status}: directory not found.")
                continue

            patient_folders = sorted(os.listdir(mgmt_path))
            for patient in tqdm(patient_folders, desc=f"Extracting LBP for {mgmt_status}"):
                patient_path = os.path.join(mgmt_path, patient)
                patient_data = self.process_patient(patient_path, label)
                if patient_data is not None:
                    all_data.append(patient_data)

        if all_data:
            patient_ids = np.array([data['patient_id'] for data in all_data])
            features = np.array([data['features'] for data in all_data])
            labels = np.array([data['label'] for data in all_data])
            np.savez(self.output_file_path, patient_ids=patient_ids, features=features, labels=labels)
            logging.info(f"LBP feature extraction completed! Saved to {self.output_file_path}")
        else:
            logging.warning("No features were extracted.")

if __name__ == "__main__":
    # Example usage
    PREPROCESSED_PATH = "./output/filtered_dataset"
    FEATURES_PATH = "./output/features"
    
    # Create feature extractors
    glcm_extractor = GLCMExtractor(PREPROCESSED_PATH, os.path.join(FEATURES_PATH, "glcm_features.npz"))
    hog_extractor = HOGExtractor(PREPROCESSED_PATH, os.path.join(FEATURES_PATH, "hog_features.npz"))
    lbp_extractor = LBPExtractor(PREPROCESSED_PATH, os.path.join(FEATURES_PATH, "lbp_features.npz"))
    
    # Run feature extraction
    glcm_extractor.run_extraction()
    hog_extractor.run_extraction()
    lbp_extractor.run_extraction() 