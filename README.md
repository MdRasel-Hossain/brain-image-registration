# Brain Image Registration and 3D Reconstruction

**Computer Vision | Academic Project | Heinrich Heine University Düsseldorf**

## 1. Project Overview

This project explores image registration techniques for aligning consecutive 2D brain scan images as part of a 3D reconstruction workflow. The primary objective is to estimate geometric transformations between adjacent image slices and improve their alignment using both manually selected and automatically detected image features.
The project investigates image processing, feature detection, feature matching, robust transformation estimation, and image registration.

## 2. Technologies and Libraries

- **Programming Language:** Python
- **Image Processing:** OpenCV, scikit-image, Pillow, tifffile
- **Numerical Computing:** NumPy, SciPy
- **Visualization:** Matplotlib, napari
- **Development Environment:** Google Colab, Jupyter Notebook

### 3.1 Manual Image Registration

The initial stage explores image alignment using corresponding keypoints between adjacent brain scan images.

The workflow includes:

- Loading and visualizing brain scan images.
- Examining image properties and intensity statistics.
- Identifying corresponding keypoints.
- Estimating rigid and affine transformations.
- Applying geometric transformations to align images.

### 3.2 Image Processing and Enhancement

Several image processing techniques are investigated to prepare the images for registration.

These include:

- Image downsampling and bilinear interpolation.
- Histogram equalization.
- Gaussian, mean, and median filtering.
- Sobel edge detection.
- Anti-aliasing before downsampling.

### 3.3 Automatic Feature Detection and Matching

Automatic feature-based registration is explored using SIFT (Scale-Invariant Feature Transform).

The workflow includes:

1. Detecting image keypoints and extracting descriptors.
2. Matching features between adjacent images.
3. Identifying reliable correspondences.
4. Using RANSAC to reject outlier matches.
5. Estimating geometric transformations.
6. Applying transformations to align the images.

### 3.4 Reconstruction Workflow

The final stage investigates the registration of consecutive 2D brain scan slices using the techniques developed in the preceding stages.

The complete 3D reconstruction and its evaluation require further verification before reporting a final result.

**brain_image_registration.ipynb:** Main notebook containing experiments, visualizations, image processing and registration workflows.

**functions_student.py:** Supporting Python implementations for transformations, filtering, feature extraction, matching, and RANSAC.

## 5. Running the Project

The project was developed using Google Colab.

1. Open `brain_image_registration.ipynb` in Google Colab.
2. Make the required Python dependencies available.
3. Place `functions_student.py` in the notebook's working directory.
4. Provide the required image data using the directory structure expected by the notebook.
5. Run the notebook cells in sequence.

**Dataset availability:** The brain scan images were provided for university coursework and are not included in this repository. Access and redistribution are subject to the original data owner's permissions.

**Reproducibility note:** The complete pipeline has not yet been independently verified from a clean environment.

## 6. Results and Evaluation

The notebook contains visual experiments involving image transformations, filtering, feature matching, and registration.

A quantitative evaluation of registration accuracy, reconstruction quality, and runtime performance has not yet been finalized for this README.

## 7. Limitations and Future Improvements

- Verify the full registration workflow from a clean environment.
- Evaluate alignment quality using appropriate quantitative metrics.
- Compare manual and automatic registration approaches.
- Improve robustness to incorrect feature matches.
- Complete and validate the final 3D reconstruction workflow.
- Document reproducible experiments and dependencies.

## 8. Academic Context

This project was completed as part of Computer Vision coursework at Heinrich Heine University Düsseldorf.

The project follows a university-provided assignment and was developed in a group setting. University-provided materials and individual contributions should be credited appropriately before public release.

## Dataset and Acknowledgements

The brain scan images used in this project were provided by a research group at Heinrich Heine University Düsseldorf (HHU) for educational purposes as part of the Computer Vision coursework.

The dataset is not included in this repository because redistribution permission has not been confirmed.

The project explores image registration techniques using university-provided data and assignment materials.
GitHub: [MdRasel-Hossain](https://github.com/MdRasel-Hossain)
