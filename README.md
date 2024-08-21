# Plant Recognition

This project is a plant recognition system that uses machine learning techniques to identify various species of plants from images.

## Project Overview

The Plant Recognition system is designed to help botanists, gardeners, and plant enthusiasts identify different plant species quickly and accurately. It uses a trained machine learning model to analyze images of plants and predict their species.

## Repository Overview

This repository leverages BioCLIP to detect plants. It includes a forked version of the BioCLIP repository for batch image processing and features a Python notebook for applying a sliding window technique to drone images. 

## Contents

### 3rdparty Folder

- Contains a fork of the BioCLIP repository. This fork was initially used for batch processing, though the official BioCLIP repository has since incorporated this feature.
- Also contains a clone of the YOLO-World repo, although I never managed to get it working.

### `test.ipynb`

- A Jupyter notebook implementing a "sliding window" technique over images, similar to a convolution operation in CNNs.
- Feeds cropped sections of drone images into BioCLIP.
- You can customize the classes BioCLIP searches for by modifying the `classes` variable in the notebook.

### `models` Directory

- Contains a segmentation model that can be loaded using `torch.load()`.
- This model segments 4096x2048 images into three classes: Foliage, Soil, and Rock.
- Annotated data and training code can be found in the [plant_dataset_Forillon](#) repository.

### `plant_dataset_Forillon` Repository

- This repo contains all the data we have.
- Contains trainign code for the segmentation model as well as annotated data.
- Includes JSON files with information on bounding box positions, dimensions, class info, and source files.
- JSON files are derived from hand-drawn annotations. Note that Parks Canada is in the process of reannotating these files, so use the current files as temporary placeholders.

## Ideas for Future Work

- **YOLO-World Integration**: Investigate replacing YOLO-World's clip text encoder with BioCLIP to see if it improves plant detection. I never manged to get this working but it seems like a promising area.
- **Filtering False Positives**: Filter out positives where the segmentation model indicates only rock or no foliage to reduce false positives.
- **BioCLIP Fine-Tuning**: Fine-tune BioCLIP with new data to improve detection accuracy.
- **Performance Metrics**: Develop and implement proper performance metrics to quantitatively evaluate the effectiveness of different approaches.
