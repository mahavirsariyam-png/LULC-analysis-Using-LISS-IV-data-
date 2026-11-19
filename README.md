# Land Use and Land Cover Classification using LISS-IV Data

A deep learning project that automatically classifies every pixel of ISRO LISS-IV
satellite imagery (5.8 m resolution) of Shillong, Meghalaya into land-cover classes:
agriculture, barren, built-up, forest, grass and water.

## How it works
1. **Data:** LISS-IV scenes from ISRO's Bhoonidhi/Bhuvan portals, composed into
   False Color Composites in QGIS.
2. **Annotation:** 154 images labeled with polygons in the VIA tool, then converted
   to pixel-level masks using a Python/OpenCV pipeline.
3. **Preprocessing:** Resize to 256x256, normalization, and synchronized image-mask
   augmentation (rotation, zoom, shift, flip).
4. **Model:** U-Net with a pretrained VGG16 encoder (transfer learning) and a decoder
   with skip connections, trained with Adam, sparse categorical cross-entropy,
   learning-rate scheduling and early stopping.

## Tech stack
Python, TensorFlow/Keras, OpenCV, NumPy, scikit-learn, Matplotlib, QGIS, Google Colab

## Results
Validation pixel accuracy of about 64.6%. The model performs best on forest and
background; minority classes (agriculture, barren, built-up, grass) are limited by
class imbalance.

## Future work
Tile-based training, class-weighted/Dice loss, NDVI as an input channel, and
multi-year change detection.
