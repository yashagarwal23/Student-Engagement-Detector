# Tracking Concentration of students in online classes

## Requirements

1. Linux
2. Nvidia GPU

## Geting Started

1. Install [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

2. Make the following modifications in OpenFace:

   a. Replace the FeatureExtraction.cpp in OpenFace/exe/FeatureExtraction with FeatureExtraction.cpp in OpenFace mods folder.

   b. Do the same for SequenceCapture.cpp in OpenFace/lib/local/Utilities/src and for SequenceCapture.h in OpenFace/lib/local/Utilities/include

   c. Go to OpenFace/build and exectute `make`.

3. Delete the folder OpenFace/build/processed

4. Open terminal in OpenFace/build directory and run:

```bash
./bin/FeatureExtraction -wild -device 0 -pose -gaze -2Dfp -3Dfp
```

This starts the video input and starts storing preprocessed data in the OpenFace/build/processed directory.

5. From another terminal in the repository folder, run:

```bash
python predict.py
```
