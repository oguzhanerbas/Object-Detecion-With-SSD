# Object-Detecion-With-SSD
Object Detecion With SSD

### Install
- Python 3.7
- Tensorflow 1.15 (pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim) https://www.tensorflow.org/versions
-  Were used in this project.

### Tensorflow
- https://github.com/tensorflow/models
- Download the data in python project.

### Protobuf
- https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0
- Download the available version for your pc.
- Extract to models-master (only protoc.exe)
- protoc object_detection/protos/*.proto --python_out=. (This code should run in, where protoc exe, with python terminal)
- also run python setup.py build and python setup.py install

### Run
- Run webcam_detection.py
- Install Spyder, Numpy, Opencv
- pip install tf_slim

### Labelimg
- Create "custom object" file in models-master
- Create "data, images, training" file in custom object
- Create "test and train" file in the images file
- Download https://github.com/tzutalin/labelImg
- Extract to custom object file
- conda install pyqt=5 
- conda install -c anaconda lxml
- pyrcc5 -o libs/resources.py resources.qrc
- Python labelImg.py
- Detect images what you want to detect with labelimg
- Seperate images (test and train)

### Training
- Python xml_to_csv.py
- Change the generate_tfrecord.py
- python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/

