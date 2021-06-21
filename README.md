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
- Python xml_to_csv.py
- Change the generate_tfrecord.py
- python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/

### Training

- Download Config File https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
- Send file to model/research/object_detection
- Edit config file
- Run python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config –logtostderr in python ide terminal.

### Testing
- Copy frozen_inference_graph.pb file and paste a new file
- Create .py file to new file

### Python Code
        import numpy as np
    import tensorflow as tf
    import cv2 as cv
    canli = cv.VideoCapture('video.mp4')
    # Read the graph.
    with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')


        while(canli.isOpened()):
            ret,img=canli.read()     
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB 
        # Read and preprocess an image.
        # img = cv.imread('example.jpg')
        # rows = img.shape[0]
        # cols = img.shape[1]
        # inp = cv.resize(img, (300, 300))
        # inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.5:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    print(x)
                    cv.imshow('TensorFlow MobileNet-SSD', img)
                    if cv.waitKey(20)  &  0xFF==('q0'):
                        break








