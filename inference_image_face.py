import glob
import io
import sys
from random import randint

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import label_map_util
from utils import visualization_utils_color as vis_util

sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

if len(sys.argv) >= 2:
  image_folders = sys.argv[0:]
else:
  image_folders = ['test_images']

output = open("result.csv", "w+")

images = []
for folder in image_folders:
  images.extend(glob.iglob(folder + '/**/*.jpg', recursive=True))
  images.extend(glob.iglob(folder + '/**/*.jpeg', recursive=True))

with detection_graph.as_default():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=detection_graph, config=config) as sess:
    for image_path in images:
      try:
        image_data = open(image_path, "rb").read()
        image = Image.open(io.BytesIO(image_data))
        rgb_im = image.convert('RGB')
        width = rgb_im.size[0]
        height = rgb_im.size[1]

        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

        boxes_coordinate = vis_util.get_boxes(
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index)

        readable_score = vis_util.get_scores(
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index)

        i = 0
        for box, color in boxes_coordinate:
          y_min, x_min, y_max, x_max = box
          x1 = int(x_min * width)
          y1 = int(y_min * height)
          x2 = int(x_max * width)
          y2 = int(y_max * height)
          row = '{},{},{},{},{},{}'.format(image_path, readable_score[i], x1, y1, x2, y2)
          output.write(row + "\n")
          if randint(0, 30) == 15:
            output.flush()
          i += 1
      except:
        print('error during detection')

output.close()
