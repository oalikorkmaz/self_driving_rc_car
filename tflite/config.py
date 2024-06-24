import os
cwd = os.getcwd()

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = f'{cwd}/signdetection/train'
VALID_DATASET_PATH = f'{cwd}/signdetection/valid'
MODEL_PATH = f'{cwd}/model'

MODEL = 'efficientdet_lite0'
MODEL_NAME = 'sign2.tflite'
CLASSES = ['-', 'Roboflow is an end-to-end computer vision platform that helps you', 'This dataset was exported via roboflow.com on June 28- 2023 at 4-46 PM GMT', 'lighttraffic1 - v10 2023-05-16 12-32pm']
EPOCHS = 40
BATCH_SIZE = 6
