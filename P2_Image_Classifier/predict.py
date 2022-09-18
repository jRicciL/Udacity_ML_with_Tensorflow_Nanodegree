import argparse
import os
import json
import tensorflow as tf
import tensorflow_hub as hub
from utils import process_image, load_image, predict

parser = argparse.ArgumentParser(
            description = 'Given an image and a pretrained model,' +
                          'predict the most probable classes of the image')

# Check that the inputs are correct
def check_image_path(image_path: str) -> str:
    if not os.path.isfile(f'./{image_path}'):
        raise argparse.ArgumentTypeError(
            "Please provide a valid path to the image.")
    return image_path

def check_model_path(model_path: str) -> str:
    if not os.path.isfile(f'./{model_path}'):
        raise argparse.ArgumentTypeError(
            "Please provide a valid `h5` file with a pretrained `tf.keras` model.")
    return model_path

def check_labels_path(category_names_json: str) -> str:
    if not os.path.isfile(f'./{category_names_json}'):
        raise argparse.ArgumentTypeError(
            "Please provide a valid `category_names_json` file.")
    return category_names_json

# Parser arguments
parser.add_argument('-i', '--image_path', 
                    required = True, type = check_image_path,
                    help = 'Path to the flower image to be predicted.')
parser.add_argument('-m', '--model_path', 
                    required = True, type = check_model_path,
                    help = 'Path to the pretrained model (.h5 format).')
parser.add_argument('-k', '--top_k_classes', 
                    required = False, default = 5,
                    help = 'Return the top `k` most likely classes.')
parser.add_argument('-c', '--category_names_json', 
                    required = False, default = 'label_map.json',
                    help = 'A json file mapping the category names.')

# Capture the input arguments
args = vars(parser.parse_args())
IMAGE_PATH = args['image_path']
MODEL_PATH = args['model_path']
TOP_K      = args['top_k_classes']
LABEL_MAP  = args['category_names_json']


if __name__ == '__main__':
    # 1) Load the image
    img = load_image(IMAGE_PATH)
    img_array = process_image(img)
    
    # 2) Load the model
    model = tf.keras.models.load_model(MODEL_PATH, 
                                       custom_objects={'KerasLayer': hub.KerasLayer}, 
                                       compile = False)
    
    # 3) Make the prediction
    probs, pred_classes = predict(img_array, model = model, top_k = TOP_K)
    
    # 4) Load class names
    with open(LABEL_MAP, 'r') as f:
        all_class_names = json.load(f)

    # 5) Map the predicted labels and print the predictions
    pred_class_names = [all_class_names.get(key) 
                        for key in pred_classes]
    print('*'*20)
    print(f'Model predictions for image {IMAGE_PATH}:')
    print('*'*20)
    for c, p in zip(pred_class_names, probs):
        print(c.capitalize())
        print('- Prob:', round(p, 4))
        print('-', '|' * int(p*50))
        print()
