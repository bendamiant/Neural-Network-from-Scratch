import numpy as np
import random

'''
training_example: a 2-element numpy array [x, y] where x is the 
input image, and y is its label.

translation, rotate, and scaling return the augmented training_example.
'''

def translation(x_shift, y_shift, training_example):
    input_image_vector = training_example[0]
    image_length = int(np.sqrt(len(input_image_vector)))
    input_image = input_image_vector.reshape(image_length, image_length)
    new_image = np.zeros((image_length, image_length))
    for y_new in range(image_length):
        for x_new in range(image_length):
            x_old = x_new - x_shift
            y_old = y_new - y_shift
            if (0 <= x_old < image_length) and (0 <= y_old < image_length):
                new_image[y_new, x_new] = input_image[y_old, x_old]
    return np.array([new_image.reshape(-1), training_example[1]], dtype=object)

def rotation(degree, training_example):
    radians = degree * (np.pi / 180)
    input_image_vector = training_example[0]
    image_length = int(np.sqrt(len(input_image_vector)))
    input_image = input_image_vector.reshape(image_length, image_length)
    new_image = np.zeros((image_length, image_length))
    center = image_length / 2
    for y_new in range(image_length):
        for x_new in range(image_length):
            x_old = (x_new - center) * np.cos(-radians) - (y_new - center) * np.sin(-radians) + center
            y_old = (x_new - center) * np.sin(-radians) + (y_new - center) * np.cos(-radians) + center
            x_old, y_old = int(round(x_old)), int(round(y_old))
            if (0 <= x_old < image_length) and (0 <= y_old < image_length):
                new_image[y_new, x_new] = input_image[y_old, x_old]
    return np.array([new_image.reshape(-1), training_example[1]], dtype=object)
    
def scaling(factor, training_example):
    input_image_vector = training_example[0]
    image_length = int(np.sqrt(len(input_image_vector)))
    input_image = input_image_vector.reshape(image_length, image_length)
    new_image = np.zeros((image_length, image_length))
    center = image_length / 2
    inverse_factor = 1 / factor
    for y_new in range(image_length):
        for x_new in range(image_length):
            x_old = (x_new - center) * inverse_factor + center
            y_old = (y_new - center) * inverse_factor + center
            x_old, y_old = int(round(x_old)), int(round(y_old))
            if (0 <= x_old < image_length) and (0 <= y_old < image_length):
                new_image[y_new, x_new] = input_image[y_old, x_old]
    return np.array([new_image.reshape(-1), training_example[1]], dtype=object)

def apply_random_augmentation(training_example):
    transformation = random.choice([translation, rotation, scaling])
    if (transformation == translation):
        x_shift = random.randint(-4, 4)
        y_shift = random.randint(-4, 4)
        return transformation(x_shift, y_shift, training_example)
    elif (transformation == rotation):
        degree = random.randint(-20, 20)
        return transformation(degree, training_example)
    else:
        scaling_factor = random.randint(8, 12) / 10
        return transformation(scaling_factor, training_example)
    
def augment_training_images(training_set, augmented_fraction):
    print(f"training_set.shape: {training_set.shape}\n")
    num_to_augment = int(augmented_fraction * len(training_set))
    augmented_set = []
    for training_example in training_set[0:num_to_augment]:
       augmented_set.append(apply_random_augmentation(training_example))
    augmented_training_set = np.concatenate([training_set, np.array(augmented_set)], axis=0)
    print(f"augmented_training_set.shape: {augmented_training_set.shape}\n")
    return augmented_training_set