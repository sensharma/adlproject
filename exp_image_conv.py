# import keras
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K

# import torch
# from models.cnn import Net
# from paths import MNIST_CNN_PATH

import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import compile_func
from deeplift.util import get_integrated_gradients_function

import shap

import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utils import scale_zero_one, viz_scores
from paths import SCORES_PATH

# import data.dataloaders as dl
import data.keras_dataloaders as kdl

path = os.path.join(os.path.dirname(__file__), 'saved_models')

# loading keras cnn for DeepLIFT and KernelSHAP
# saved_model_file = 'keras2_mnist_cnn_allconv.h5'
saved_model_file = os.path.join(path, 'keras_deeplift_model')
keras_model = keras.models.load_model(saved_model_file)
keras_model.summary()

# loading pytorch cnn model for DeepSHAP
# use_cuda = torch.cuda.is_available()
# device = "cuda" if use_cuda else "cpu"

# pytorch_model = Net()
# checkpoint = torch.load(f'{MNIST_CNN_PATH}/checkpoint', map_location=device)
# pytorch_model.load_state_dict(checkpoint['model_state_dict'])
# print(pytorch_model)
# pytorch_model.to(device)

# numpy array of indices, can add random images here
# data_idxs = np.array([10, 20])
data_idxs = np.array([5846])
X_train, y_train, X_test, y_test, rX_test, ry_test = kdl.keras_mnist_loader(
    data_idxs)

X_test = X_test / 255
rX_test = rX_test / 255


# DeepLIFT RevealCancel
revealcancel_model = kc.convert_model_from_saved_files(
    h5_file=saved_model_file, nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel
)

# grad_model required for integrated_gradient
grad_model = kc.convert_model_from_saved_files(
    h5_file=saved_model_file, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient
)

deeplift_model = revealcancel_model
deeplift_prediction_func = compile_func(
    [deeplift_model.get_layers()[0].get_activation_vars()],
    deeplift_model.get_layers()[-1].get_activation_vars(),
)

revealcancel_func = revealcancel_model.get_target_contribs_func(
    find_scores_layer_idx=0, target_layer_idx=-2
)

gradient_func = grad_model.get_target_multipliers_func(
    find_scores_layer_idx=0, target_layer_idx=-2
)

integrated_grads_10 = get_integrated_gradients_function(gradient_func, 10)

# kernel shap -> 0s as background
ker_shap_background = np.zeros((1, 784))
# kernel shap predict function format


def f(z):
    return keras_model.predict(z.reshape(z.shape[0], 28, 28, 1))


ker_exp = shap.KernelExplainer(f, ker_shap_background)


SCORING_METHODS = [
    ("revealcancel", "DL_RC", revealcancel_func),
    ("integrated_grads_10", "IG_10", integrated_grads_10),
    # ("deep_shap", "d_shap", None),
    ("kernel_shap", "k_shap", None),
]

# image_scores = []
method_scores = dict.fromkeys(['scores', 'labels'])
method_scores['scores'] = {}

# for i in range(len(ry_test)):
for count, (method_name, _, score_func) in enumerate(SCORING_METHODS):
    if method_name in ["revealcancel", "integrated_grads_10"]:
        print(f"Computing scores for method_name {method_name}")
        # method_scores['scores'][method_name] = dict.fromkeys(['scores', 'labels'])
        method_scores['scores'][method_name] = []
        for task_idx in range(10):
            # task_idx = int(ry_test[count])
            print("\tComputing scores for digit: " + str(task_idx))
            scores = np.array(
                score_func(
                    task_idx=task_idx,
                    input_data_list=[rX_test],  # [X_test],
                    input_references_list=[np.zeros_like(rX_test)],
                    batch_size=1,  # 1000,
                    progress_update=None))
            method_scores['scores'][method_name].append(scores)

method_scores['labels'] = ry_test
# image_scores.append(method_scores)

# How to access
# lbl0 = method_scores['labels'][0]
# print(method_scores['scores']['revealcancel'][lbl0].shape)

# DeepSHAP
# DeepSHAP can average across images to create background, 100 works. Providing from training data
# deep_shap_background = torch.Tensor(X_train[:100]).unsqueeze(1)
# deep_shap_background = deep_shap_background/255

# could also use 0s as background
# deep_shap_background = torch.zeros_like(images[0:1]).to(device)

# deep_exp = shap.DeepExplainer(pytorch_model, deep_shap_background.to(device))

'''
# For deepshap -> requires TF2
# deep_exp = shap.DeepExplainer(keras_model, deep_shap_background)
print("\n \n Computing scores for DeepSHAP:")
method_name = "deep_shap"
# for i in range(len(ry_test)):
    # d_shap_values = deep_exp.shap_values(rX_test_ds)
    # d_shap_values = [np.swapaxes(np.swapaxes(d, 1, -1), 1, 2)
    #                 for d in d_shap_values]
# rX_test_ds = torch.Tensor(rX_test).permute(1, -1).permute(-1, -2)
rX_test_ds = np.swapaxes(np.swapaxes(rX_test, 1, -1), -2, -1)
rX_test_ds = torch.Tensor(rX_test_ds)
d_shap_values = deep_exp.shap_values(rX_test_ds.to(device))
method_scores['scores'][method_name] = d_shap_values
'''

"""
# Should not be required any more?
keys = np.arange(len(d_shap_values))
keys = [str(key) for key in keys]
ds_dict = dict.fromkeys(keys)
for key in keys:
    ds_dict[key] = d_shap_values[int(key)]
method_scores[method_name] = ds_dict
"""

# KernelSHAP
# reshaping as per kernel shap requirements
print("\n \n Computing scores for KernelSHAP:")
method_name = 'kernel_shap'
rX_test_ker = rX_test.copy()
rX_test_ker = rX_test_ker.squeeze().reshape(rX_test_ker.shape[0], -1)
k_shap_values = ker_exp.shap_values(
    rX_test_ker, nsamples=20000)  # runs model nsamples no. of times, paper = 50000 

k_shap_values = [ks.reshape(scores.shape[0],
                            int(ks.shape[1]**0.5),
                            int(ks.shape[1]**0.5),
                            1) for ks in k_shap_values]

method_scores['scores'][method_name] = k_shap_values

"""
keys = np.arange(len(k_shap_values))
keys = [str(key) for key in keys]
ks_dict = dict.fromkeys(keys)
for key in keys:
    ks_dict[key] = k_shap_values[int(key)]
method_scores[method_name] = ks_dict
"""


def get_masked_image(idx, scores, task_1, task_2, n_to_erase):
    difference = scores[(task_1)][idx].ravel() - \
        scores[(task_2)][idx].ravel()
    # highlight the top n
    """Note: UP TO n_to_erase, as long as diff > 0"""
    top_nth_threshold = max(
        sorted(difference, key=lambda x: -x)[n_to_erase], 0.0)
    thresholded_points = 1.0 * (difference <= top_nth_threshold)
    masked_inp = thresholded_points.reshape(
        28, 28, 1) * rX_test[idx]  # orig: X_test
    return masked_inp


# Function to plot the result of masking on a single example, for converting to anoother digit
def plot_mask_figures(idx, task_1, task_2, method_names, n_to_erase):
    print("example index: " + str(idx))
    print(
        "Columns:",
        "task " + str(task_1) + " scores;",
        "task " + str(task_2) + " scores;",
        str(task_1) + "->" + str(task_2) + " masking;",
    )

    print("Order of the methods is: " + ", ".join(str(x)
                                                  for x in method_names))
    for method_name in method_names:
        scores = method_scores['scores'][method_name]
        mean_scores_over_all_tasks = np.mean(
            np.array([scores[i][idx] for i in range(10)]), axis=0
        )
        f, axarr = plt.subplots(1, 4, sharey=False, figsize=(15, 10))
        viz_scores(rX_test[idx], axarr[0])  # orig: X_test
        viz_scores(scores[task_1][idx] -
                   mean_scores_over_all_tasks, axarr[1])
        viz_scores(scores[task_2][idx] -
                   mean_scores_over_all_tasks, axarr[2])
        viz_scores(get_masked_image(
            idx, scores, task_1, task_2, n_to_erase), axarr[3])
    plt.show()


n_to_erase = 157  # approx 20% [of 784 (28 x 28)] = 157

# method_names = ["revealcancel", "integrated_grads_10", "deep_shap", "kernel_shap"]
method_names = ["revealcancel", "integrated_grads_10", "kernel_shap"]

# Not so good 8->3; 8->6 example
# plot_two_way_figures(61,8,3,6,method_names,n_to_erase)

# Good 8->3; 8->6 example
# plot_two_way_figures(5846,8,3,6,method_names,n_to_erase)

# Bad 3->8; 3->9 example; doesn't work very well when switching from the
# minority class to the majority class
# (in terms of proximity to 0s and 1s)
shift_map = {
    0: 8,
    1: 7,
    2: 5,
    3: 1,
    4: 1,
    5: 6,
    6: 8,
    7: 1,
    8: 3,
    9: 4,
}

for idx, label in enumerate(ry_test):
    plot_mask_figures(idx, label, shift_map[label],
                      method_names, n_to_erase)   # orig index 5846, not 5
