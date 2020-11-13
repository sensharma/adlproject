import keras
from keras.datasets import mnist
from keras import backend as K

import torch
from models.cnn import Net
from paths import MNIST_CNN_PATH

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
import data.dataloaders as dl
import data.keras_dataloaders as kdl


# loading keras cnn for DeepLIFT and KernelSHAP
saved_model_file = "keras2_mnist_cnn_allconv.h5"
keras_model = keras.models.load_model(saved_model_file)
keras_model.summary()

# loading pytorch cnn model for DeepSHAP
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

pytorch_model = Net()
checkpoint = torch.load(f'{MNIST_CNN_PATH}/checkpoint', map_location=device)
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
print(pytorch_model)
pytorch_model.to(device)


len_test_set = 10000
test_indices = np.arange(len_test_set)
len_rand_sample = 60
rand_indices = np.random.choice(
    test_indices, size=len_rand_sample, replace=False)

# loading the random selectin of test items from keras
X_train, y_train, X_test, y_test, rX_test, ry_test = kdl.keras_mnist_loader(rand_indices)

# loading the random selectin of test items from pytorch
# rand_test_loader_pt = dl.mnist_loader(train=False,
#                                       batch_size=len_rand_sample,
#                                       randomtestsample=True,
#                                       rand_indices=rand_indices)

# # loading this to provide background for DeepSHAP
# train_loader_pt = dl.mnist_loader(train=True,
#                                   batch_size=160)

# scaling to [0, 1]
X_test = X_test / 255
rX_test = rX_test / 255

# rX_test_ds, _ = next(iter(rand_test_loader_pt))
# rX_test_ds = rX_test_ds.to(device)
# rX_test_ds = rX_test_ds / 255
# rX_test_Tensor = torch.from_numpy(rX_test_ds).to(device)
# X_inp = X_test[100:200]

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


def predict_keras(z):
    return keras_model.predict(z.reshape(z.shape[0], 28, 28, 1))

def predict_pytorch(z):
    return pytorch_model(z.reshape(z.shape[0], 1, 28, 28))

ker_exp = shap.KernelExplainer(predict_keras, ker_shap_background)

# DeepSHAP can average across images to create background, 100 works. Providing from training data
# images, _ = next(iter(train_loader_pt))
# images = images / 255
# deep_shap_background = images[:100]
deep_shap_background = np.expand_dims(X_train[:100], axis=1)
deep_shap_background = deep_shap_background/255
# could also use 0s as background
# deep_shap_background = torch.zeros_like(images[0:1]).to(device)

deep_exp = shap.DeepExplainer(pytorch_model, deep_shap_background.to(device))

# SCORING_METHODS = [
#     ("revealcancel", "DL_RC", revealcancel_func),
#     ("integrated_grads_10", "IG_10", integrated_grads_10),
#     ("deep_shap", "d_shap", None),
# ]

SCORING_METHODS = [
    ("revealcancel", "DL_RC", revealcancel_func),
    ("integrated_grads_10", "IG_10", integrated_grads_10),
    ("deep_shap", "d_shap", None),
    ("kernel_shap", "k_shap", None),
]

# load score files if found, else create them
# try:
#     method_scores = OrderedDict()
#     for method in SCORING_METHODS:
#         score_file = f"{SCORES_PATH}/{method[1]}.npz"
#         scores = np.load(score_file)
#         method_scores[f"{method[0]}"] = scores
# except FileNotFoundError:
method_scores = OrderedDict()

for method_name, _, score_func in SCORING_METHODS:
    if method_name in ["revealcancel", "integrated_grads_10"]:
        print("Computing scores for:", method_name)
        method_scores[method_name] = {}
        for task_idx in range(10):
            print("\tComputing scores for digit: " + str(task_idx))
            scores = np.array(
                score_func(
                    task_idx=task_idx,
                    input_data_list=[rX_test],  # [X_test],
                    input_references_list=[np.zeros_like(rX_test)],
                    batch_size=5,  # 1000,
                    progress_update=None))
            method_scores[method_name][str(task_idx)] = scores

# DeepSHAP
print("\n \n Computing scores for DeepSHAP:")
method_name = "deep_shap"
rX_test_ds = np.swapaxes(np.swapaxes(rX_test, 1, -1), -1, -2)
d_shap_values = deep_exp.shap_values(rX_test_ds)
d_shap_values = [np.swapaxes(np.swapaxes(d, 1, -1), 1, 2)
                 for d in d_shap_values]

keys = np.arange(len(d_shap_values))
keys = [str(key) for key in keys]
ds_dict = dict.fromkeys(keys)
for key in keys:
    ds_dict[key] = d_shap_values[int(key)]
method_scores[method_name] = ds_dict

# KernelSHAP
# reshaping as per kernel shap requirements
print("\n \n Computing scores for KernelSHAP:")
method_name = 'kernel_shap'
rX_test_ker = rX_test.copy()
rX_test_ker = rX_test_ker.squeeze().reshape(rX_test_ker.shape[0], -1)
k_shap_values = ker_exp.shap_values(
    rX_test_ker, nsamples=1000)  # runs model nsamples times

k_shap_values = [ks.reshape(scores.shape[0],
                            int(ks.shape[1]**0.5),
                            int(ks.shape[1]**0.5),
                            1) for ks in k_shap_values]

keys = np.arange(len(k_shap_values))
keys = [str(key) for key in keys]
ks_dict = dict.fromkeys(keys)
for key in keys:
    ks_dict[key] = k_shap_values[int(key)]
method_scores[method_name] = ks_dict

# outfile = f"{SCORES_PATH}/{short_method_name}.npz"
# np.savez(outfile, **method_scores[method_name])

# load saved scores
# Function that masks out UP TO top n pixels where the score for
# task_1 is higher than the score for task_2


# Function to compute change in log-odds scores after
# pixels have been masked to convert from original_class to target_class
"""AFTER pixels have been masked -> which pixels and how?"""


def compute_delta_log_odds(X, y, method_name, predict_func, imp_scores,
                           original_class, target_class,
                           num_perturbations):
    ''''original class mask is the numpy mask (True/False) for original class, obtained from the labels'''
    original_class_mask = ry_test == original_class

    '''selects training data based on the mask -> np.compress'''
    X = np.compress(condition=original_class_mask, a=X, axis=0)
    # X_pt = np.swapaxes(np.swapaxes(X, 1, -1), 2, -1)
    # X_pt = torch.from_numpy(X_pt).float().to(device)

    """predictions: using the original test image"""
    # compute log-odds of model for those two classes (predict function is "pre_softmax", i.e., outputs before feeding to softmax)
    ''' What if I change the predict function to a function that return the same from pytorch?'''

    if method_name in ['revealcancel', 'integrated_grads_10']:
        predictions = np.array(deeplift.util.run_function_in_batches(predict_func,
                                                                     input_data_list=[
                                                                         X],
                                                                     batch_size=X.shape[0], progress_update=None))   # orig batch_size = 200
        orig_log_odds = predictions[:, original_class] - \
            predictions[:, target_class]
    elif method_name == 'deep_shap':
        predictions = keras_model.predict(X)
        # predictions = pytorch_model(X_pt).cpu().detach().numpy()
        orig_log_odds = np.log(
            predictions[:, original_class] / predictions[:, target_class])
    elif method_name == 'kernel_shap':
        predictions = keras_model.predict(X)
        orig_log_odds = np.log(
            predictions[:, original_class] / predictions[:, target_class])

    ''' Why are the "difference in predictions" from pre-softmax linear layer for original class vs target class log odds?'''
    # print(len(predictions))
    # print(orig_log_odds.shape)
    # make num_perturbations to move from original_class
    # to target_class according to imp_scores
    # first, get the difference of imp_scores for the two classes

    diff_of_scores = (np.compress(original_class_mask, imp_scores[str(original_class)].reshape((-1, 784)), axis=0) -
                      np.compress(original_class_mask, imp_scores[str(target_class)].reshape((-1, 784)), axis=0))
    modified_inp = []

    """This is the masking -> 0-mask for MNIST"""
    """modified_inp -> list of masked images"""
    # then, for each example, sort the scores and zero out indices

    # if method_name == 'deep_shap':
    #     X = X_pt.cpu().detach().numpy()
    #     X = np.swapaxes(np.swapaxes(X, 1, -1), 1, 2)

    for inp, diff_of_scores in zip(X, diff_of_scores):
        top_nth_threshold = max(
            sorted(diff_of_scores, key=lambda x: -x)[num_perturbations], 0.0)
        thresholded_points = 1.0*(diff_of_scores <= top_nth_threshold)
        modified_inp.append(thresholded_points.reshape(28, 28, 1)*inp)
    modified_inp = np.array(modified_inp)
    # print(f"method name: {method_name}")
    # print(f"modified input shape: {modified_inp.shape}")

    # if method_name == 'deep_shap':
    #     modified_inp = np.swapaxes(np.swapaxes(modified_inp, 1, -1), 2, -1)
    #     modified_inp = torch.from_numpy(modified_inp).float().to(device)

    """new_predictions: for the masked images, using trained classifier"""
    # assess change in log-odds for the modified images
    if method_name in ['revealcancel', 'integrated_grads_10']:
        new_predictions = np.array(deeplift.util.run_function_in_batches(predict_func,
                                                                         input_data_list=[
                                                                             modified_inp],
                                                                         batch_size=X.shape[0], progress_update=None))   # orig batch_size = 200
        new_log_odds = new_predictions[:, original_class] - \
            new_predictions[:, target_class]
    elif method_name == 'deep_shap':
        new_predictions = keras_model.predict(modified_inp)
        # new_predictions = pytorch_model(modified_inp).cpu().detach().numpy()
        new_log_odds = np.log(
            new_predictions[:, original_class] / new_predictions[:, target_class])
    elif method_name == 'kernel_shap':
        new_predictions = keras_model.predict(modified_inp)
        new_log_odds = np.log(
            new_predictions[:, original_class] / new_predictions[:, target_class])

    # print(
    #     f"orig_lo shape: {orig_log_odds.shape}, new_lo shape: {new_log_odds.shape}")
    to_return = orig_log_odds - new_log_odds
    return (to_return,
            sum(new_log_odds < 0.0)/float(len(new_log_odds)),
            new_predictions[:, [original_class, target_class]],
            predictions[:, [original_class, target_class]])


# pre_softmax_func computes the output of the linear layer preceding the
# Final softmax noninearity
pre_softmax_func_inner = K.function([keras_model.input, K.learning_phase()],
                                    [keras_model.layers[-2].output])


def pre_softmax_func(x): return pre_softmax_func_inner(x+[False])[0]


def barplot_scores(original_class, target_class, scoring_methods, n_to_erase):
    # def barplot_scores(rX_test, rX_test_ds, ry_test, scoring_methods, n_to_erase):

    # print("converting: "+str(original_class)+"->"+str(target_class))
    method_names = [x[0] for x in scoring_methods]
    short_names = [x[1] for x in scoring_methods]
    # original_class_mask = ry_test == original_class
    scores_to_plot = []
    fig, ax = plt.subplots(figsize=(2*len(method_names), 5))
    """
    logodds obtained here by calling compute_delta_log_odds.
    - what is predict_func=pre_softmax_fuc ? -> linear layer prior to softmax
    """
    for method_name in method_names:
        logodds_diff, flipped, new_predictions, old_predictions = compute_delta_log_odds(
            X=rX_test, y=ry_test,
            method_name=method_name,
            predict_func=pre_softmax_func,
            imp_scores=method_scores[method_name],
            original_class=original_class,
            target_class=target_class,
            num_perturbations=n_to_erase)
        # figure out indices with big shifts
        # retained_indices = np.compress(condition=original_class_mask, a=np.arange(len(y_test)))

        # sorted_shifts
        # sorted_shifts = sorted(enumerate(
        # zip(logodds_diff, new_predictions, old_predictions)), key=lambda x: -x[1][0])
        # print("top indices for "+str(method_name)+": "+" ".join([str(retained_indices[x[0]]) for x in sorted_shifts[:10]]))
        scores_to_plot.append(logodds_diff)
    ax.boxplot(scores_to_plot, widths=[
               0.5 for x in method_names], showmeans=True, meanline=True)
    # ax.set_ylim(-1000,17000)
    ax.set_ylabel("Change in log-odds")
    ax.set_xticklabels(short_names)
    plt.title(str(original_class)+" --> "+str(target_class), fontsize=24)
    plt.tick_params(labelsize=17)
    plt.show()


n_to_erase = 157  # approx 20% [of 784 (28 x 28)] = 157

barplot_scores(8, 6, SCORING_METHODS, n_to_erase)

# def get_masked_image(idx, scores, task_1, task_2, n_to_erase):
#     difference = scores[str(task_1)][idx].ravel() - \
#         scores[str(task_2)][idx].ravel()
#     # highlight the top n
#     """Note: UP TO n_to_erase, as long as diff > 0"""
#     top_nth_threshold = max(
#         sorted(difference, key=lambda x: -x)[n_to_erase], 0.0)
#     thresholded_points = 1.0 * (difference <= top_nth_threshold)
#     masked_inp = thresholded_points.reshape(
#         28, 28, 1) * rX_test[idx]  # orig: X_test
#     return masked_inp


# # Function to plot the result of masking on a single example, for converting to anoother digit
# def plot_mask_figures(idx, task_1, task_2, method_names, n_to_erase):
#     print("example index: " + str(idx))
#     print(
#         "Columns:",
#         "task " + str(task_1) + " scores;",
#         "task " + str(task_2) + " scores;",
#         str(task_1) + "->" + str(task_2) + " masking;",
#     )

#     print("Order of the methods is: " + ", ".join(str(x)
#                                                   for x in method_names))
#     for method_name in method_names:
#         scores = method_scores[method_name]
#         mean_scores_over_all_tasks = np.mean(
#             np.array([scores[str(i)][idx] for i in range(10)]), axis=0
#         )
#         f, axarr = plt.subplots(1, 4, sharey=False, figsize=(15, 10))
#         viz_scores(rX_test[idx], axarr[0])  # orig: X_test
#         viz_scores(scores[str(task_1)][idx] -
#                    mean_scores_over_all_tasks, axarr[1])
#         viz_scores(scores[str(task_2)][idx] -
#                    mean_scores_over_all_tasks, axarr[2])
#         viz_scores(get_masked_image(
#             idx, scores, task_1, task_2, n_to_erase), axarr[3])
#     plt.show()

# # method_names = ["revealcancel", "integrated_grads_10", "deep_shap", "kernel_shap"]
# method_names = ["revealcancel",
#                 "integrated_grads_10", "deep_shap", "kernel_shap"]

# # Not so good 8->3; 8->6 example
# # plot_two_way_figures(61,8,3,6,method_names,n_to_erase)

# # Good 8->3; 8->6 example
# # plot_two_way_figures(5846,8,3,6,method_names,n_to_erase)

# # Bad 3->8; 3->9 example; doesn't work very well when switching from the
# # minority class to the majority class
# # (in terms of proximity to 0s and 1s)
# shift_map = {
#     0: 8,
#     1: 7,
#     2: 5,
#     3: 1,
#     4: 1,
#     5: 6,
#     6: 8,
#     7: 1,
#     8: 3,
#     9: 4,
# }

# plot_mask_figures(5, ry_test[5], shift_map[ry_test[5]],
#                   method_names, n_to_erase)   # orig index 5846, not 5

# def rand_barplot_scores(rX_test, rX_test_ds, ry_test, scoring_methods, n_to_erase):

#     # print("converting: "+str(original_class)+"->"+str(target_class))
#     method_names = [x[0] for x in scoring_methods]
#     short_names = [x[1] for x in scoring_methods]
#     # original_class_mask = ry_test == original_class
#     scores_to_plot = []
#     fig, ax = plt.subplots(figsize=(2*len(method_names), 5))
#     """
#     logodds obtained here by calling compute_delta_log_odds.
#     - what is predict_func=pre_softmax_fuc ? -> linear layer prior to softmax
#     """
#     for method_name in method_names:
#         logodds_diff, flipped, new_predictions, old_predictions = compute_delta_log_odds(
#             X=rX_test, y=ry_test,
#             method_name=method_name,
#             predict_func=pre_softmax_func,
#             imp_scores=method_scores[method_name],
#             # original_class=original_class,
#             # target_class=target_class,
#             num_perturbations=n_to_erase)
#         # figure out indices with big shifts
#         # retained_indices = np.compress(condition=original_class_mask, a=np.arange(len(y_test)))

#         # sorted_shifts
#         # sorted_shifts = sorted(enumerate(
#         # zip(logodds_diff, new_predictions, old_predictions)), key=lambda x: -x[1][0])
#         # print("top indices for "+str(method_name)+": "+" ".join([str(retained_indices[x[0]]) for x in sorted_shifts[:10]]))
#         scores_to_plot.append(logodds_diff)
#     ax.boxplot(scores_to_plot, widths=[
#                0.5 for x in method_names], showmeans=True, meanline=True)
#     # ax.set_ylim(-1000,17000)
#     ax.set_ylabel("Change in log-odds")
#     ax.set_xticklabels(short_names)
#     plt.title(str(original_class)+" --> "+str(target_class), fontsize=24)
#     plt.tick_params(labelsize=17)
#     plt.show()
