from tensorflow.keras.losses import *
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import *
import gc
from scipy.stats import pearsonr
import numpy as np


class GetSCAMetric(Callback):
    def __init__(self, dataset, settings, args):
        super().__init__()
        # parameters:
        self.x_train = dataset.x_attack
        self.y_train = np.array(args["y_train"])
        self.x_val = dataset.x_validation
        self.y_val = np.array(args["y_val"])
        self.epochs = settings["epochs"]
        self.correct_key = settings['good_key']
        self.loss = args["loss"]

        """ 
        - loss value for each epoch and for each key candidate 
        - the loss is computed for the training and validation sets
        """
        self.loss_candidates_epochs_train = np.zeros((self.epochs, 256))
        self.loss_candidates_epochs_val = np.zeros((self.epochs, 256))

        """ 
        Correlation between true and predicted labels
        - correlation value for each epoch and for each key candidate 
        - the correlation is computed for the training and validation sets
        """
        self.corr_epochs_train = np.zeros((self.epochs, 256))
        self.corr_epochs_val = np.zeros((self.epochs, 256))

        """ 
        Key rank evolution for the correct key (we assume a known key analysis here to be able to compute key rank)
        Key rank is computed for training and validation sets, considering both loss and correlation metrics.
        """
        self.key_rank_evolution_loss_train = np.zeros(self.epochs)
        self.key_rank_evolution_corr_train = np.zeros(self.epochs)
        self.key_rank_evolution_loss_val = np.zeros(self.epochs)
        self.key_rank_evolution_corr_val = np.zeros(self.epochs)

        """
        Objective function to measure the quality of the model. The objective function value is computed for each training epoch. 
        """
        self.objective_function_from_loss_train = np.zeros(self.epochs)
        self.objective_function_from_corr_train = np.zeros(self.epochs)
        self.objective_function_from_loss_val = np.zeros(self.epochs)
        self.objective_function_from_corr_val = np.zeros(self.epochs)

        self.objective_function_from_corr_train = np.zeros(self.epochs)
        self.objective_function_from_corr_val = np.zeros(self.epochs)

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)

    def compute_loss(self, y_pred, y_true):

        loss_candidates = np.zeros(256)
        if self.loss == "mse":
            for k in range(256):
                loss_candidates[k] = mean_squared_error(y_true[:, k], y_pred[:, k])

        if self.loss == "mae":
            for k in range(256):
                loss_candidates[k] = mean_absolute_error(y_true[:, k], y_pred[:, k])

        if self.loss == "huber":
            for k in range(256):
                delta = 1
                abs_error = np.mean(np.abs(y_true[:, k] - y_pred[:, k]))
                if abs_error <= delta:
                    loss = 0.5 * mean_squared_error(y_true[:, k], y_pred[:, k])
                else:
                    loss = np.mean(delta * (np.abs(y_true[:, k] - y_pred[:, k]) - 0.5 * delta))
                loss_candidates[k] = loss

        if self.loss == "corr":
            for k in range(256):
                loss_candidates[k] = 1 - abs(pearsonr(y_true[:, k], y_pred[:, k])[0])

        if self.loss == "key_rank":
            y_pred = 1 - (np.abs(y_true - y_pred) / 255)
            y_pred_k = [self.softmax(p) for p in y_pred]
            y_pred_k = np.log(np.array(y_pred_k) + 1e-36)

            for k in range(256):
                loss_candidates[k] = - np.mean(np.array(y_pred_k)[:, k])
            del y_pred_k

        if self.loss == "z_score_mse":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                loss_candidates[k] = mean_squared_error(y_true_score, y_pred_score)

        if self.loss == "z_score_mae":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                loss_candidates[k] = mean_absolute_error(y_true_score, y_pred_score)

        if self.loss == "z_score_huber":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                delta = 1
                abs_error = np.mean(np.abs(y_true_score - y_pred_score))
                if abs_error <= delta:
                    loss = 0.5 * mean_squared_error(y_true_score, y_pred_score)
                else:
                    loss = np.mean(delta * (np.abs(y_true_score - y_pred_score) - 0.5 * delta))
                loss_candidates[k] = loss

        if self.loss == "z_score_corr":
            for k in range(256):
                y_true_score = (y_true[:, k] - np.mean(y_true[:, k])) / np.maximum(float(np.std(y_true[:, k])), 1e-10)
                y_pred_score = (y_pred[:, k] - np.mean(y_pred[:, k])) / np.maximum(float(np.std(y_pred[:, k])), 1e-10)
                loss_candidates[k] = 1 - abs(pearsonr(y_true_score, y_pred_score)[0])

        return loss_candidates

    def get_key_rank(self, epoch, loss_candidates_epochs, validation=False):
        k_sum_sorted = np.argsort(loss_candidates_epochs[epoch, :])
        found_key = k_sum_sorted[0]
        if validation:
            self.key_rank_evolution_loss_val[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank (val): {self.key_rank_evolution_loss_val[epoch]} (found_key: {found_key})")
        else:
            self.key_rank_evolution_loss_train[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank: {self.key_rank_evolution_loss_train[epoch]} (found_key: {found_key})")
        return found_key

    def get_key_rank_corr(self, y_pred, y_true, epoch, validation=False):
        corr = np.zeros(256)
        for key in range(256):
            corr[key] = abs(pearsonr(y_pred[:, key], y_true[:, key])[0])
        k_sum_sorted = np.argsort(corr)[::-1]
        found_key = k_sum_sorted[0]

        if validation:
            self.key_rank_evolution_corr_val[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank (val): {self.key_rank_evolution_corr_val[epoch]} (found_key: {found_key})")
        else:
            self.key_rank_evolution_corr_train[epoch] = list(k_sum_sorted).index(self.correct_key) + 1
            print(f"Key rank: {self.key_rank_evolution_corr_train[epoch]} (found_key: {found_key})")

        return found_key, corr

    def get_objective_function(self, epoch, found_key, loss_epochs):

        """
        Compute objective function from loss values between true and predicted labels
        This function subtracts loss value for the found key candidate from the average of wrong keys
        """

        loss_most_likely_key = loss_epochs[epoch, found_key]
        mean_loss_other_keys = 0
        for key in range(256):
            if key != found_key:
                mean_loss_other_keys += loss_epochs[epoch, key]
        mean_loss_other_keys /= 255
        return abs(loss_most_likely_key - mean_loss_other_keys)

    def on_epoch_end(self, epoch, logs=None):

        """ Predict training and validation sets """
        y_pred_train = self.model.predict(self.x_train)
        y_pred_val = self.model.predict(self.x_val)

        """ Compute loss for each key candidate """
        self.loss_candidates_epochs_train[epoch] = self.compute_loss(y_pred_train, self.y_train)
        self.loss_candidates_epochs_val[epoch] = self.compute_loss(y_pred_val, self.y_val)

        """ 
        Compute key rank for the correct key candidate 
        Key rank is computed from loss and correlation between true and predicted labels 
        """
        found_key_loss_train = self.get_key_rank(epoch, self.loss_candidates_epochs_train)
        found_key_corr_train, corr = self.get_key_rank_corr(y_pred_train, self.y_train, epoch)
        found_key_val = self.get_key_rank(epoch, self.loss_candidates_epochs_val, validation=True)
        found_key_corr_val, corr_val = self.get_key_rank_corr(y_pred_val, self.y_val, epoch, validation=True)

        self.objective_function_from_loss_train[epoch] = self.get_objective_function(epoch, found_key_loss_train,
                                                                                     self.loss_candidates_epochs_train)
        self.objective_function_from_corr_train[epoch] = self.get_objective_function(epoch, found_key_corr_train,
                                                                                     self.loss_candidates_epochs_train)
        self.objective_function_from_loss_val[epoch] = self.get_objective_function(epoch, found_key_val,
                                                                                   self.loss_candidates_epochs_val)
        self.objective_function_from_corr_val[epoch] = self.get_objective_function(epoch, found_key_corr_val,
                                                                                   self.loss_candidates_epochs_val)

        """ Take the maximum correlation values as the objective function"""
        self.objective_function_from_corr_train[epoch] = np.max(corr)
        self.objective_function_from_corr_val[epoch] = np.max(corr_val)
        self.corr_epochs_train[epoch] = corr
        self.corr_epochs_val[epoch] = corr_val

        del y_pred_train
        gc.collect()
