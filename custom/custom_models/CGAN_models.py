import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import random
import numpy as np
import argparse
from datetime import datetime
import os
from os.path import exists
import matplotlib.pyplot as plt
from scripts.datasets.load_ascadr import *
from scripts.datasets.load_ascadf import *
from scripts.datasets.load_dpav42 import *
from scripts.datasets.load_eshard import *
from scripts.datasets.load_chesctf import *
from scripts.datasets.paths import *
from custom.custom_models.profiling_and_attack import *


def get_arguments(dataset_root_path, results_root_path, features_root_path):
    parser = argparse.ArgumentParser(add_help=False)

    """ root path for datasets """
    parser.add_argument("-dataset_root_path", "--dataset_root_path", default=dataset_root_path)

    """ root path for results """
    parser.add_argument("-results_root_path", "--results_root_path", default=results_root_path)

    """ root path for reference features """
    parser.add_argument("-features_root_path", "--features_root_path", default=features_root_path)

    """ dataset_reference: name of reference dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset_reference", "--dataset_reference", default="ascad-variable")

    """ dataset_reference_dim: number of features (samples) in reference dataset """
    parser.add_argument("-dataset_reference_dim", "--dataset_reference_dim", default=1400)

    """ dataset_target: name of target dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset_target", "--dataset_target", default="ASCAD")

    """ dataset_target_dim: number of features (samples) in target dataset """
    parser.add_argument("-dataset_target_dim", "--dataset_target_dim", default=700)

    """ features: number of features extracted from reference dataset and output dimension of generator """
    parser.add_argument("-features", "--features", default=100)

    """ n_profiling_reference: number of profiling traces from the reference dataset (always profiling set from .h5 datasets) """
    parser.add_argument("-n_profiling_reference", "--n_profiling_reference", default=200000)

    """ n_attack_reference: number of profiling traces from the reference dataset (always profiling set from .h5 datasets) """
    parser.add_argument("-n_attack_reference", "--n_attack_reference", default=5000)

    """ n_profiling_target: number of profiling traces from the target dataset """
    parser.add_argument("-n_profiling_target", "--n_profiling_target", default=50000)

    """ n_validation_target: number of validation traces from the target dataset """
    parser.add_argument("-n_validation_target", "--n_validation_target", default=0)

    """ n_attack_target: number of attack traces from the target dataset """
    parser.add_argument("-n_attack_target", "--n_attack_target", default=10000)

    """ n_attack_ge: number of attack traces for guessing entropy calculation """
    parser.add_argument("-n_attack_ge", "--n_attack_ge", default=2000)

    """ target_byte_reference: key byte index in the reference dataset """
    parser.add_argument("-target_byte_reference", "--target_byte_reference", default=2)

    """ target_byte_target: key byte index in the target dataset """
    parser.add_argument("-target_byte_target", "--target_byte_target", default=2)

    """ leakage_model: leakage model type (ID or HW) """
    parser.add_argument("-leakage_model", "--leakage_model", default="ID")

    """ epochs: number of training epochs for CGAN """
    parser.add_argument("-epochs", "--epochs", default=200)

    """ batch_size: batch size for training the CGAN """
    parser.add_argument("-batch_size", "--batch_size", default=400)

    """ std_gaussian_noise_reference: standard deviation for Gaussian noise artificially added to reference dataset (default mean is 0) """
    parser.add_argument("-std_gaussian_noise_reference", "--std_gaussian_noise_reference", default=0.0)

    """ std_gaussian_noise_target: standard deviation for Gaussian noise artificially added to target dataset (default mean is 0) """
    parser.add_argument("-std_gaussian_noise_target", "--std_gaussian_noise_target", default=0.0)

    """ random_search_generator: when set, generator hyperparameters are generated randomly (default is 0) """
    parser.add_argument("-random_search_generator", "--random_search_generator", default=0)

    """ random_search_discriminator: when set, discriminator hyperparameters are generated randomly (default is 0) """
    parser.add_argument("-random_search_discriminator", "--random_search_discriminator", default=0)

    return parser.parse_args()


def fixed_discriminator(features_dim: int, n_classes: int = 256, kern_init='random_normal'):
    # label input
    in_label = Input(shape=1)
    y = Embedding(n_classes, n_classes)(in_label)
    y = Dense(200, kernel_initializer=kern_init)(y)
    y = Flatten()(y)

    in_features = Input(shape=(features_dim,))

    merge = Concatenate()([y, in_features])

    x = Dense(100, kernel_initializer=kern_init)(merge)
    x = LeakyReLU()(x)
    x = Dropout(0.60)(x)

    # output
    out_layer = Dense(1, activation='sigmoid')(x)

    model = Model([in_label, in_features], out_layer)
    model.summary()
    return model


def fixed_generator(input_dim: int, output_dim: int, n_classes=256):
    # input_random_data = Input(shape=(self.traces_target_dim,))
    # rnd = Dense(400, activation='elu')(input_random_data)

    in_traces = Input(shape=(input_dim,))
    # x = Lambda(self.add_gaussian_noise)(in_traces)
    x = Dense(400, activation='linear')(in_traces)
    x = Dense(200, activation='linear')(x)
    x = Dense(100, activation='linear')(x)
    out_layer = Dense(output_dim, activation='linear')(x)

    model = Model([in_traces], out_layer)
    model.summary()
    return model


class CreateModels:

    def __init__(self, args, dir_results):
        # helper functions for tensorflow compiling
        self.real_accuracy_metric = BinaryAccuracy()
        self.fake_accuracy_metric = BinaryAccuracy()
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.cross_entropy_disc = BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0025, beta_1=0.5)

        classes = 9 if args["leakage_model"] == "HW" else 256

        if args["random_search_generator"] or args["random_search_discriminator"]:

            self.hp_d = {
                "neurons_embed": random.choice([100, 200, 500]),
                "neurons_dropout": random.choice([100, 200, 500]),
                "neurons_bilinear": random.choice([100, 200, 500]),
                "layers_embed": random.choice([1, 2, 3]),
                "layers_dropout": random.choice([1, 2, 3]),
                "layers_bilinear": random.choice([1, 2, 3]),
                "dropout": random.choice([0.5, 0.6, 0.7, 0.8]),
            }
            if args["generator_type"] == "cnn":
                self.hp_g = {
                    "neurons_1": random.choice([100, 200, 300, 400, 500]),
                    "layers": random.choice([1, 2, 3, 4]),
                    "conv_layers": random.choice([1, 2]),
                    "filters_1": random.choice([8, 16, 32]),
                    "kernel_size_1": random.choice([10, 20, 40]),
                    "strides_1": random.choice([5, 10]),
                    "activation": random.choice(["elu", "selu", "relu", "leakyrelu", "linear", "tanh"]),
                }
                for l_i in range(1, self.hp_g["conv_layers"]):
                    self.hp_g[f"filters_{l_i + 1}"] = self.hp_g[f"filters_{l_i}"] * 2
                    self.hp_g[f"kernel_size_{l_i + 1}"] = random.choice([10, 20, 40]),
                    self.hp_g[f"strides_{l_i + 1}"] = random.choice([5, 10]),
                for l_i in range(1, self.hp_g["layers"]):
                    options_neurons = list(range(100, self.hp_g[f"neurons_{l_i}"] + 100, 100))
                    self.hp_g[f"neurons_{l_i + 1}"] = random.choice(options_neurons)
            else:  # mlp
                self.hp_g = {
                    "neurons_1": random.choice([100, 200, 300, 400, 500]),
                    "layers": random.choice([1, 2, 3, 4]),
                    "activation": random.choice(["elu", "selu", "relu", "leakyrelu", "linear", "tanh"]),
                }
                for l_i in range(1, self.hp_g["layers"]):
                    options_neurons = list(range(100, self.hp_g[f"neurons_{l_i}"] + 100, 100))
                    self.hp_g[f"neurons_{l_i + 1}"] = random.choice(options_neurons)

            # create the discriminator
            self.discriminator = self.define_discriminator_random(args["features"], n_classes=classes)
            # create the generator
            self.generator = self.define_generator_random(args["dataset_target_dim"], args["features"])

            np.savez(f"{dir_results}/hp.npz", hp_d=self.hp_d, hp_g=self.hp_g)
        else:

            self.best_models_random_search(args["dataset_reference"], args["dataset_target"])
            # create the discriminator
            self.discriminator = self.define_discriminator_random(args["features"], n_classes=classes)
            # create the generator
            self.generator = self.define_generator_random(args["dataset_target_dim"], args["features"])

    def best_models_random_search(self, reference, target):
        if reference == "ascad-variable":
            if target == "ASCAD":
                # reference ascad-variable (25000) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 3,
                    'dropout': 0.7,
                    'neurons_bilinear': 200, 'layers_bilinear': 1,
                }
                self.hp_g = {
                    'neurons_1': 300, 'layers': 1, 'activation': 'linear'
                }
            if target == "dpa_v42":
                # reference ascad-variable (25000) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 200, 'layers': 4, 'activation': 'linear', 'neurons_2': 200, 'neurons_3': 200,
                    'neurons_4': 100
                }
            if target == "ches_ctf":
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.5
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100,
                    'neurons_4': 100
                }
            if target == "eshard":
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 3, 'activation': 'leakyrelu', 'neurons_2': 500, 'neurons_3': 100
                }
        if reference == "ASCAD":
            if target == "ascad-variable":
                # reference ASCAD (10000) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1,
                    'dropout': 0.6,
                }
                self.hp_g = {
                    'neurons_1': 200, 'layers': 3, 'activation': 'leakyrelu', 'neurons_2': 200, 'neurons_3': 100
                }
            if target == "dpa_v42":
                # reference ASCAD (10000) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 2, 'layers_dropout': 2, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 300, 'layers': 2, 'activation': 'linear', 'neurons_2': 100
                }
            if target == "ches_ctf":
                # reference ASCAD (10000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.5
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100,
                    'neurons_4': 100
                }
            if target == "eshard":
                # reference ASCAD (10000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 2, 'activation': 'selu', 'neurons_2': 400
                }
        if reference == "dpa_v42":
            if target == "ascad-variable":
                # reference dpa_v42 (15000) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 100, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 1, 'activation': 'elu'
                }
            if target == "ASCAD":
                # reference dpa_v42 (15000) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100,
                    'neurons_4': 100
                }
            if target == "ches_ctf":
                # reference dpa_v42 (15000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 1, 'activation': 'linear'
                }
            if target == "eshard":
                # reference dpa_v42 (15000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 400, 'layers': 2, 'activation': 'selu', 'neurons_2': 300
                }
        if reference == "eshard":
            if target == "ascad-variable":
                # reference eshard (1400) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100,
                    'neurons_4': 100
                }
            if target == "ASCAD":
                # reference eshard (1400) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 400, 'layers': 2, 'activation': 'linear', 'neurons_2': 300
                }
            if target == "dpa_v42":
                # reference eshard (1400) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 2, 'layers_dropout': 2, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 1, 'activation': 'linear'
                }

    def discriminator_loss(self, real, fake):
        real_loss = self.cross_entropy_disc(tf.ones_like(real), real)
        fake_loss = self.cross_entropy_disc(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def generator_loss(self, fake):
        return self.cross_entropy(tf.ones_like(fake), fake)

    def define_discriminator_random(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        for l_i in range(self.hp_d["layers_embed"]):
            y = Dense(self.hp_d["neurons_embed"], kernel_initializer=kern_init)(y)
            y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        merge = Concatenate()([y, in_features])

        x = None
        for l_i in range(self.hp_d["layers_dropout"]):
            x = Dense(self.hp_d["neurons_dropout"], kernel_initializer=kern_init)(merge if l_i == 0 else x)
            x = LeakyReLU()(x)
            x = Dropout(self.hp_d["dropout"])(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model

        # define the standalone generator model

    # define a random generator model
    def define_generator_random(self, input_dim: int, output_dim: int):

        in_traces = Input(shape=(input_dim,))
        x = None
        for l_i in range(self.hp_g["layers"]):
            x = Dense(self.hp_g[f"neurons_{l_i + 1}"],
                      activation=self.hp_g["activation"] if self.hp_g["activation"] != "leakyrelu" else None)(
                in_traces if l_i == 0 else x)
            if self.hp_g["activation"] == "leakyrelu":
                x = LeakyReLU()(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model




def attack(dataset, generator, features_dim: int, attack_model=None, synthetic_traces=True, original_traces=False):
    """ Generate a batch of synthetic measurements with the trained generator """
    if original_traces:
        features_target_attack = np.array(dataset.dataset_target.x_attack)
        features_target_profiling = np.array(dataset.dataset_target.x_profiling)
    else:
        if synthetic_traces:
            features_target_attack = np.array(generator.predict([dataset.dataset_target.x_attack]))
            features_target_profiling = np.array(generator.predict([dataset.dataset_target.x_profiling]))
        else:
            features_target_attack = np.array(dataset.features_target_attack)
            features_target_profiling = np.array(dataset.features_target_profiling)

    """ Define a neural network (MLP) to be trained with synthetic traces """

    if attack_model is None:
        model = mlp(dataset.dataset_target.classes, features_dim)
    else:
        model = attack_model
    model.fit(
        x=features_target_profiling,
        y=to_categorical(dataset.dataset_target.profiling_labels, num_classes=dataset.dataset_target.classes),
        batch_size=400,
        verbose=2,
        epochs=50,
        shuffle=True,
        validation_data=(
            features_target_attack, to_categorical(dataset.dataset_target.attack_labels, num_classes=dataset.dataset_target.classes)),
        callbacks=[])

    """ Predict the trained MLP with target/attack measurements """
    predictions = model.predict(features_target_attack)
    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, dataset.dataset_target.labels_key_hypothesis_attack,
                                         dataset.dataset_target.correct_key_attack, 2000)
    pi = information(predictions, dataset.dataset_target.attack_labels, dataset.dataset_target.classes)
    return ge, nt, pi, ge_vector

class PrepareDatasets:

    def __init__(self, args):
        self.features_dim = args["features"]
        self.target_byte_reference = args["target_byte_reference"]
        self.target_byte_target = args["target_byte_target"]
        self.path = args["dataset_root_path"]

        self.traces_reference_dim = args["dataset_reference_dim"]
        self.traces_target_dim = args["dataset_target_dim"]

        self.dataset_reference = self.load_dataset(args, args["dataset_reference"], self.target_byte_reference,
                                                   self.traces_reference_dim,
                                                   args["n_profiling_reference"], 0, args["n_attack_reference"],
                                                   reference=True)
        self.dataset_target = self.load_dataset(args, args["dataset_target"], self.target_byte_target,
                                                self.traces_target_dim,
                                                args["n_profiling_target"], args["n_validation_target"],
                                                args["n_attack_target"])

        self.add_gaussian_noise(args)

        self.dataset_reference.x_profiling, self.dataset_reference.x_attack = self.scale_dataset(
            self.dataset_reference.x_profiling,
            self.dataset_reference.x_attack,
            StandardScaler())
        self.dataset_target.x_profiling, self.dataset_target.x_attack = self.scale_dataset(
            self.dataset_target.x_profiling,
            self.dataset_target.x_attack,
            StandardScaler())

        self.features_reference_profiling, self.features_reference_attack = self.dataset_reference.x_profiling, self.dataset_reference.x_attack

        """ the following is used only for verification, not in the CGAN training """
        self.features_target_profiling, self.features_target_attack = get_features(self.dataset_target,
                                                                                   self.target_byte_target,
                                                                                   n_poi=self.features_dim)

    def load_dataset(self, args, identifier, target_byte, traces_dim, n_prof, n_val, n_attack, reference=False):

        implement_reference_feature_selection = False
        reference_features_shortcut = ""
        num_features = args["features"]
        dataset_file = get_dataset_filepath(args["dataset_root_path"], identifier, traces_dim, args["leakage_model"])
        if reference:
            """ If features were already computed for this dataset, target key byte, 
            and leakage model, there is no need to compute it again"""
            reference_features_shortcut = f'{args["features_root_path"]}/selected_{args["features"]}_features_snr_{args["dataset_reference"]}_{self.traces_reference_dim}_target_byte_{self.target_byte_reference}.h5'
            if exists(reference_features_shortcut):
                print("Reference features already created.")
                dataset_file = reference_features_shortcut
                traces_dim = num_features
            else:
                implement_reference_feature_selection = True

        dataset = None
        if identifier == "ascad-variable":
            dataset = ReadASCADr(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                 number_of_samples=traces_dim)
        if identifier == "ASCAD":
            dataset = ReadASCADf(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                 number_of_samples=traces_dim)
        if identifier == "eshard":
            dataset = ReadEshard(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                 number_of_samples=traces_dim)
        if identifier == "dpa_v42":
            dataset = ReadDPAV42(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                 number_of_samples=traces_dim)
        if identifier == "ches_ctf":
            dataset = ReadCHESCTF(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)

        if implement_reference_feature_selection:
            self.generate_features_h5(dataset, target_byte, reference_features_shortcut, num_features)
            dataset_file = reference_features_shortcut
            traces_dim = num_features

            if identifier == "ascad-variable":
                return ReadASCADr(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "ASCAD":
                return ReadASCADf(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "eshard":
                return ReadEshard(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "dpa_v42":
                return ReadDPAV42(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "ches_ctf":
                return ReadCHESCTF(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                   number_of_samples=traces_dim)
        else:
            return dataset

    def scale_dataset(self, prof_set, attack_set, scaler):
        prof_new = scaler.fit_transform(prof_set)
        if attack_set is not None:
            attack_new = scaler.transform(attack_set)
        else:
            attack_new = None
        return prof_new, attack_new

    def add_gaussian_noise(self, args):
        if args["std_gaussian_noise_reference"] > 0.0:
            print(f"adding gaussian noise of {args['std_gaussian_noise_reference']}")
            noise = np.random.normal(0, args["std_gaussian_noise_reference"],
                                     np.shape(self.dataset_reference.x_profiling))
            self.dataset_reference.x_profiling = np.add(self.dataset_reference.x_profiling, noise)
            noise = np.random.normal(0, args["std_gaussian_noise_reference"], np.shape(self.dataset_reference.x_attack))
            self.dataset_reference.x_attack = np.add(self.dataset_reference.x_attack, noise)

        if args["std_gaussian_noise_target"] > 0.0:
            print(f"adding gaussian noise of {args['std_gaussian_noise_target']}")
            noise = np.random.normal(0, args["std_gaussian_noise_target"], np.shape(self.dataset_target.x_profiling))
            self.dataset_target.x_profiling = np.add(self.dataset_target.x_profiling, noise)
            noise = np.random.normal(0, args["std_gaussian_noise_target"], np.shape(self.dataset_target.x_attack))
            self.dataset_target.x_attack = np.add(self.dataset_target.x_attack, noise)

    def generate_features_h5(self, dataset, target_byte, save_file_path, num_features):

        profiling_traces_rpoi, attack_traces_rpoi = get_features(dataset, target_byte, num_features)
        out_file = h5py.File(save_file_path, 'w')

        profiling_index = [n for n in range(dataset.n_profiling)]
        attack_index = [n for n in range(dataset.n_attack)]

        profiling_traces_group = out_file.create_group("Profiling_traces")
        attack_traces_group = out_file.create_group("Attack_traces")

        profiling_traces_group.create_dataset(name="traces", data=profiling_traces_rpoi,
                                              dtype=profiling_traces_rpoi.dtype)
        attack_traces_group.create_dataset(name="traces", data=attack_traces_rpoi, dtype=attack_traces_rpoi.dtype)

        metadata_type_profiling = np.dtype(
            [("plaintext", dataset.profiling_plaintexts.dtype, (len(dataset.profiling_plaintexts[0]),)),
             ("key", dataset.profiling_keys.dtype, (len(dataset.profiling_keys[0]),)),
             ("masks", dataset.profiling_masks.dtype, (len(dataset.profiling_masks[0]),))
             ])
        metadata_type_attack = np.dtype(
            [("plaintext", dataset.attack_plaintexts.dtype, (len(dataset.attack_plaintexts[0]),)),
             ("key", dataset.attack_keys.dtype, (len(dataset.attack_keys[0]),)),
             ("masks", dataset.attack_masks.dtype, (len(dataset.attack_masks[0]),))
             ])

        profiling_metadata = np.array(
            [(dataset.profiling_plaintexts[n], dataset.profiling_keys[n], dataset.profiling_masks[n]) for n in
             profiling_index],
            dtype=metadata_type_profiling)
        profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

        attack_metadata = np.array(
            [(dataset.attack_plaintexts[n], dataset.attack_keys[n], dataset.attack_masks[n]) for n in attack_index],
            dtype=metadata_type_attack)
        attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

        out_file.flush()
        out_file.close()


class TrainCGAN:

    def __init__(self, args, models, dir_results):
        self.args = args
        self.datasets = PrepareDatasets(self.args)
        self.models = models
        self.dir_results = dir_results

        """ Metrics to assess quality of profiling attack: Max SNR, Guessing entropy, Ntraces_GE = 1, Perceived Information """
        self.snr_features_share_1 = []
        self.snr_features_share_2 = []
        self.max_snr_share_1 = []
        self.max_snr_share_2 = []

        self.ge_fake = []
        self.nt_fake = []
        self.pi_fake = []

        self.ge_real = []
        self.nt_real = []
        self.pi_real = []

        self.ge_real_original = []
        self.nt_real_original = []
        self.pi_real_original = []

        self.ge_real_ta = []
        self.nt_real_ta = []
        self.pi_real_ta = []

        """ Just for plot """
        self.x_axis_epochs = []

        """ Accuracy for real and synthetic data """
        self.real_acc = []
        self.fake_acc = []

        """ Generator and Discriminator Losses """
        self.g_loss = []
        self.d_loss = []

    def generate_reference_samples(self, batch_size):
        rnd = np.random.randint(0, self.datasets.dataset_reference.n_profiling - batch_size)
        features = self.datasets.features_reference_profiling[rnd:rnd + batch_size]
        labels = self.datasets.dataset_reference.profiling_labels[rnd:rnd + batch_size]
        return [features, labels]

    def generate_target_samples(self, batch_size):
        rnd = np.random.randint(0, self.datasets.dataset_target.n_profiling - batch_size)
        traces = self.datasets.dataset_target.x_profiling[rnd:rnd + batch_size]
        labels = self.datasets.dataset_target.profiling_labels[rnd:rnd + batch_size]
        return [traces, labels]

    @tf.function
    def train_step(self, traces_batch, label_traces, features, label_features):

        with tf.GradientTape() as disc_tape:
            fake_features = self.models.generator(traces_batch)
            real_output = self.models.discriminator([label_features, features])
            fake_output = self.models.discriminator([label_traces, fake_features])
            disc_loss = self.models.discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.models.discriminator.trainable_variables)
        self.models.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.models.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_features = self.models.generator(traces_batch)
            fake_output = self.models.discriminator([label_traces, fake_features])
            gen_loss = self.models.generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.models.generator.trainable_variables)
        self.models.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.models.generator.trainable_variables))

        self.models.fake_accuracy_metric.update_state(tf.zeros_like(fake_features), fake_output)
        self.models.real_accuracy_metric.update_state(tf.ones_like(features), real_output)

        return gen_loss, disc_loss

    def compute_snr_reference_features(self):
        batch_size_reference = 10000

        # prepare traces from target dataset
        rnd_reference = random.randint(0, len(self.datasets.dataset_reference.x_profiling) - batch_size_reference)
        features_reference = self.datasets.features_reference_profiling[
                             rnd_reference:rnd_reference + batch_size_reference]

        snr_reference_features_share_1 = snr_fast(features_reference,
                                                  self.datasets.dataset_reference.share1_profiling[
                                                  self.datasets.target_byte_reference,
                                                  rnd_reference:rnd_reference + batch_size_reference])
        snr_reference_features_share_2 = snr_fast(features_reference,
                                                  self.datasets.dataset_reference.share2_profiling[
                                                  self.datasets.target_byte_reference,
                                                  rnd_reference:rnd_reference + batch_size_reference])
        plt.plot(snr_reference_features_share_1)
        plt.plot(snr_reference_features_share_2)

        plt.xlim([1, self.datasets.features_dim])
        plt.savefig(f"{self.dir_results}/snr_reference_features.png")
        plt.close()

        np.savez(f"{self.dir_results}/snr_reference_features.npz",
                 snr_reference_features_share_1=snr_reference_features_share_1,
                 snr_reference_features_share_2=snr_reference_features_share_2)

    def compute_snr_target_features(self, epoch, synthetic_traces=True):
        batch_size_target = 10000

        # prepare traces from target dataset
        rnd_target = random.randint(0, len(self.datasets.dataset_target.x_profiling) - batch_size_target)

        if synthetic_traces:
            traces_target = self.datasets.dataset_target.x_profiling[rnd_target:rnd_target + batch_size_target]
            features_target = self.models.generator.predict([traces_target])
        else:
            features_target = self.datasets.features_target_profiling[rnd_target:rnd_target + batch_size_target]

        snr_target_features_share_1 = snr_fast(features_target,
                                               self.datasets.dataset_target.share1_profiling[
                                               self.datasets.target_byte_target,
                                               rnd_target:rnd_target + batch_size_target]).tolist()
        snr_target_features_share_2 = snr_fast(features_target,
                                               self.datasets.dataset_target.share2_profiling[
                                               self.datasets.target_byte_target,
                                               rnd_target:rnd_target + batch_size_target]).tolist()

        plt.plot(snr_target_features_share_1)
        plt.plot(snr_target_features_share_2)
        plt.xlim([1, self.datasets.features_dim])
        if synthetic_traces:
            plt.savefig(f"{self.dir_results}/snr_target_features_fake_{epoch}.png")
            plt.close()
            self.snr_features_share_1.append(snr_target_features_share_1)
            self.snr_features_share_2.append(snr_target_features_share_2)
            np.savez(f"{self.dir_results}/snr_target_features_fake.npz",
                     snr_target_features_share_1=self.snr_features_share_1,
                     snr_target_features_share_2=self.snr_features_share_2)
        else:
            plt.savefig(f"{self.dir_results}/snr_target_features_real_{epoch}.png")
            plt.close()

            np.savez(f"{self.dir_results}/snr_target_features_real.npz",
                     snr_target_features_share_1=snr_target_features_share_1,
                     snr_target_features_share_2=snr_target_features_share_2)

        if synthetic_traces:
            self.max_snr_share_1.append(np.max(snr_target_features_share_1))
            self.max_snr_share_2.append(np.max(snr_target_features_share_2))
            plt.plot(self.max_snr_share_1, label="Max SNR Share 1")
            plt.plot(self.max_snr_share_2, label="Max SNR Share 2")
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("SNR")
            plt.savefig(f"{self.dir_results}/max_snr_shares.png")
            plt.close()
            np.savez(f"{self.dir_results}/max_snr_shares.npz", max_snr_share_1=self.max_snr_share_1,
                     max_snr_share_2=self.max_snr_share_2)

    def attack_eval(self, epoch):

        ge_fake, nt_fake, pi_fake, ge_vector_fake = attack(self.datasets, self.models.generator,
                                                           self.datasets.features_dim)
        self.ge_fake.append(ge_fake)
        self.nt_fake.append(nt_fake)
        self.pi_fake.append(pi_fake)

        self.x_axis_epochs.append(epoch + 1)

        plt.plot(self.x_axis_epochs, self.ge_fake, label="CGAN-SCA")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Guessing Entropy")
        plt.savefig(f"{self.dir_results}/ge.png")
        plt.close()

        plt.plot(self.x_axis_epochs, self.nt_fake, label="CGAN-SCA")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Number of Traces for GE=1")
        plt.yscale('log')
        plt.savefig(f"{self.dir_results}/nt.png")
        plt.close()

        plt.plot(self.x_axis_epochs, self.pi_fake, label="CGAN-SCA")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Perceived Information")
        plt.savefig(f"{self.dir_results}/pi.png")
        plt.close()

        np.savez(f"{self.dir_results}/metrics.npz",
                 ge_fake=self.ge_fake,
                 nt_fake=self.nt_fake,
                 pi_fake=self.pi_fake
                 )

        plt.plot(ge_vector_fake, label="CGAN-SCA")
        plt.legend()
        plt.xscale('log')
        plt.xlabel("Attack Traces")
        plt.ylabel("Guessing Entropy")
        plt.savefig(f"{self.dir_results}/ge_epoch_{epoch}.png")
        plt.close()

        np.savez(f"{self.dir_results}/ge_vector_epoch_{epoch}.npz", ge_vector_fake=ge_vector_fake)

    def train(self):
        training_set_size = max(self.datasets.dataset_reference.n_profiling, self.datasets.dataset_target.n_profiling)

        # determine half the size of one batch, for updating the discriminator
        batch_size = self.args["batch_size"]
        n_batches = int(training_set_size / batch_size)

        epoch_snr_step = 10 if self.args["random_search_generator"] or self.args["random_search_discriminator"] else 1

        # manually enumerate epochs
        for e in range(self.args["epochs"]):
            for b in range(n_batches):
                [features_reference, labels_reference] = self.generate_reference_samples(batch_size)
                [traces_target, labels_target] = self.generate_target_samples(batch_size)

                # Custom training step for speed and versatility
                g_loss, d_loss = self.train_step(traces_target, labels_target, features_reference, labels_reference)

                if (b + 1) % 100 == 0:
                    self.real_acc.append(self.models.real_accuracy_metric.result())
                    self.fake_acc.append(self.models.fake_accuracy_metric.result())
                    self.g_loss.append(g_loss)
                    self.d_loss.append(d_loss)

                    plt.plot(self.real_acc, label="Real")
                    plt.plot(self.fake_acc, label="Fake")
                    plt.axhline(y=0.5, linestyle="dashed", color="black")
                    plt.legend()
                    plt.savefig(f"{self.dir_results}/acc.png")
                    plt.close()

                    plt.plot(self.g_loss, label="g_loss")
                    plt.plot(self.d_loss, label="d_Loss")
                    plt.legend()
                    plt.savefig(f"{self.dir_results}/loss.png")
                    plt.close()

                    print(
                        f"epoch: {e}, batch: {b}, d_loss: {d_loss}, g_loss: {g_loss}, real_acc: {self.models.real_accuracy_metric.result()}, fake_acc: {self.models.fake_accuracy_metric.result()}")
                    np.savez(f"{self.dir_results}/acc_and_loss.npz",
                             g_loss=self.g_loss, d_loss=self.d_loss,
                             real_acc=self.real_acc, fake_acc=self.fake_acc)

            # Split eval steps up as attacking takes significant time while snr computation is fast
            if e == 0:
                self.compute_snr_reference_features()
                self.compute_snr_target_features(e, synthetic_traces=False)
            if (e + 1) % epoch_snr_step == 0:
                self.compute_snr_target_features(e)
                # self.attack_eval_synthetic(e)
            if (e + 1) % 50 == 0:
                self.attack_eval(e)
                if not self.args["random_search_generator"] and not self.args["random_search_discriminator"]:
                    self.models.generator.save(
                        f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_{e}.h5")
        if not self.args["random_search_generator"] and not self.args["random_search_discriminator"]:
            self.models.generator.save(
                f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_{self.args['epochs'] - 1}.h5")


def create_directory_results(args, path):
    now = datetime.now()
    now_str = f"{now.strftime('%d_%m_%Y_%H_%M_%S')}_{np.random.randint(1000000, 10000000)}"
    dir_results = f"{path}/{args['dataset_reference']}_vs_{args['dataset_target']}_{now_str}"
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)
    return dir_results


class CGANSCA:

    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.main_path = self.args["results_root_path"]

    def train_cgan(self):
        dir_results = create_directory_results(self.args, self.main_path)
        models = CreateModels(self.args, dir_results)
        np.savez(f"{dir_results}/args.npz", args=self.args)
        train_cgan = TrainCGAN(self.args, models, dir_results)
        train_cgan.train()
