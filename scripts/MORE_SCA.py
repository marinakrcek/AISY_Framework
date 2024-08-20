import time
import aisy_sca
from aisy_sca.callbacks.CallbackControls import CallbackControls
from aisy_sca.metrics import SCAMetrics
import tensorflow.keras.backend as backend
from termcolor import colored
from aisy_sca.analysis.MultiModel import MultiModel
from aisy_sca.AnalysisDatabaseControls import *
from aisy_sca.utils import Utils
from aisy_sca.datasets import DatasetControls
from aisy_sca.AnalysisControls import *
from aisy_sca.analysis.HyperparameterSearchProcess import HyperparameterSearchProcess
from aisy_sca.LeakageModelControls import *
from custom.custom_models.MORE_models import mlp_np, cnn_np
import random
from app import *


class NonProfiling:
    """
    Class to run non-profiling SCA on the dataset and model
    model: keras model already compiled
    settings: analysis settings
    dataset: dataset structure containing traces, labels and metadata (plaintext, ciphertext, keys)
    search_index: for hyper-parameter search purposes
    train_best_model: for hyper-parameter search purposes
    """

    def __init__(self, settings, dataset, da_function):
        self.settings = settings
        self.dataset = dataset
        self.da_function = da_function
        self.builtin_callbacks = None
        self.custom_callbacks = None
        self.labels_key_guesses_attack_set = None
        self.labels_key_guesses_validation_set = None

    def train_model(self, model, n_traces=None):

        """ reshape traces if needed """
        input_layer_shape = model.get_layer(index=0).input_shape

        """ Check if model is created with Sequential or Model class from Keras """
        if len(input_layer_shape) == 1:
            """ Model was created with Model class """
            input_layer_shape = input_layer_shape[0]

        """ Check if neural network is a cnn or mlp """
        if len(input_layer_shape) == 3:
            self.dataset.reshape_for_cnn()
        else:
            self.dataset.reshape_for_mlp()

        """ Create callbacks """
        callback_controls = CallbackControls(self.dataset, self.settings)
        callback_controls.create_callbacks()
        callbacks = callback_controls.get_callbacks()
        self.builtin_callbacks = callback_controls.get_builtin_callbacks()
        self.custom_callbacks = callback_controls.get_custom_callbacks()

        if self.settings["split_test_set"]:
            validation_set = (self.dataset.x_validation, self.labels_key_guesses_validation_set.T)  # it is from the attack set
        else:
            validation_set = (self.dataset.x_attack, self.labels_key_guesses_attack_set.T)

        if self.settings["use_data_augmentation"]:

            """  If data augmentation is used, then x_attack needs to be reshaped back to 2D. """

            x_training = self.dataset.x_attack.reshape((self.dataset.x_attack.shape[0], self.dataset.x_attack.shape[1]))

            da_method = self.da_function(x_training, self.labels_key_guesses_attack_set.T, self.settings["batch_size"],
                                         input_layer_shape)
            history = model.fit_generator(
                generator=da_method,
                steps_per_epoch=self.settings["data_augmentation"][1],
                epochs=self.settings["epochs"],
                verbose=2,
                validation_data=validation_set,
                validation_steps=1,
                callbacks=callbacks)

        else:

            """ Train the model """

            history = model.fit(
                x=self.dataset.x_attack[:n_traces] if self.settings[
                    "use_profiling_analyzer"] else self.dataset.x_attack,
                y=self.labels_key_guesses_attack_set.T[:n_traces] if self.settings[
                    "use_profiling_analyzer"] else self.labels_key_guesses_attack_set.T,
                batch_size=self.settings["batch_size"],
                verbose=2,
                epochs=self.settings["epochs"],
                shuffle=True,
                validation_data=validation_set,
                callbacks=callbacks)

        return history

    def get_builtin_callbacks(self):
        return self.builtin_callbacks

    def get_custom_callbacks(self):
        return self.custom_callbacks

    def compute_metrics(self, model, analysis_db_controls, hp_id, n_traces=None, random_states=None):

        """ Compute SCA metrics: guessing entropy and success rate for attack and validation sets """

        """ 
        Compute Guessing Entropy, Success Rate and Number of Traces for Success (GE < 2) for attack and validation sets 
        Return scores/probabilities for each trace (p(l=leakage|trace=x))
        Retrieve random states for reproducibility
        Save results to database
        """

        label_name = f"Attack Set" if n_traces is None else f"Attack Set {n_traces} traces"
        random_states_ge_sr = random_states[f"{model['index']}"][f"{label_name}"] if random_states is not None else None
        ge, sr, nt, sk, r = SCAMetrics(model["model"], self.dataset.x_attack, self.settings,
                                       self.labels_key_guesses_attack_set).run(
            random_states=random_states_ge_sr)

        analysis_db_controls.save_sca_metrics_to_database(ge, sr, label_name, hp_id, r, model['index'])

        if self.dataset.x_validation is not None:
            label_name = f"Validation Set" if n_traces is None else f"Validation Set {n_traces} traces"
            random_states_ge_sr = random_states[f"{model['index']}"][
                f"{label_name}"] if random_states is not None else None
            ge, sr, nt, sk, r = SCAMetrics(model["model"], self.dataset.x_validation, self.settings,
                                           self.labels_key_guesses_validation_set).run(
                random_states=random_states_ge_sr)
            analysis_db_controls.save_sca_metrics_to_database(ge, sr, label_name, hp_id, r, model['index'])

        """
        For Early Stopping Metrics:
        Compute Guessing Entropy, Success Rate and Number of Traces for Success (GE < 2) for attack and validation sets 
        Return scores/probabilities for each trace (p(l=leakage|trace=x))
        Retrieve random states for reproducibility        
        """

        """ TODO: search for saved h5 models in folder and then run metrics"""

        if self.settings["use_early_stopping"]:

            early_stopping_metrics = self.get_early_stopping_metrics(
                self.builtin_callbacks["early_stopping_callback"].get_metric_results())

            for es_metric in early_stopping_metrics:
                self.load_model(model["model"], es_metric)

                print(colored("Computing Guessing Entropy and Success Rate for Attack Set", "green"))
                label_name = f"ES Attack Set {es_metric}" if n_traces is None else f"ES Attack Set {es_metric} {n_traces} traces"
                random_states_ge_sr = random_states[f"{model['index']}"][
                    f"{label_name}"] if random_states is not None else None
                ge, sr, nt, sk, r = SCAMetrics(model["model"], self.dataset.x_attack, self.settings,
                                               self.labels_key_guesses_attack_set).run(
                    random_states=random_states_ge_sr)
                analysis_db_controls.save_sca_metrics_to_database(ge, sr, label_name, hp_id, r, model['index'])

                print(colored("Computing Guessing Entropy and Success Rate for Validation Set", "green"))
                label_name = f"ES Validation Set {es_metric}" if n_traces is None else f"ES Validation Set {es_metric} {n_traces} traces"
                random_states_ge_sr = random_states[f"{model['index']}"][
                    f"{label_name}"] if random_states is not None else None
                ge, sr, nt, sk, r = SCAMetrics(model["model"], self.dataset.x_validation, self.settings,
                                               self.labels_key_guesses_validation_set).run(
                    random_states=random_states_ge_sr)
                analysis_db_controls.save_sca_metrics_to_database(ge, sr, label_name, hp_id, r, model['index'])

                self.delete_model(es_metric)

    def load_model(self, model, setting, idx=None):

        """ Load weights from saved early stopping model from 'resource/models' directory"""

        if idx is None:
            model_name = f"{self.settings['resources_root_folder']}models/best_model_{setting}_{self.settings['timestamp']}.h5"
        else:
            model_name = f"{self.settings['resources_root_folder']}models/best_model_{setting}_{self.settings['timestamp']}_{idx}.h5"
        print(model_name)
        model.load_weights(model_name)

    def delete_model(self, setting, idx=None):

        """ Load model from 'resource/models' directory """

        if idx is None:
            model_name = f"{self.settings['resources_root_folder']}models/best_model_{setting}_{self.settings['timestamp']}.h5"
        else:
            model_name = f"{self.settings['resources_root_folder']}models/best_model_{setting}_{self.settings['timestamp']}_{idx}.h5"
        os.remove(model_name)

    def get_early_stopping_metrics(self, early_stopping_metric_results):
        early_stopping_metric_names = []
        for early_stopping_metric in self.settings["early_stopping"]["metrics"]:
            if isinstance(early_stopping_metric_results[early_stopping_metric][0], list):
                for idx in range(len(early_stopping_metric_results[early_stopping_metric][0])):
                    early_stopping_metric_names.append(f"{early_stopping_metric}_{idx}")
            else:
                early_stopping_metric_names.append(early_stopping_metric)
        return early_stopping_metric_names

    def get_early_stopping_values_epochs(self, early_stopping_metric_results):
        early_stopping_values_epochs = {}
        for early_stopping_metric in self.settings["early_stopping"]["metrics"]:
            if isinstance(early_stopping_metric_results[early_stopping_metric][0], list):
                for idx in range(len(early_stopping_metric_results[early_stopping_metric][0])):
                    early_stopping_values_epochs[f"{early_stopping_metric}_{idx}"] = np.array(
                        early_stopping_metric_results[early_stopping_metric])[:, idx]
            else:
                early_stopping_values_epochs[f"{early_stopping_metric}"] = early_stopping_metric_results[
                    early_stopping_metric]
        return early_stopping_values_epochs


class SingleModel(NonProfiling):

    def __init__(self, settings, dataset, da_function, random_states=None):
        super().__init__(settings, dataset, da_function)
        self.settings = settings
        self.dataset = dataset
        self.da_function = da_function
        self.labels_key_guesses_attack_set = None
        self.labels_key_guesses_validation_set = None
        self.analysis_db_controls = None
        self.custom_callbacks = None
        self.random_states = random_states

    def run(self, model):

        initial_weights = model["model"].get_weights()
        model["model"].summary()

        start = time.time()

        hp_id = None

        for idx, n_profiling in enumerate(self.settings["profiling_analyzer_steps"]):

            """ Run training phase """

            model["model"].set_weights(initial_weights)
            history = self.train_model(model["model"], n_traces=n_profiling)
            self.custom_callbacks = self.get_custom_callbacks()

            """ Save hyper-parameters combination to database """
            if idx == 0:
                hyper_parameters_list = Utils().get_hyperparameters_from_model(model["model"])
                hp_id = self.analysis_db_controls.save_hyper_parameters_to_database(hyper_parameters_list)

            """ Retrieve metrics from model and save them to database """
            if self.settings["use_early_stopping"]:
                early_stopping_metrics = self.get_early_stopping_metrics(
                    self.get_builtin_callbacks()["early_stopping_callback"].get_metric_results())
                early_stopping_epoch_values = self.get_early_stopping_values_epochs(
                    self.get_builtin_callbacks()["early_stopping_callback"].get_metric_results())
            else:
                early_stopping_metrics = None
                early_stopping_epoch_values = None
            self.analysis_db_controls.save_generic_metrics(history, early_stopping_metrics, early_stopping_epoch_values,
                                                           hp_id,
                                                           n_profiling_traces=n_profiling if self.settings[
                                                               "use_profiling_analyzer"] else None)

            """ Compute SCA metrics and save them to database """
            self.compute_metrics(model, self.analysis_db_controls, hp_id,
                                 n_traces=n_profiling if self.settings["use_profiling_analyzer"] else None,
                                 random_states=self.random_states)

            """ Save model description (keras style) to database """
            model_description = Utils().keras_model_as_string(model["model"], model["method_name"])
            self.analysis_db_controls.save_model_description_to_database(model_description, model["method_name"], hp_id)

            """ Save visualization results to database """
            if self.settings["use_visualization"]:
                self.analysis_db_controls.save_visualization_results_to_database(
                    self.get_builtin_callbacks()["input_gradients_callback"].input_gradients_epochs(),
                    self.get_builtin_callbacks()["input_gradients_callback"].input_gradients_sum(), hp_id)

            """ Save confusion matrix results to database """
            if self.settings["use_confusion_matrix"]:
                self.analysis_db_controls.save_confusion_matrix_results_to_database(
                    self.get_builtin_callbacks()["confusion_matrix_callback"].get_confusion_matrix(), hp_id)

            """ update database settings"""
            self.analysis_db_controls.update_results_in_database(time.time() - start)

            backend.clear_session()


class SingleProcess:

    def __init__(self, dataset, settings, models, da_function, labels_key_guesses_attack_set=None,
                 labels_key_guesses_validation_set=None,
                 random_states=None):
        self.dataset = dataset
        self.settings = settings
        self.models = models
        self.da_function = da_function
        self.custom_callbacks = None
        self.labels_key_guesses_attack_set = labels_key_guesses_attack_set
        self.labels_key_guesses_validation_set = labels_key_guesses_validation_set
        self.random_states = random_states

    def run(self):

        """ Create analysis in database """
        analysis_db_controls = AnalysisDatabaseControls(self.settings)
        self.settings["analysis_id"] = analysis_db_controls.analysis_id
        analysis_db_controls.save_leakage_model_to_database()

        """ Generate list of labels for all key guesses for validation and attack sets """
        if self.labels_key_guesses_attack_set is None and self.labels_key_guesses_validation_set is None:
            lm_control = LeakageModelControls(self.settings)
            self.labels_key_guesses_attack_set, self.labels_key_guesses_validation_set = lm_control.compute_labels_key_guesses(
                self.dataset)

        """ If number of neural networks is larger than 1, starts multi-model process. Otherwise, runs single-model process """
        if len(self.models) > 1:
            multi_model = MultiModel(self.settings, self.dataset, self.da_function, random_states=self.random_states)
            multi_model.labels_key_guesses_attack_set = self.labels_key_guesses_attack_set
            multi_model.labels_key_guesses_validation_set = self.labels_key_guesses_validation_set
            multi_model.analysis_db_controls = analysis_db_controls
            multi_model.run(self.models)
            self.custom_callbacks = multi_model.custom_callbacks
        else:
            single_model = SingleModel(self.settings, self.dataset, self.da_function, random_states=self.random_states)
            single_model.labels_key_guesses_attack_set = self.labels_key_guesses_attack_set
            single_model.labels_key_guesses_validation_set = self.labels_key_guesses_validation_set
            single_model.analysis_db_controls = analysis_db_controls
            single_model.run(self.models["0"])
            self.custom_callbacks = single_model.custom_callbacks


def generate_random_hyperparameters(model_type="mlp"):
    """
    Function to generate a random set of hyperparameters
    """

    if model_type == "mlp":
        hp = {
            "neurons": random.choice([200, 300, 400, 500, 600, 700, 800, 900, 1000]),
            "batch_size": random.choice([50, 100]),
            "layers": random.choice([1, 2, 3, 4]),
            "activation": random.choice(["elu", "selu", "relu"]),
            "learning_rate": random.choice(
                [0.005, 0.0025, 0.002, 0.001, 0.0025, 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
            "kernel_initializer": random.choice(
                ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
            # "kernel_regularizer": random.choice([None, "l1", "l2"]),
            "kernel_regularizer": random.choice([None]),
            # "dropout": random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "dropout": random.choice([0]),
            "optimizer": random.choice(["Adam"])
        }
    else:
        hp = {
            "neurons": random.choice([200, 300, 400, 500, 600, 700, 800, 900, 1000]),
            "batch_size": random.choice([50, 100, 200, 300, 400, 500, 1000, 2000]),
            "layers": random.choice([1, 2]),
            "filters": random.choice([4, 8, 12, 16, 32]),
            "kernel_size": random.choice([5, 10, 20, 30, 40]),
            "pool_type": random.choice(["Average", "Max"]),
            "pool_size": random.choice([2, 4]),
            "conv_layers": random.choice([1, 2, 3, 4]),
            "activation": random.choice(["elu", "selu", "relu"]),
            "learning_rate": random.choice(
                [0.005, 0.0025, 0.002, 0.001, 0.0025, 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
            "kernel_initializer": random.choice(
                ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
            # "kernel_regularizer": random.choice([None, "l1", "l2"]),
            "kernel_regularizer": random.choice([None]),
            # "dropout": random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "dropout": random.choice([0]),
            "optimizer": random.choice(["Adam", "RMSprop"])
        }
        hp["pool_strides"] = hp["pool_size"]
        conv_stride_options = [1, 2, 3, 4, 5, 10, 15, 20]
        possible_stride_options = []
        for i, st in enumerate(conv_stride_options):
            if st <= hp["kernel_size"]:
                possible_stride_options.append(st)
        # hp["strides"] = random.choice(possible_stride_options)
        hp["strides"] = random.choice([1, 2, 3, 4])

    if hp["kernel_regularizer"] is not None:
        hp["kernel_regularizer_value"] = random.choice([0.0005, 0.0001, 0.0002, 0.00025, 0.00005, 0.00001])

    hp["seed"] = np.random.randint(1048576)

    return hp


class NPAisy(aisy_sca.Aisy):
    def run(self, key_rank_executions=100, key_rank_report_interval=10, key_rank_attack_traces=1000, visualization=None,
            data_augmentation=None, ensemble=None, grid_search=None, random_search=None, early_stopping=None,
            confusion_matrix=False,
            callbacks=None, profiling_analyzer=None):

        """
        Main AISY framework function. This function runs neural network training for non-profiled side-channel analysis in a known-key setting.
        """

        self.settings["key_rank_executions"] = key_rank_executions
        self.settings["key_rank_attack_traces"] = key_rank_attack_traces
        self.settings["key_rank_report_interval"] = key_rank_report_interval
        self.settings["timestamp"] = str(time.time()).replace(".", "")

        analysis_controls = AnalysisControls(self.settings, self.models)
        analysis_controls.update_settings(random_search, grid_search, visualization, early_stopping, data_augmentation,
                                          callbacks, ensemble,
                                          confusion_matrix, profiling_analyzer)
        analysis_controls.check_errors()
        da_function = data_augmentation[0] if data_augmentation is not None else None

        """ Load data sets """
        if self.dataset is None:
            ds_controls = DatasetControls(self.settings)
            ds_controls.read()
            self.dataset = ds_controls.get_dataset()

        """ Run main process """
        if self.settings["hyperparameter_search"]:
            hyperparameter_search_process = HyperparameterSearchProcess(
                self.dataset, self.settings, self.models, da_function, self.hyperparameters_reproducible,
                labels_key_guesses_attack_set=self.labels_key_guesses_attack_set,
                labels_key_guesses_validation_set=self.labels_key_guesses_validation_set,
                loss_function=self.loss_function, random_states=self.random_states)
            hyperparameter_search_process.run()

            """ Get custom callbacks """
            self.custom_callbacks = hyperparameter_search_process.custom_callbacks
        else:
            single_process = SingleProcess(
                self.dataset, self.settings, self.models, da_function,
                labels_key_guesses_attack_set=self.labels_key_guesses_attack_set,
                labels_key_guesses_validation_set=self.labels_key_guesses_validation_set,
                random_states=self.random_states)
            single_process.run()

            """ Get custom callbacks """
            self.custom_callbacks = single_process.custom_callbacks


aisy = NPAisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["ASCAD.h5"])
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_batch_size(400)
nb_epochs = 20
hp_values = generate_random_hyperparameters(model_type="cnn")
hp_values["epochs"] = nb_epochs
aisy.set_epochs(nb_epochs)
# dataset has to be read
aisy.settings['split_test_set'] = True
ds_controls = DatasetControls(aisy.settings, split_test_set_ratio=0.8)
ds_controls.read()
aisy.dataset = ds_controls.get_dataset()
features = len(aisy.dataset.x_attack[0])
loss_function = "z_score_mse"
aisy.set_neural_network(mlp_np(features, loss_function, hp_values))

if aisy.labels_key_guesses_attack_set is None and aisy.labels_key_guesses_validation_set is None:
    lm_control = LeakageModelControls(aisy.settings)
    aisy.labels_key_guesses_attack_set, aisy.labels_key_guesses_validation_set = lm_control.compute_labels_key_guesses(
        aisy.dataset)

custom_callbacks = [
    {
        "class": "custom.custom_callbacks.MORE_callback.GetSCAMetric",
        "name": "GetSCAMetric",
        "parameters": {
            "loss": loss_function,
            "y_train": aisy.labels_key_guesses_attack_set.T.tolist(),
            "y_val": aisy.labels_key_guesses_validation_set.T.tolist()
        }
    }
]

aisy.run(
    callbacks=custom_callbacks
)


