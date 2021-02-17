from commons.sca_aisy_aes import Aisy

aisy = Aisy()
# aisy.set_datasets_root_folder("D:/traces/")
aisy.set_dataset("ascad-variable.h5")
aisy.set_database_name("database_ascad.sqlite")
aisy.set_aes_leakage_model(leakage_model="HW", byte=2)
aisy.set_number_of_profiling_traces(100000)
aisy.set_number_of_attack_traces(2000)
aisy.set_batch_size(400)
aisy.set_epochs(10)

# # for each hyper-parameter, specify the options in the grid search
grid_search = {
    "neural_network": "mlp",
    "hyper_parameters_search": {
        # 'conv_layers': [1, 2],
        # 'kernel_1': [4, 8],
        # 'kernel_2': [2, 4],
        # 'stride_1': [1],
        # 'stride_2': [1],
        # 'filters_1': [8, 16],
        # 'filters_2': [8, 16],
        # 'pooling_type_1': ["Average", "Max"],
        # 'pooling_type_2': ["Average", "Max"],
        # 'pooling_size_1': [1, 2],
        # 'pooling_size_2': [1, 2],
        # 'pooling_stride_1': [1, 2],
        # 'pooling_stride_2': [1, 2],
        'neurons': [100, 200],
        'layers': [3, 4],
        'learning_rate': [0.001],
        'activation': ["selu"]
    },
    "metric": "guessing_entropy",
    "stop_condition": True,
    "stop_value": 1.0,
    "train_after_search": True
}

# for each hyper-parameter, specify the options in the grid search
# grid_search = {
#     "neural_network": "mlp",
#     "hyper_parameters_search": {
#         'neurons': [100, 200],
#         'layers': [3, 4],
#         'learning_rate': [0.001, 0.0001],
#         'activation': ["relu", "selu"]
#     },
#     "metric": "guessing_entropy",
#     "stop_condition": False,
#     "stop_value": 1.0,
#     "train_after_search": True
# }

aisy.run(
    grid_search=grid_search
)
