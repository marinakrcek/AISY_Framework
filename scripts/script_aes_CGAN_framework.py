import aisy_sca
from app import *
from custom.custom_models.CGAN_models import *
from custom.custom_models.neural_networks import *

if __name__ == "__main__":
    dataset_root_path = "C:/Users/mkrcek/Documents/PhDTUDelft/source_code/datasets"
    results_root_path = "C:/Users/mkrcek/Documents/PhDTUDelft/source_code/AISY_Framework/results"
    features_root_path = "C:/Users/mkrcek/Documents/PhDTUDelft/source_code/AISY_Framework/features"

    arg_list = get_arguments(dataset_root_path, results_root_path, features_root_path)

    arguments = {
        "dataset_root_path": arg_list.dataset_root_path,
        "results_root_path": arg_list.results_root_path,
        "features_root_path": arg_list.features_root_path,
        "dataset_reference": arg_list.dataset_reference,
        "dataset_reference_dim": int(arg_list.dataset_reference_dim),
        "dataset_target": arg_list.dataset_target,
        "dataset_target_dim": int(arg_list.dataset_target_dim),
        "n_profiling_reference": int(arg_list.n_profiling_reference),
        "n_attack_reference": int(arg_list.n_attack_reference),
        "n_profiling_target": int(arg_list.n_profiling_target),
        "n_validation_target": int(arg_list.n_validation_target),
        "n_attack_target": int(arg_list.n_attack_target),
        "n_attack_ge": int(arg_list.n_attack_ge),
        "target_byte_reference": int(arg_list.target_byte_reference),
        "target_byte_target": int(arg_list.target_byte_target),
        "features": int(arg_list.features),
        "leakage_model": arg_list.leakage_model,
        "epochs": int(arg_list.epochs),
        "batch_size": int(arg_list.batch_size),
        "std_gaussian_noise_reference": float(arg_list.std_gaussian_noise_reference),
        "std_gaussian_noise_target": float(arg_list.std_gaussian_noise_target),
        "random_search_generator": False if int(arg_list.random_search_generator) == 0 else True,
        "random_search_discriminator": False if int(arg_list.random_search_discriminator) == 0 else True,
    }

    aisy = aisy_sca.Aisy()

    aisy.set_resources_root_folder(resources_root_folder)
    aisy.set_database_root_folder(databases_root_folder)
    aisy.set_datasets_root_folder(datasets_root_folder)
    aisy.set_database_name("database_ascad.sqlite")

    cgan = CGANSCA(args=arguments)
    cgan.train_cgan()

    # ds_controls = aisy_sca.DatasetControls(self.settings)
    # ds_controls.read()
    # target_dataset = ds_controls.get_dataset()
    #
    # key_byte = 2
    # sca_profiling_model = ASCAD_mlp
    # aisy.set_dataset(datasets_dict[arguments["dataset_target"]])
    # aisy.set_aes_leakage_model(leakage_model=arguments["leakage_model"], byte=key_byte)
    # aisy.set_batch_size(arguments["batch_size"])
    # aisy.set_epochs(arguments["epochs"])
    # aisy.set_neural_network(sca_profiling_model)
    # aisy_sca.aes_intermediates(aisy.dataset.plaintext_profiling, aisy.dataset.ciphertext_profiling)
    # aisy.run()
