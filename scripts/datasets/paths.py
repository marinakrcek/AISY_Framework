def get_dataset_filepath(dataset_root_folder, dataset_name, npoi, leakage_model):
    dataset_dict = {
        "ASCAD": {
            10000: f"{dataset_root_folder}/ASCAD_nopoi_window_20.h5",
            700: f"{dataset_root_folder}/ascad.h5"
        },
        "ascad-variable": {
            25000: f"{dataset_root_folder}/ascad-variable_nopoi_window_20.h5",
            1400: f"{dataset_root_folder}/ascad-variable.h5"
        },
        "dpa_v42": {
            15000: f"{dataset_root_folder}/dpa_v42_nopoi_window_20.h5"
        },
        "ches_ctf": {
            15000: f"{dataset_root_folder}/ches_ctf_nopoi_window_20.h5"
        },
        "eshard": {
            1400: f"{dataset_root_folder}/eshard.h5",
        }
    }
    return dataset_dict[dataset_name][npoi]
