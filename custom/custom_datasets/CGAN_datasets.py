import aisy_sca


class CGANDataset (aisy_sca.DatasetControls):

    def prepare_dataset(self, args):
        self.read()
        features_dim = args["features"]
        target_byte_reference = args["target_byte_reference"]
        target_byte_target = args["target_byte_target"]
        path = args["dataset_root_path"]
        traces_reference_dim = args["dataset_reference_dim"]
        traces_target_dim = args["dataset_target_dim"]
