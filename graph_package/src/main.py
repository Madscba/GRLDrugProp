"""main module."""

import torch
from torchdrug import core, datasets, tasks, models

# https://torchdrug.ai/docs/quick_start.html

# Make a train_fig for each type of model we want to train. Then we will save and load this
train_config = {
    "model": "GIN",
    "data_type": "triple",
    "train_device": "cpu",
    "split_method": "naive",
}  # "logger": "local"


def load_data(model_type: str = ""):
    """Fetch formatted data depending on modelling task

    data_type: str: [triplets, DDI,DPI,PPI, SMILE]
    data should be saved in as a torch.data.dataset, that can be inserted into torch.data.dataloader()
    """
    if model_type == "triplet":
        # Load triplet data with function from src/data_processing
        pass
    elif model_type == "DDI,DPI,PPI":
        # Load drug-drug, drug-protein, protein-protein interaction data with function from src/data_processing
        pass
    elif model_type == "SMILE":
        # Load SMILE data with function from src/data_processing
        pass
    else:  # Take the dataset from the torch drug demo
        dataset = datasets.ClinTox("~/molecule-datasets/")
    return dataset


def split_dataset(
    dataset, split_pct: dict = {"train": 0.8, "test": 0.2}, split_method: str = "naive"
):
    if split_method == "standard":
        # Use defined split_method to make a more intelligent split
        pass
    else:
        lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
        lengths += [len(dataset) - sum(lengths)]
        train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
    return train_set, valid_set, test_set


def load_model(model: str = "GIN"):
    if model == "RESCAL":
        # LOAD RESCAL MODEL
        pass
    elif model == "DeepDDS":
        # LOAD DeepDDS MODEL
        pass
    else:
        model = models.GIN(
            input_dim=dataset.node_feature_dim,
            hidden_dims=[256, 256, 256, 256],
            short_cut=True,
            batch_norm=True,
            concat_hidden=True,
        )
    return model


if __name__ == "__main__":
    # 1. Load dataset
    dataset = load_data(train_config["data_type"])

    # 2. Split dataset
    train_set, valid_set, test_set = split_dataset(
        dataset, split_method=train_config["split_method"]
    )

    # 3. Model definition
    model = load_model(train_config["model"])

    # 4. Define task, criterion and metrics
    if train_config["model"] == "GIN":
        task = tasks.PropertyPrediction(
            model, task=dataset.tasks, criterion="bce", metric=("auprc", "auroc")
        )

    # 4. Define optimize
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)

    # 5. Setup train and test loop.
    if False:
        logger_kwargs = {logger: "wandb"}
    else:
        logger_kwargs = {}

    if True:  # train using GPU or not. WANDB or not
        solver = core.Engine(
            task,
            train_set,
            valid_set,
            test_set,
            optimizer,
            batch_size=1024,
            **logger_kwargs
        )
    else:
        solver = core.Engine(
            task,
            train_set,
            valid_set,
            test_set,
            optimizer,
            batch_size=1024,
            gpus=[0],
            **logger_kwargs
        )

    # 6. Train model
    solver.train(num_epoch=1)

    # 7. Evaluate model
    # solver.evaluate("valid")
    # batch = data.graph_collate(valid_set[:8])
    # pred = task.predict(batch)

    # 8. Save or load model
    if False:
        with open("clintox_gin.json", "w") as fout:  # Save model
            json.dump(solver.config_dict(), fout)
            solver.save("clintox_gin.pth")

        with open("clintox_gin.json", "r") as fin:  # load model
            solver = core.Configurable.load_config_dict(json.load(fin))
            solver.load("clintox_gin.pth")
