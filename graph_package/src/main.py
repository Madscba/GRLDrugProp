"""main module."""

import torch
from torchdrug import datasets
from torchdrug import core, models, tasks

# https://torchdrug.ai/docs/quick_start.html

# Make a train_fig for each type of model we want to train. Then we will save and load this
train_config = {
    "model": "RESACL",
    "data_type": "triple",
    "train_device": "cpu",
}  # "logger": "local"


def load_data(model_type: str = ""):
    """Fetch formatted data depending on modelling task

    data_type: str: [triplets, DDI,DPI,PPI, SMILE]"""
    if model_type == "triplet":
        # Load triplet data with function from src/data_processing
        pass
    elif model_type == "DDI,DPI,PPI":
        # Load drug-drug, drug-protein, protein-protein interaction data with function from src/data_processing
        pass
    elif model_type == "SMILE":
        # Load SMILE data with function from src/data_processing
        pass
    else:
        dataset = datasets.ClinTox("~/molecule-datasets/")
    return dataset


if __name__ == "__main__":
    # 1. Load dataset
    dataset = load_data(train_config["data_type"])
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

    # 2. Model definition
    model = models.GIN(
        input_dim=dataset.node_feature_dim,
        hidden_dims=[256, 256, 256, 256],
        short_cut=True,
        batch_norm=True,
        concat_hidden=True,
    )

    # 3. Define task, criterion and metrics
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
