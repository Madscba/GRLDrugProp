from graph_package.configs.directories import Directories


def save_rescal_embedding(config, model):
    import pandas as pd

    emb_save_path = (
        Directories.DATA_PATH / "embeddings" / f"rescal_{config.model.dim}.csv"
    )
    # with open(emb_save_path, 'wb') as handle:
    #     pickle.dump(model.model.ent_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(model.model.ent_embeddings.weight.data.numpy())
    pd.DataFrame(model.model.ent_embeddings.weight.data.numpy()).to_csv(emb_save_path)
    print("dumped file at", emb_save_path)
