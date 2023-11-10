# todo
# load model from checkout

# use model to predict

# save predictions

# allow easy fetching of meta data

# visualize AUC PR and AUC ROC with some data points plottet (corresponding to best f1). (add flag to show plots in GUI or Wandb)
# report AUC measures and F-1

# confusion matrices
# Histogram of incorrect, correct over drug, cancer-cell, drug-combination, drug-cell combination.

# Fairness evaluation (split across nodes)

# find inherently hard triplets

if __name__ == "__main__":
    # load checkpoint
    checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    model = LitAutoEncoder.load_from_checkpoint(
        checkpoint, encoder=encoder, decoder=decoder
    )

    # choose your trained nn.Module
    encoder = model.encoder
    encoder.eval()

    # embed 4 fake images!
    fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
    trainer.test(
        model,
        dataloaders=data_loaders["test"],
        ckpt_path=checkpoint_callback.best_model_path,
    )

    # check_point_path = Directories.CHECKPOINT_PATH / model_name
    # if os.path.isdir(check_point_path):
    #     shutil.rmtree(check_point_path)
    # else:
    #     print("Model not found found!")
    #     trainer.test(
    #         model,
    #         dataloaders=data_loaders['test'],
    #         ckpt_path=checkpoint_callback.best_model_path,
    #     )
