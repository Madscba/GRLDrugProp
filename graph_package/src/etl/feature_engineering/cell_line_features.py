import pandas as pd
from graph_package.configs.directories import Directories
from graph_package.utils.helpers import logger
import gget
import json
import re


def make_cell_line_features():
    """Generate cell line features from the CCLE gene expressions."""
    save_path = Directories.DATA_PATH / "features" / "cell_line_features"
    save_path.mkdir(parents=True, exist_ok=True)
    if not (save_path / "raw" / "OmicsCNGene.csv").exists():
        logger.error(
            "Please download the file OmicsCNGene.csv.csv gene expressions from https://depmap.org/portal/download/all/.\n \
                Store it in ~/data/features/cell_line_features/raw"
        )
        raise FileNotFoundError
    gene_expressions = pd.read_csv(
        save_path / "raw" / "OmicsCNGene.csv.csv", index_col=0
    )
    gene_expressions = pd.read_csv('OmicsCNGene.csv', index_col=0)
    if not (save_path / "ncbi_ids.csv").exists():
        # This is not used since landmark genes are uploaded to github, it is only here for reference.
        url = "https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/data/CCLE_RNAseq_rsem_transcripts_tpm_20180929/CCLE_RNAseq_rsem_transcripts_tpm_20180929_954.csv"
        landmark_genes = pd.read_csv(url)
        ens_ids = list(landmark_genes['gene_id'])
        logger.info('Downloading NCBI ids for landmark genes. Takes ~hour.')
        ncbi_ids = gget.info(ens_ids=ens_ids,uniprot=False,verbose=True)
        ncbi_ids.to_csv(save_path / "ncbi_ids.csv")

    ncbi_ids = pd.read_csv(save_path / "ncbi_ids.csv", index_col=0, dtype=str)
    rename_dict = {
        gene: re.findall(r"\((.*?)\)", gene)[0] for gene in gene_expressions.columns
    }
    gene_expressions.rename(columns=rename_dict, inplace=True)
    gene_ids = set(ncbi_ids["ncbi_gene_id"]).intersection(set(gene_expressions.columns))
    gene_expressions = gene_expressions[list(gene_ids)]
    gene_expressions.fillna(gene_expressions.mean(), inplace=True)

    cell_dict = json.load(
        open(Directories.DATA_PATH / "bronze" / "drugcomb" / "cell_line_dict.json", "r")
    )
    feature_dict = {
        k: list(gene_expressions.loc[v["depmap_id"]])
        for k, v in cell_dict.items()
        if v["depmap_id"] in gene_expressions.index
    }
    json.dump(feature_dict, open(save_path / "CCLE_954_gene_express.json", "w"))


if __name__ == "__main__":
    make_cell_line_features()