import csv
import pandas as pd

# Define the input and output file paths
csv_file_path = "data/bronze/summary_v_1_5_subset.csv"
tsv_file_path = "data/bronze/output.tsv"


# Function to convert CSV to TSV
def csv_to_tsv(input_file, output_file):
    with open(input_file, "r", newline="") as csv_input_file:
        with open(output_file, "w", newline="") as tsv_output_file:
            csv_reader = csv.reader(csv_input_file)
            tsv_writer = csv.writer(tsv_output_file, delimiter="\t")

            for row in csv_reader:
                tsv_writer.writerow(row)


if __name__ == "__main__":
    df = pd.read_csv(csv_file_path)
    unique_drug_names = df["drug_row"].unique()
    unique_relation_names = df["drug_row"].unique()

    entity_vocab = {index: value for index, value in enumerate(unique_drug_names)}
    inv_entity_vocab = {value: index for index, value in enumerate(unique_drug_names)}
    relation_vocab = {index: value for index, value in enumerate(unique_relation_names)}
    inv_relation_vocab = {
        value: index for index, value in enumerate(unique_relation_names)
    }

    self.graph = data.Graph(triplets, num_node=num_node, num_relation=num_relation)
    self.entity_vocab = entity_vocab
    self.relation_vocab = relation_vocab
    self.inv_entity_vocab = inv_entity_vocab
    self.inv_relation_vocab = inv_relation_vocab
    try:
        csv_to_tsv(csv_file_path, tsv_file_path)
        print(
            f'CSV file "{csv_file_path}" has been successfully converted to TSV file "{tsv_file_path}".'
        )

    except Exception as e:
        print(f"An error occurred: {str(e)}")
