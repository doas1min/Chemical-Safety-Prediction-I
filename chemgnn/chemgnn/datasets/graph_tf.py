# chemgnn/datasets/graph_tf.py
import tensorflow as tf
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor

def build_preprocessor(atom_featurizer, bond_featurizer, smiles_series):
    preprocessor = SmilesPreprocessor(
        atom_features=atom_featurizer,
        bond_features=bond_featurizer,
        explicit_hs=False,
    )
    for s in smiles_series:
        preprocessor.construct_feature_matrices(s, train=True)
    return preprocessor


def data_generator(preprocessor, df, smiles_col, target_col):
    for _, row in df.iterrows():
        inputs = preprocessor.construct_feature_matrices(
            row[smiles_col], train=False
        )
        yield (
            {
                "atom": inputs["atom"],
                "bond": inputs["bond"],
                "connectivity": inputs["connectivity"],
            },
            row[target_col],
        )


def make_tf_dataset(preprocessor, df, smiles_col, target_col,
                    batch_size, shuffle=False):
    output_signature = (
        preprocessor.output_signature,
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(preprocessor, df, smiles_col, target_col),
        output_signature=output_signature,
    )
    if shuffle:
        ds = ds.shuffle(200)
    return ds.padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
