import tensorflow as tf
import nfp
import tensorflow_addons as tfa

def build_nfp_gnn(preprocessor, num_features, layers, lr):
    atom = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name="atom")
    bond = tf.keras.layers.Input(shape=[None], dtype=tf.int32, name="bond")
    conn = tf.keras.layers.Input(shape=[None, 2], dtype=tf.int32, name="connectivity")

    atom_state = tf.keras.layers.Embedding(
        preprocessor.atom_classes,
        num_features,
        mask_zero=True
    )(atom)

    bond_state = tf.keras.layers.Embedding(
        preprocessor.bond_classes,
        num_features,
        mask_zero=True
    )(bond)

    global_state = nfp.GlobalUpdate(units=8, num_heads=1)(
        [atom_state, bond_state, conn]
    )

    for _ in range(layers):
        bond_state = tf.keras.layers.Add()([
            bond_state,
            nfp.EdgeUpdate()([atom_state, bond_state, conn, global_state])
        ])
        atom_state = tf.keras.layers.Add()([
            atom_state,
            nfp.NodeUpdate()([atom_state, bond_state, conn, global_state])
        ])
        global_state = tf.keras.layers.Add()([
            global_state,
            nfp.GlobalUpdate(units=8, num_heads=1)(
                [atom_state, bond_state, conn]
            )
        ])

    out = tf.keras.layers.Dense(1)(global_state)
    model = tf.keras.Model([atom, bond, conn], out)

    model.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-5
        ),
        loss="mae",
    )
    return model
