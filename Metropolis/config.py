# config.py

def get_config(model_name):

    return {
        "model_name": model_name,

        "T_in": 580,
        "T_out": 100,
        "n_features": 2,

        "latent_dim": 64,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "patch_size": 16,

        "dropout": 0.1,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 50,

        "uses_teacher_forcing": model_name in [
            "lstm",
            "lstm_attention",
            "bilstm_transformer"
        ]
    }