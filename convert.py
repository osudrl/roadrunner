import argparse
import os

import torch

from nn.actor import LSTMActor
from nn.critic import LSTMCritic


"""
Convert the old LSTM model to the new and faster LSTM model.
Because of limitations with the torch.nn.LSTM, we can only support models with symmetric layers.
Such as: "128,128". We no longer support models like "128,64".
"""


def map_weights(og_model_state_dict):
    """
    Naming scheme map from old lstms to new lstms
    """
    new_model_state_dict = {}
    for key, value in og_model_state_dict.items():
        if "layers." in key:
            layer_num = key.split('.')[1]
            new_key = key.replace(f"layers.{layer_num}", f"lstm") + f"_l{layer_num}"
            new_model_state_dict[new_key] = value
        else:
            new_model_state_dict[key] = value
    return new_model_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, required=True, help="Size of each hidden layer. Only supports symmetric layers.")
    parser.add_argument("--dir_path", type=str, required=True, help="Will convert all .pt models in this dir")
    args = parser.parse_args()

    if not os.path.isdir(args.dir_path):
        raise RuntimeError("dir_path is not a directory")

    for file_name in os.listdir(args.dir_path):
        if file_name.endswith(".pt") and not file_name[:-3].endswith("new"):
            print("\n\n", "="*50, "\n", file_name, "\n", "="*50)
            file_path = os.path.join(args.dir_path, file_name)
            model_dict = torch.load(file_path, map_location='cpu')

            # load the model to get the model_state_dict architecture
            layers = [args.hidden_size] * args.num_layers
            if model_dict["model_class_name"] == "LSTMActor":
                model = LSTMActor(
                    model_dict["obs_dim"],
                    model_dict["action_dim"],
                    std=model_dict["std"],
                    bounded=model_dict["bounded"],
                    layers=layers,
                    learn_std=model_dict["learn_std"],
                    )
            elif model_dict["model_class_name"] == "LSTMCritic":
                model = LSTMCritic(
                    model_dict["input_dim"], # for some reason this is called input_dim instead of obs_dim
                    layers=layers
                    )
            else:
                raise RuntimeError("Unknown model class name")


            # print the model architectures to confirm they match
            print("\nOG MODEL DICT:")
            for name, weights in model_dict["model_state_dict"].items():
                print(f"{name}: {weights.shape}, mean value: {weights.mean():.6f}")

            print("\nNEW MODEL DICT:")
            for name, weights in model.state_dict().items():
                print(f"{name}: {weights.shape}")


            # remap the weights to the new model
            # assert len(model_dict["model_state_dict"]) == len(list(model.named_parameters())), "Number of weight matrices do not match between models"
            new_model_dict = map_weights(model_dict["model_state_dict"])


            print("\nOG MODEL DICT AFTER REMAPPING NAMES:")
            for name, weights in new_model_dict.items():
                print(f"{name}: {weights.shape}, mean value: {weights.mean():.6f}")


            # save the new model
            model_dict["model_state_dict"] = new_model_dict
            torch.save(model_dict, f"{file_path[:-3]}_new.pt")