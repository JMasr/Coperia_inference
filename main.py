# This is a sample Python script.
import argparse
import configparser

from src.infer import inference

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', '-i', required=True)  # Path to a .scp file
    parser.add_argument('--output_path', '-o', required=True)     # Path to an output .txt file with the results
    parser.add_argument('--model_path', '-p', default='models/breathing-deep/models/final.pt')  # Path to model .ptl
    parser.add_argument('--model_config', '-m', default='config/model_config')  # Path to model's configuration
    parser.add_argument('--inference_config', '-c', default='config/train_config')  # Path to inference's configuration
    parser.add_argument('--feature_config', '-f', default='config/feature_config')  # Path to feature's configuration

    args = parser.parse_args()

    inference_config = configparser.ConfigParser()
    inference_config.read(args.inference_config)

    feature_config = configparser.ConfigParser()
    feature_config.read(args.feature_config)

    model_config = configparser.ConfigParser()
    model_config.read(args.model_config)

    inference(args.model_path, args.file_list, args.output_path, inference_config, model_config, feature_config)
