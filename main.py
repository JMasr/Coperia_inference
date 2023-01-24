# This is a sample Python script.
import argparse
import configparser

from src.infer import inference

# Press the green button in the gutter to run the script.
from src.scoring import scoring

if __name__ == '__main__':
    # Load inference arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', '-i', default='data/wav_test.scp')  # Path to a .scp file
    parser.add_argument('--scores_path', '-s', default='results/wav_test_scores.txt')     # Path to an output .txt file with the results
    parser.add_argument('--model_path', '-p', default='models/breathing-deep/models/final.pt')  # Path to model .ptl
    parser.add_argument('--model_config', '-m', default='config/model_config')  # Path to model's configuration
    parser.add_argument('--inference_config', '-c', default='config/train_config')  # Path to inference's configuration
    parser.add_argument('--feature_config', '-f', default='config/feature_config')  # Path to feature's configuration
    # Load scoring arguments
    parser.add_argument('--ref_file', '-r', default='data/reference')  # Path to the reference file with the labels
    parser.add_argument('--output_file', '-o', default='results/wav_test_results.pkl')  # Path to results .pkl file
    args = parser.parse_args()
    # Load the inference configuration
    inference_config = configparser.ConfigParser()
    inference_config.read(args.inference_config)
    # Load the feature configuration
    feature_config = configparser.ConfigParser()
    feature_config.read(args.feature_config)
    # Load the model configuration
    model_config = configparser.ConfigParser()
    model_config.read(args.model_config)
    # Run the inference
    inference(args.model_path, args.file_list, args.scores_path, inference_config, model_config, feature_config)
    # Run the scoring
    scoring(args.ref_file, args.scores_path, args.output_file)
