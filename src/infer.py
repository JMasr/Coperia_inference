from src.utils import *
from src.models import *

from tqdm import tqdm
from configparser import ConfigParser


def inference(file_list: str, output_path: str,
              inference_config: ConfigParser, model_config: ConfigParser, feature_config: ConfigParser,
              device: str = torch.device('cpu')) -> dict:
    """Script to do inference using trained model config, feature_config: model configuration
    and feature configuration files
    :param device: device where the inference take place
    :param file_list: List of files as in "<id> <file-path>" format
    :param output_path: output file, its content will be "<id> <probability-score>"
    :param inference_config: Configuration of inference flow
    :param model_config: Configuration of model
    :param feature_config: Configuration of the feature extractor
    :return: A list of id -> labels with a probability score
    """
    # Load model parameters
    model_args = {}
    if model_config is not None:
        for key in model_config['default'].keys():
            model_args[key] = convert_type(model_config['default'][key])
    else:
        raise ValueError('Expected an architecture')

    # Load model, use CPU for inference
    model = getNet(model_args)

    # Feature extractor
    FE = FeatureExtractor(feature_config['default'])

    # Loop over all files
    scores = {}
    file_list = open(file_list).readlines()
    file_list = [line.strip().split() for line in file_list]
    for fileId, path in tqdm(file_list):
        # Prepare features
        try:
            F = FE.extract(path)
        except IOError:
            print('failed for ' + fileId)
            continue

        if inference_config['training_dataset'].get('apply_mean_norm', False):
            F = F - torch.mean(F, dim=0)
        if inference_config['training_dataset'].get('apply_var_norm', False):
            F = F / torch.std(F, dim=0)
        feat = F.to(device)

        # Input mode
        seg_mode = inference_config['training_dataset'].get('mode', 'file')
        if seg_mode == 'file':
            feat = [feat]
        elif seg_mode == 'segment':
            segment_length = int(inference_config['training_dataset'].get('segment_length', 300))
            segment_hop = int(inference_config['training_dataset'].get('segment_hop', 10))
            feat = [feat[i:i + segment_length, :] for i in range(0, max(1, F.shape[0] - segment_length), segment_hop)]
        else:
            raise ValueError('Unknown eval model')

        if model_args['architecture'].lower() == 'lstmclassifier':
            with torch.no_grad():
                output_score = model.predict_proba(feat)
                output_score = sum(output_score)[0].item() / len(output_score)
        else:
            output_score = model.validate([F])
            output_score = np.mean(output_score[0], axis=0)[1]

        # Average the scores of all segments from the input file
        scores[fileId] = output_score

    # Write output scores
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for item in scores:
            f.write(item + " " + str(scores[item]) + "\n")

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', '-i', required=True)  # Path to a .scp file
    parser.add_argument('--output_path', '-o', required=True)  # Path to an output .txt file with the results
    parser.add_argument('--model_config', '-m', default='config/model_config')  # Path to model's configuration
    parser.add_argument('--inference_config', '-c', default='config/infer_config')  # Path to inference's configuration
    parser.add_argument('--feature_config', '-f', default='config/feature_config')  # Path to featur's configuration

    args = parser.parse_args()

    inf_config = ConfigParser()
    inf_config.read(args.inference_config)

    feat_config = ConfigParser()
    feat_config.read(args.feature_config)

    mdl_config = ConfigParser()
    mdl_config.read(args.model_config)

    inference(args.file_list, args.output_path, inf_config, mdl_config, feat_config)
