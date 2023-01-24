# Coperia_inference
A useful toolkit for inference with Coperia's models.

## Installation
```bash
git clone https://github.com/JMasr/coperia_inference.git
cd coperia_inference
conda env create -f environment.yml
pip install -r requirements.txt
```

## Usage
### Demo case with fake data (for testing)
```bash
python inference.py
```

### Arguments

```bash
usage: inference.py [-i] [--file_list path/to/data.scp] 
                    [-s][--scores_path path/to/scores.txt]
                    [-p][--model_path path/to/final.pt]
                    [-m][--model_config path/to/model_config]
                    [-c][--inference_config path/to/inference_config]
                    [-f][--feature_config path/to/feature_config]
                    [-r][--ref_file path/to/references]
                    [-o][--output_file path/to/results.pkl]
```