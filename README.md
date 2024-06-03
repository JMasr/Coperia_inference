# Coperia-Inference
This repository is an useful toolkit for inference with [COPERIA project](https://coperia.es/)'s models.The project is developed by the [Multimedia Technologies Group](https://gtm.uvigo.es/en/) at the **atlanTTic Research Center, Universidade de Vigo**, in collaboration with [Bahia Software](https://bahiasoftware.es/home) and the [Galcian Health Service (SERGAS)](https://www.sergas.es/).

## About

To address the significant impact of Post-Acute Sequelae of SARS-CoV-2 (PASC) on global health, the [COPERIA project](https://coperia.es/) has been initiated in collaboration with the "Persistent COVID Unit of the Ourense Hospital" and primary care centers in the health area. The main objective of this project is to develop and clinically validate a comprehensive multidisciplinary platform that utilizes artificial intelligence for the diagnosis, empowerment, and clinical management of PASC patients. As part of this broader initiative, our study specifically focuses on investigating the potential of voice signal analysis as a non-invasive and accessible tool for classifying individuals as either patients with PASC or healthy individuals. The clinical study conducted in the context of the COPERIA project received ethical approval from the Clinical Research Ethics Committee of Galicia, and all procedures were conducted in compliance with the ethical principles outlined in the Declaration of Helsinki. Informed consent was obtained from all participants prior to their involvement in the study. The study was registered in the US Clinical Trials Registry under the code [NCT05629793].

## Getting Started
To get started with the demo, follow these steps:

1. Clone the repository:
```bash
git clone git@github.com:JMasr/coperia_inference.git
```
2. Navigate to the project directory:
```bash
cd coperia_demo
```
3. Import the envairoment using conda:
```bash
conda env create -f environment.yml
```
4. Install requirements:
```bash
pip install -r requirements.txt
```

5. Demo case with fake data (use only for testing)
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

## Acknowledgements
*. This work was supported by the CONECTA COVID programme, co-financed by the European Regional Development Fund (ERDF) within the Galicia ERDF operational programme 2014-2020 as part of the EU’s response to the COVID19 pandemic, and Axencia Galega de Innovacion (GAIN).

*. This work has received financial support from the Xunta de Galicia (Centro singular de investigación de Galicia accreditation 2019-2022).

*. This research has been funded by the Galician Regional Government under project ED431B 2021/24“GPC".

*. Thanks to the “Unidad de COVID Persistente del Hospital de Ourense” and the patients involved in the study.
