# Foundational FSOD with RoboFlow-20VL: MQDet baseline
[![arXiv](https://img.shields.io/badge/arXiv-2312.14494-b31b1b.svg)](https://arxiv.org/abs/2312.14494)
[![challenge](https://img.shields.io/badge/EvalAI-FSOD_Challenge-green)](https://eval.ai/web/challenges/challenge-page/2459/overview)

### This repository is adapted from the [Foundational FSOD paper's code](https://github.com/anishmadan23/foundational_fsod/tree/mqdet). It currently supports MQ-Det baseline for the [RF-20VL Challenge](https://eval.ai/web/challenges/challenge-page/2459/overview). 


## Updates
- Switch to the FSOD_RF20VL branch for the Detic Baselines as part of the [Roboflow-VL Challenge](https://eval.ai/web/challenges/challenge-page/2459/overview)
- Switch to the main branch to run nuImages Detic experiments. 
<!-- - NOTE: Use the [test_set.json](https://huggingface.co/anishmadan23/foundational_fsod/blob/main/nuimages_mqdet_annotation_data/no_wc/test_set.json) file for evaluating nuImages test-set performance. -->

See [MQDet README](MQDET_README.md) for details on installation and setup.


## Replicating Detic ZS baseline on all datasets in RF-20VL
1. We provide the csv file with links: `datasets_links.csv` . To download and preprocess the data, run 
   
```bash
python rf_scripts/preprocess_data.py
```

2. Generate training/evaluation configs to run baselines
   
```bash
python rf_scripts/generate_cfg_files.py
```

3. Generate commands for extracting vision queries, to be used for visual prompting

```bash
python rf_scripts/generate_vision_query_extraction_cmds.py
```

4. Run script with generated commands to extract vision queries

```bash 
sh rf_scripts/extract_vision_queries.sh
```

5.  Generate commands for running fine-tuning free evaluation
Change number of gpus for each of the experiments (text-only, vision-only, text+vision) in the script before running as mentioned below:

```bash 
python rf_scripts/generate_ftfree_cmds.py
```

6. The above command generates 3 shell scripts: 1 for each (text-only, vision-only, text+vision). Run scripts like
   
```bash 
sh rf_scripts/run_ftfree_eval_model_large_text_only.sh
```

7. Finally, combine predictions for all 3 experiments across datasets into the format used for competition submissions. Note that MQDet results are 1-indexed whereas the competition submissions expect it to be 0-indexed. We take care of this issue in the script below, so no additional checks are required from the user.
   
```bash 
sh rf_scripts/combine_preds.py
```



### Acknowledgements
We thank the authors of the [MQDet Repository](https://github.com/YifanXu74/MQ-Det) for their open-source implementations of the MQDet method. This repository is an adaptation of the MQDet codebase to support Foundational FSOD on the nuImages dataset.

### License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


