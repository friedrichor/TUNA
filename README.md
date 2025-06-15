# TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos (ACL 2025 Main)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2505.20124) [![Project Page](https://img.shields.io/badge/Project-Website-green)](https://friedrichor.github.io/projects/TUNA) [![🤗Hugging Face Dataset](https://img.shields.io/badge/🤗&nbsp;HF-Dataset-yellow)](https://huggingface.co/datasets/friedrichor/TUNA-Bench) [![🏆 Leaderboard](https://img.shields.io/badge/🏆&nbsp;Leaderboard-yellow)](https://friedrichor.github.io/projects/TUNA/#leaderboard)
</div>


## 👀 Overall

Videos are unique in their integration of temporal elements, including camera, scene, action, and attribute, along with their dynamic relationships over time. However, existing benchmarks for video understanding often treat these properties separately or narrowly focus on specific aspects, overlooking the holistic nature of video content. To address this, we introduce TUNA, a temporal-oriented benchmark for fine-grained understanding on dense dynamic videos, with two complementary tasks: captioning and QA. Our TUNA features diverse video scenarios and dynamics, assisted by interpretable and robust evaluation criteria. We evaluate several leading models on our benchmark, providing fine-grained performance assessments across various dimensions. This evaluation reveals key challenges in video temporal understanding, such as limited action description, inadequate multi-subject understanding, and insensitivity to camera motion, offering valuable insights for improving video understanding models.


<p align="center">
    <img src="./asserts/comparison_overall.png" width="90%">
    <br>
    Comparison with various video understanding benchmarks across several aspects: number of videos (<b>#Videos</b>); number of samples (<b>#Samp.</b>); annotation method (<b>Anno.</b>, with M/A denoting manual/automatic); domain (<b>Domain</b>); temporal orientation (<b>Temporal Orientated</b>); presence of scene transitions (<b>Scene Trans.</b>); consideration of camera (<b>Camera</b>) and scene (<b>Scene</b>); use of keypoints (<b>Key.</b>) for controllability and interpretability; Judgement of semantically identical yet diverse representations (<b>Sem.</b>); availability of multi-dimensional scores (<b>M.D.</b>); if global (<b>Global</b>) and fine-grained (<b>Fine.</b>) understanding are concerned.
</p>


## 🔍 Dataset


<p align="center">
    <img src="./asserts/dataset_statistics.png" width="60%">
</p>


## 🏆 Leaderboard

[here](https://friedrichor.github.io/projects/TUNA/#leaderboard)

## ⚖️ Evaluation

### Installation

```
cd VLMEvalKit
pip install -e .
```

### TUNA-CAP (Video Captioning)

You can run inference for TUNA-CAP using VLMEvalKit:

```bash
bash scripts/infer_tuna_cap.sh
## Equivalent to:
# cd VLMEvalKit
# bash scripts_tuna/infer_tuna_cap.sh
```

This will generate an output file, such as: `VLMEvalKit/outputs/MODEL/T2025xxx/MODEL_TUNA_CAP_1fps.xlsx`.  

To convert the above inference result into the standard JSON format required for submission and evaluation:

```bash
python evaluation/infer_result_to_submission.py
```

You can refer to the example submission format here: `evaluation/example/MyModel_TUNA_CAP_1fps_submission.json`.

Finally, evaluate TUNA-CAP and obtain the score:

```bash
bash scripts/eval_tuna_cap.sh
```

> **Note**: Due to the stochastic nature of some evaluation steps — primarily caused by the variability of API-based models and randomness in text generation — the scores for individual instances may slightly differ between evaluation runs. However, our evaluation framework ensures that the average performance across the entire TUNA-CAP dataset remains robust and statistically significant.


### TUNA-MCQ (Video Multi-Choice QA)

To evaluate TUNA-MCQ, simply run:

```bash
cd VLMEvalKit
bash scripts_tuna/eval_tuna_mcq.sh
```

## Acknowledgments

The code is largely based on the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank the authors for their great work.

## 📋 Citation

If you find our work helpful, feel free to give us a cite.

```
@article{kong2025tuna,
  title={TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos},
  author={Kong, Fanheng and Zhang, Jingyuan and Zhang, Hongzhi and Feng, Shi and Wang, Daling and Yu, Linhao and Ji, Xingguang and Tian, Yu and W., Victoria and Zhang, Fuzheng},
  journal={arXiv preprint arXiv:2505.20124},
  year={2025}
}
```