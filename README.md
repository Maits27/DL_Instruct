# Evaluating Instruct Models in Medical Tasks with Zero Shot Prompting

LLMs have been widely used for various tasks, showing great performance across many domains. Now, they have been refined into their instruct version, enabling them to perform tasks like text summarization, rewriting, MCQA, and reasoning. 

While they are valuable zero-shot tools, this training doesn’t guarantee improved or retained performance. This work compares base and instruct models on two medical tasks to evaluate their effectiveness and identify the best-performing instruct model.

## Research Questions

* **RQ1**: Which instruct model performs better in medical Question Answering and in Atrial Fibrillation Recurrence Prediction tasks?
* **RQ2**: How do instruct models perform using zero-shot prompting compared to their base version in medical tasks?

## Evaluated tasks 

### Multiple Choice QA

Evaluation of LLMs on clinical case-based multiple-choice  questions regarding possible diagnoses or treatments. Along with the original dataset, a balanced version is also used, changing answer positions to reduce possible bias.

This task will be tested using CasiMedicos and MedMCQA datasets (files avaliable in [data folder](data): [casimedicos](data/casimedicos) and [mcqa](data/mcqa))

### Atrial Fibrilation Recurrence Detection

Using only the debut report —containing the patient's first atrial fibrillation occurrence, blood analysis, and other data— the model predicts whether the patient will experience another AF episode.  This is notably more challenging than MCQA, since doctors typically rely on additional clinical records that are unavailable to the model.

## Models used
Four different models will be tested in each task:

1. [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
2. [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
3. [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)
4. [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## Hyperparameters

The parameters used for all the models in common are:

| Temperature   | Max New Tokens | Batch size | Prompt Language |
|---------------|----------------|------------|-----------------|
| 0             | 1              | 16         | English         |
| 0             | 1              | 16         | Content language|

All the specific parameters used for each model can be seen in the following folders:

[Configuration folder](src/configuration) shows the different templates used for each task and in [slurm folder](src/slurm) the tested parameters for each model.

The specific prompts for each task are avaliable in the configuration template of each task:

* MCQA CasiMedicos: [plantilla.json](src/configuration/plantilla.json)
* MCQA MedMCQA: [plantilla_MCQA.json](src/configuration/plantilla_MCQA.json)
* AFRP: [plantilla_FA.json](src/configuration/plantilla_FA.json)

