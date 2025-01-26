import trl
import torch
import peft
import json
import logging

#import wandb
from pathlib import Path
import os
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support, classification_report
from transformers import pipeline, TrainingArguments, AutoTokenizer, AutoConfig, PreTrainedTokenizerFast,  set_seed, AutoModelForCausalLM
import numpy as np
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from huggingface_hub import login

token = ''

with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('token='):
            token = linea.split('=')[1]
            break

login(token=token)

with open(f'../configuration/params.json', 'r', encoding='utf-8') as f:
    script_params = json.load(f)

# -----------------------------------------------------
# SET THE SEED
set_seed(script_params['seed'])

# -----------------------------------------------------

logger = logging.getLogger(__name__)


def main():

    if script_params['output_dir'] is not None and not os.path.exists(script_params['output_dir']) and script_params[
        'do_train'] and not script_params['overwrite_output_dir']:
        os.makedirs(script_params['output_dir'])
    else:
        raise ValueError(
            f"Output directory ({script_params['output_dir']}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Recoger ficheros entrenamiento
    data_files = {}
    if script_params['do_train']:
        if script_params['train_file'] is not None:
            data_files['train'] = script_params['train_file']
        else:
            raise ValueError('Train file error')
    print('train')
    if script_params['do_eval']:
        if script_params['dev_file'] is not None:
            data_files['dev'] = script_params['dev_file']
        else:
            raise ValueError('Dev file error')
    print('eval')
    if script_params['do_predict']:
        if script_params['test_file'] is not None:
            data_files['test'] = script_params['test_file']
        else:
            raise ValueError('Test file error')

    print('test')

    if Path(script_params['loading_script_path']).is_file():
        loading_script = script_params['loading_script_path']
    else:
        loading_script = script_params['train_file'].split(".")[-1]

    datasets = load_dataset(loading_script, data_files=data_files, cache_dir=script_params['cache_dir'])
    print('datasets')


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")

        predictions = predictions.argmax(axis=-1)
        predictions = predictions.flatten()  # Shape: (63 * 512,)
        labels = labels.flatten()
        # (Optional) Mask out padding tokens (if padding token ID is -100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        assert len(predictions) == len(labels), "Predictions and labels must have the same length after masking."

        eval_micro_f1 = f1_score(labels, predictions, average='micro')
        eval_macro_f1 = f1_score(labels, predictions, average='macro')

        eval_micro_precision = precision_score(labels, predictions, average='micro')
        eval_macro_precision = precision_score(labels, predictions, average='macro')

        eval_micro_recall = recall_score(labels, predictions, average='micro')
        eval_macro_recall = recall_score(labels, predictions, average='macro')

        eval_accuracy = accuracy_score(labels, predictions)
        matthew = matthews_corrcoef(labels, predictions)
	    # Diccionario con las métricas
        m = {
            'eval_micro_f1': eval_micro_f1,
            'eval_macro_f1': eval_macro_f1,
            'eval_micro_precision': eval_micro_precision,
            'eval_macro_precision': eval_macro_precision,
            'eval_micro_recall': eval_micro_recall,
            'eval_macro_recall': eval_macro_recall,
            'eval_accuracy': eval_accuracy,
            "matthews": matthew
        }
        return m

    config = AutoConfig.from_pretrained(
        script_params['config_name'] if script_params['config_name'] else script_params['model_path'],
        num_labels=5,
        cache_dir=script_params['cache_dir'],
        use_auth_token=True
    )
    # print('tokenizer')
    #
    # # Cargamos el tokenizador:
    tokenizer = AutoTokenizer.from_pretrained(
        script_params['tokenizer_name'] if script_params['tokenizer_name'] else script_params['model_path'],
        cache_dir=script_params['cache_dir'],
        use_fast=True,
        config=config,
        add_prefix_space=True,
        label_column_name="label"
    )
    tokenizer.pad_token = tokenizer.eos_token
    # Descargar modelo y tokenizer
    # compute_dtype =getattr(torch, )
    model = AutoModelForCausalLM.from_pretrained(
        script_params['model_path'],
        device_map='auto',
        config=config,
    )

    def tokenizar_textos(textos):
        # args = (textos['text'], textos['options'])
        tokenized = tokenizer(
            textos['text'],#*args,
            padding='max_length',
            truncation="only_first",
            return_tensors='pt',
            max_length=512  # 512 #256
        )
        # tokenized["labels"] = textos["correct_option"]
        return tokenized

    print(f"overwrite cache: {script_params['overwrite_cache']}")
    datasets = datasets.map(
        tokenizar_textos,
        batched=True,
        # num_proc=script_params['preprocessing_num_workers'],
        load_from_cache_file=not script_params['overwrite_cache']
    )
    print(datasets['train'][0])
    # datasets['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # datasets['dev'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = SFTConfig(
        max_seq_length=512,
        output_dir=script_params['output_dir'],
        overwrite_output_dir=script_params['overwrite_output_dir'],
        eval_strategy=script_params['evaluation_strategy'],
        do_train=script_params['do_train'],
        do_eval=script_params['do_eval'],
        num_train_epochs=script_params['num_train_epochs'],  # número de épocas de entrenamiento
        per_device_train_batch_size=script_params['per_device_train_batch_size'],
        per_device_eval_batch_size=script_params['per_device_eval_batch_size'],
        gradient_accumulation_steps=script_params['gradient_accumulation_steps'],
        learning_rate=script_params['learning_rate'],
        seed=script_params['seed'],
        weight_decay=script_params['weight_decay'],
        warmup_ratio=script_params['warmup_ratio'],
        save_strategy=script_params['save_strategy'],
        save_total_limit=2,
        metric_for_best_model=script_params['metric_for_best_model'],
        run_name=script_params['run_name'],
        load_best_model_at_end=script_params['load_best_model_at_end'],
        hub_token=token
    )
    # for name, module in model.named_modules():
    #     print(name)
    print(script_params["modules"][script_params["model_path"]])
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=script_params["modules"][script_params["model_path"]],
        task_type="CAUSAL_LM",
    )


    trainer = SFTTrainer(
        model,
        train_dataset=datasets['train'] if script_params['do_train'] else None,
        eval_dataset=datasets["dev"] if script_params['do_eval'] else None,
        compute_metrics=compute_metrics,
        args=training_args,
        peft_config=peft_config,
        callbacks=[]
    )
    trainer.train()


    def predict_and_save(prediction_dataset: str, output_file: str, metric_key_prefix: str = 'eval'):
        if trainer.is_world_process_zero():
            prediction_dataset = datasets[prediction_dataset]
            output_predictions_file = os.path.join(script_params['output_dir'], output_file)

            prediction_results = trainer.evaluate(prediction_dataset)
            predictions = trainer.predict(datasets[prediction_dataset])
            preds = predictions.predictions.argmax(axis=-1)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(preds.tolist(), f, ensure_ascii=False, indent=2)
            # Log evaluation
            logger.info("***** Eval results *****")
            for key, value in prediction_results.items():
                logger.info(f"  {key} = {value}")

            # Save evaluation in json
            with open(f'{output_predictions_file}_results.json', "w", encoding='utf8') as writer:
                json.dump(prediction_results, writer, ensure_ascii=False, indent=2)

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_dir = os.path.join(script_params['output_dir'], "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_dir, "w", encoding='utf8') as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    with open(f'{script_params["output_dir"]}/params.json', 'w', encoding='utf8') as archivo_json:
        json.dump(script_params, archivo_json, ensure_ascii=False, indent=4)

    results = {}
    if script_params['do_eval'] == True:
        logger.info("*** Evaluate ***")
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!EVALUATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        predict_and_save(prediction_dataset='dev', output_file='eval_predictions')

    # Predict
    if script_params['do_predict'] == True:
        logger.info("*** Predict ***")

        predict_and_save(prediction_dataset='test', output_file='test_predictions')

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
