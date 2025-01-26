import json
import logging

#import wandb
from pathlib import Path
import os
from datasets import load_dataset, ClassLabel
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support, classification_report
from transformers import Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoModelForMultipleChoice, AutoConfig, PreTrainedTokenizerFast, DataCollatorWithPadding, set_seed, \
    default_data_collator
import numpy as np

token = ''

with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('token='):
            token = linea.split('=')[1]
            break


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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # Cargamos la configuración del modelo:
    config = AutoConfig.from_pretrained(
        script_params['config_name'] if script_params['config_name'] else script_params['model_path'],
        num_labels=5,
        cache_dir=script_params['cache_dir'],
        token=token
    )
    # Cargamos el modelo:
    model = AutoModelForSequenceClassification.from_pretrained(
        script_params['model_path'],
        config=config,
        cache_dir=script_params['cache_dir'],
        token=token,
        # device_map="auto"
    )
    print('tokenizer')

    # Cargamos el tokenizador:
    tokenizer = AutoTokenizer.from_pretrained(
        script_params['tokenizer_name'] if script_params['tokenizer_name'] else script_params['model_path'],
        cache_dir=script_params['cache_dir'],
        use_fast=True,
        config=config,
        add_prefix_space=True,
        label_column_name="correct_option",
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )
    print('tokenize')

    # Tokenizar hypothesis y premisa
    def tokenizar_textos(textos):
        args = (textos['text'], textos['options'])
        tokenized = tokenizer(
            *args,
            padding='max_length',
            truncation= True, #"only_first",
            return_tensors='pt',
            max_length= 256#512 #256
        )
        # tokenized["labels"] = textos["correct_option"]
        return tokenized

    print(f"overwrite cache: {script_params['overwrite_cache']}")
    tokenized_datasets = datasets.map(
        tokenizar_textos,
        batched=True,
        # num_proc=script_params['preprocessing_num_workers'],
        load_from_cache_file=not script_params['overwrite_cache']
    )
    print(tokenized_datasets["train"][0])
    # print('collator')
    # # Data collator TODO no se si es necesario en mi caso
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
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


    training_args = TrainingArguments(
        output_dir=script_params['output_dir'],
        num_train_epochs=script_params['num_train_epochs'],  # número de épocas de entrenamiento
        per_device_train_batch_size=script_params['per_device_train_batch_size'],
        per_device_eval_batch_size=script_params['per_device_eval_batch_size'],
        gradient_accumulation_steps=script_params['gradient_accumulation_steps'],
        learning_rate=script_params['learning_rate'],
        seed=script_params['seed'],
        weight_decay=script_params['weight_decay'],
        warmup_ratio=script_params['warmup_ratio'],
        evaluation_strategy=script_params['evaluation_strategy'],
        save_strategy=script_params['save_strategy'],
        save_total_limit=2,
        metric_for_best_model=script_params['metric_for_best_model'],
        #report_to=script_params['report_to'],
        run_name=script_params['run_name'],
        load_best_model_at_end=script_params['load_best_model_at_end'],
    )
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!! TRAINING ARGUMENTS !!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(training_args)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if script_params['do_train'] else None,
        eval_dataset=tokenized_datasets["dev"] if script_params['do_eval'] else None,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        # data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[]
    )

    def predict_and_save(prediction_dataset: str, output_file: str, metric_key_prefix: str = 'eval'):
        if trainer.is_world_process_zero():
            prediction_dataset = tokenized_datasets[prediction_dataset]
            output_predictions_file = os.path.join(script_params['output_dir'], output_file)

            prediction_results = trainer.evaluate(prediction_dataset)

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
