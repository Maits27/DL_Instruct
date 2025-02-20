import json, os, copy, csv, random
import transformers, datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
import evaluate
import pandas as pd
from huggingface_hub import login

token = ''

with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('token='):
            token = linea.split('=')[1]
            break

login(token=token)
noInstModels = {"meta-llama/Llama-3.1-8B", "google/gemma-2-9b",
                "mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.1-70B"}

with open(f'../configuration/paramsFA.json', 'r', encoding='utf-8') as f:
    params = json.load(f)


def get_file_id(prompt):
    # print(f'\n\n{prompt}\n\n{params["prompts"][params["language"]]["Q"]}\n\n')
    prompt_lang = 'en' if params['prompt_solo_en'] else params["language"]
    id_comp = prompt.split(f'{params["prompts"][prompt_lang]["Q"]}:')[1].split('\n')[0]
    return id_comp


def create_messages():
    messages_list = []
    answer_list = {}
    with open(params['train_path'], 'r', encoding='utf-8') as file:
        lines = pd.read_csv(file)

        for index, question in lines.iterrows():
            prompt_lang = 'en' if params['prompt_solo_en'] else params["language"]

            answer_list[str(question["ID"])] = question['recurrencia_FA_1']
            messages = copy.deepcopy(params['messages'][f'{params["model"]}'])

            messages[-1]['content'] = messages[-1]['content'].replace(
                '<Q>', params["prompts"][prompt_lang]["Q"])

            if params['model'] == "meta-llama/Meta-Llama-3.1-8B-Instruct" or params['model'] == "mistralai/Mistral-7B-Instruct-v0.3":
                messages[0]['content'] = params["prompts"][prompt_lang][params["prompt"]]
            else:
                messages[-1][
                    'content'] = f'{params["prompts"][prompt_lang][params["prompt"]]}{messages[-1]["content"]}'

            messages[-1]['content'] = messages[-1]['content'].replace(
                '<PATIENT_CASE>', question['informes'])
            messages[-1]['content'] = messages[-1]['content'].replace('<ID>', str(question['ID']))

            messages_list.append(messages)
    random.seed(42)
    random.shuffle(messages_list)
    return messages_list, answer_list


def predict_instances(model_id, messages, kwargs):
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,  # Enable 8-bit quantization
    #     llm_int8_threshold=6.0,  # Threshold for enabling 8-bit optimization
    #     llm_int8_skip_modules=None  # Skip layers if needed
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     quantization_config=quantization_config,  # Enables 8-bit quantization
    #     device_map="auto",  # Automatically maps layers to available devices
    #     torch_dtype=torch.bfloat16
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', token=token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16, "temperature": kwargs['temperature']},
        device_map="auto",
        token=token
    )
    if model_id in noInstModels:
        messages = ["\n".join(d['content'] for d in fila) for fila in messages]

    outputs = pipeline(
        messages,
        max_new_tokens=kwargs['max_new_tokens'],
        batch_size=params['batch_size'])
    return outputs


def inferencia():
    messages, labels = create_messages()
    print(labels)
    prediction = predict_instances(params['model'], messages, params['params'])
    label_list, prediction_list = [], []
    prompt_lang = 'en' if params['prompt_solo_en'] else params["language"]
    with open(f'{params["output_dir"]}/predictions_{params["document_partition"]}_{params["language"]}_prompt_{prompt_lang}.jsonl', 'w', encoding='utf-8') as file:
        for entry in prediction:
            output_messages = entry[0]['generated_text']
            if params['model'] in noInstModels:
                f_id= get_file_id(output_messages)
            else:
                f_id = get_file_id(output_messages[-2]['content'])

            label = labels[f'{f_id}']
            label_list.append(label)
            try:
                if params['model'] in noInstModels:
                    prediction_list.append(int(output_messages.replace(' ', '').replace('\n', '')[-1]))
                    output_messages = output_messages.split(f_id)[0]
                else:
                    prediction_list.append(int(output_messages[-1]['content'].replace(' ', '').replace('\n', '')[-1]))
                    output_messages[-1]['content'] = output_messages[-1]['content'].split(f_id)[0]

            except (ValueError, IndexError):
                prediction_list.append(0)
            p ={
                "id": f_id,
                "correct_option": label,
                "prediction": prediction_list[-1],
                "messages": output_messages}

            file.write(json.dumps(p, ensure_ascii=False) + '\n')
    evaluate_model_mio(label_list, prediction_list)


def evaluate_model_mio(labels, predictions):
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

    if not os.path.isfile(f"{params['output_dir']}/metrics.json"):
        with open(f"{params['output_dir']}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump({"Tarea": params['tarea']}, f, ensure_ascii=False, indent=4)

    with open(f"{params['output_dir']}/metrics.json", 'r+', encoding='utf-8') as f:
        try:
            metricas = json.load(f)
        except json.JSONDecodeError:
            metricas = {}

        prompt_lang = '_pen' if params['prompt_solo_en'] else ''
        metricas[f"{params['language']}_{params['document_partition']}{prompt_lang}"] = m
        f.seek(0)
        json.dump(metricas, f, ensure_ascii=False, indent=4)
        f.truncate()

def evaluate_model(labels, predictions):
    accuracy = evaluate.load('accuracy')  # Load the accuracy function
    f1 = evaluate.load('f1')  # Load the f-score function
    precision = evaluate.load('precision')  # Load the precision function
    recall = evaluate.load('recall')  # Load the recall function

    m = {
        'eval_micro_f1': f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
        'eval_macro_f1': f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
        'eval_micro_precision': precision.compute(predictions=predictions, references=labels, average='micro')['precision'],
        'eval_macro_precision': precision.compute(predictions=predictions, references=labels, average='macro')['precision'],
        'eval_micro_recall': recall.compute(predictions=predictions, references=labels, average='micro')['recall'],
        'eval_macro_recall': recall.compute(predictions=predictions, references=labels, average='macro')['recall'],
        'eval_accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy']
    }

    if not os.path.isfile(f"{params['output_dir']}/metrics.json"):
        with open(f"{params['output_dir']}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump({"Tarea": params['tarea']}, f, ensure_ascii=False, indent=4)

    with open(f"{params['output_dir']}/metrics.json", 'r+', encoding='utf-8') as f:
        try:
            metricas = json.load(f)
        except json.JSONDecodeError:
            metricas = {}

        metricas[f"{params['language']}_{params['document_partition']}"] = m
        f.seek(0)
        json.dump(metricas, f, ensure_ascii=False, indent=4)
        f.truncate()

if __name__ == '__main__':

    output = f"{params['output_dir']}/predictions_{params['language']}_{params['document_partition']}.jsonl"

    if not os.path.isdir(params['output_dir']):
        os.makedirs(params['output_dir'])

    if os.path.isfile(output):
        print(f"{output} already exists")
    else:
        inferencia()
