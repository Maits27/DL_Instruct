import json, os, copy, random
import transformers, datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
import evaluate
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
sysModels = {'HPAI-BSC', 'mistralai', 'meta-llama'}
with open(f'../configuration/params.json', 'r', encoding='utf-8') as f:
    params = json.load(f)


def get_file_id(prompt):
    prompt_lang = 'en' if params['prompt_solo_en'] else params["language"]
    id_comp = prompt.split(f'{params["prompts"][prompt_lang]["Q"]}:')[1].split('\n')[0]
    return id_comp.split('_')


def create_messages():
    messages_list = []
    answer_list = {}
    with open(params['train_path'], 'r', encoding='utf-8') as file:
        lines = [json.loads(row) for row in file]
        for question in lines:
            prompt_lang = 'en' if params['prompt_solo_en'] else params["language"]
            answer_list[f'{question["id"]}_{question["question_id_specific"]}'] = question['correct_option']
            messages = copy.deepcopy(params['messages'][f'{params["model"]}'])

            messages[-2]['content'] = messages[-2]['content'].replace(
                '<Q>', params["prompts"][prompt_lang]["Q"]).replace(
                '<QT>', params["prompts"][prompt_lang]["QT"]).replace(
                '<C>', params["prompts"][prompt_lang]["C"])
            messages[-1]['content'] = messages[-1]['content'].replace(
                '<ANS>', params["prompts"][prompt_lang]["ANS"])

            if params['model'].split('/')[0] in sysModels and params['model'].split('/')[0] not in noInstModels:
                messages[0]['content'] = params["prompts"][prompt_lang][params["prompt"]]
            else:
                messages[-2]['content'] = f'{params["prompts"][prompt_lang][params["prompt"]]}{messages[-2]["content"]}'

            messages[-2]['content'] = messages[-2]['content'].replace('<TYPE>', question['type']).replace(
                '<FULL_QUESTION>', question['full_question'])
            messages[-2]['content'] = messages[-2]['content'].replace('<ID>', str(question['id'])).replace(
                '<ID_SPECIFIC>', str(question['question_id_specific']))

            for i in range(1, 6):
                answer_i = 'null' if question['options'][f'{i}'] is None else question['options'][f'{i}']
                messages[-2]['content'] = messages[-2]['content'].replace(f'<O{i}>', answer_i)

            messages_list.append(messages)

    random.seed(42)
    random.shuffle(messages_list)

    return messages_list, answer_list


def predict_instances(model_id, messages, kwargs):
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
        batch_size=16)
    return outputs


def inferencia():
    messages, labels = create_messages()
    prediction = predict_instances(params['model'], messages, params['params'])
    label_list, prediction_list = [], []
    prompt_lang = 'en' if params['prompt_solo_en'] else params["language"]

    name = f'{params["output_dir"]}/predictions_{params["document_partition"]}_{params["language"]}.jsonl'
    if params["language"] != prompt_lang:
        name = f'{params["output_dir"]}/predictions_{params["document_partition"]}_{params["language"]}_prompt_{prompt_lang}.jsonl'

    with open(name, 'w', encoding='utf-8') as file:
        for entry in prediction:
            output_messages = entry[0]['generated_text']
            if params['model'] in noInstModels:
                f_id, q_id = get_file_id(output_messages)
            else:
                f_id, q_id = get_file_id(output_messages[-2]['content'])

            label = labels[f'{f_id}_{q_id}']
            label_list.append(label)
            try:
                if params['model'] in noInstModels:
                    prediction_list.append(int(output_messages.split(f'{params["prompts"][prompt_lang]["ANS"]}: ')[-1].replace(' ', '')[0]))
                else:
                    prediction_list.append(int(output_messages[-1]['content'].split(f'{params["prompts"][prompt_lang]["ANS"]}: ')[-1].replace(' ', '')[0]))
            except (ValueError, IndexError):
                prediction_list.append(0)
            p ={
                "id": f_id,
                "question_id_specific": q_id,
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
    # matthew = matthews_corrcoef(labels, predictions)
    # Diccionario con las métricas
    m = {
        'eval_micro_f1': eval_micro_f1,
        'eval_macro_f1': eval_macro_f1,
        'eval_micro_precision': eval_micro_precision,
        'eval_macro_precision': eval_macro_precision,
        'eval_micro_recall': eval_micro_recall,
        'eval_macro_recall': eval_macro_recall,
        'eval_accuracy': eval_accuracy,
        # "matthews": matthew
    }

    if not os.path.isfile(f"{params['output_dir']}/metrics.json"):
        with open(f"{params['output_dir']}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump({"Tarea": "casimedicos"}, f, ensure_ascii=False, indent=4)

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
            json.dump({"Tarea": f"{params['tarea']}"}, f, ensure_ascii=False, indent=4)

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
        if not (params["prompt_solo_en"]==True and params['language']=="en"):
            inferencia()
