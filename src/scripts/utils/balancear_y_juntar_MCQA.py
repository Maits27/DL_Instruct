import json
import numpy as np
import os, glob

'''
Script to generate JSONL files for the CasiMedicos-balanced dataset
'''
mipath=''

with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('mipath='):
            mipath = linea.split('=')[1]
            break
# LANG = 'es'
# SETS_PATH = [
#     f'../data/casiMedicos/JSONL/{LANG}.train_casimedicos.jsonl',
#     f'../data/casiMedicos/JSONL/{LANG}.test_casimedicos.jsonl',
#     f'../data/casiMedicos/JSONL/{LANG}.dev_casimedicos.jsonl'
# ]
def juntar_jsons(path, steps=['dev']):
    for step in steps:
        with open(f'{path}/{step}.jsonl', 'w', encoding='utf-8') as outfile:
            for file in glob.glob(os.path.join(f'{path}/', f'{step}*.jsonl')):
                print(file)
                with open(file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
    print(f'Done: {path}')

def balance_answers(path, steps=['dev']):
    np.random.seed(42) # Seed for reproducibility
    index_mapping ={1:"opa", 2:"opb", 3:"opc", 4:"opd"}
    files_to_convert = []
    for step in steps:
        files_to_convert += glob.glob(os.path.join(f'{path}/original/', f'{step}*.jsonl'))

        for set_path in files_to_convert:
            instances = []
            with open(set_path, 'r', encoding='utf-8') as file:
                instances = [json.loads(row) for row in file]

            new_file_path = f'{path}/balanced/{os.path.basename(set_path)}'

            with open(new_file_path, 'w', encoding='utf-8') as file:

                for instance in instances:
                    id = instance['id']
                    type = instance['choice_type']
                    subject_name = instance['subject_name']
                    topic_name = instance['topic_name']
                    question = instance['question']
                    correct_option_id = instance['cop']
                    correct_option = instance[index_mapping[correct_option_id]]
                    incorrect_options = [instance[index_mapping[index]] for index in range(1,5) if int(index) != correct_option_id]

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-1',
                        'choice_type': type,
                        'subject_name': subject_name,
                        'topic_name': topic_name,
                        'question': question,
                        "opa": correct_option,
                        "opb": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                        "opc": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                        "opd": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                        'cop': 1
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-2',
                        'choice_type': type,
                        'subject_name': subject_name,
                        'topic_name': topic_name,
                        'question': question,
                        "opa": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                        "opb": correct_option,
                        "opc": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                        "opd": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                        'cop': 2
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-3',
                        'choice_type': type,
                        'subject_name': subject_name,
                        'topic_name': topic_name,
                        'question': question,
                        "opa": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                        "opb": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                        "opc": correct_option,
                        "opd": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                        'cop': 3
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-4',
                        'choice_type': type,
                        'subject_name': subject_name,
                        'topic_name': topic_name,
                        'question': question,
                        "opa": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                        "opb": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                        "opc": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                        "opd": correct_option,
                        'cop': 4
                    }, ensure_ascii=False) + '\n')

                    

            print(f"File {new_file_path} created.")

if __name__=='__main__':
    path = f'{mipath}/evaluador/data/mcqa'
    balance_answers(path)
    # juntar_jsons(f'{path}/original', langs)
    # juntar_jsons(f'{path}/balanced', langs)
