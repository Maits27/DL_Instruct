import json
import numpy as np
import os, glob

'''
Script to generate JSONL files for the CasiMedicos-balanced dataset
'''

# LANG = 'es'
# SETS_PATH = [
#     f'../data/casiMedicos/JSONL/{LANG}.train_casimedicos.jsonl',
#     f'../data/casiMedicos/JSONL/{LANG}.test_casimedicos.jsonl',
#     f'../data/casiMedicos/JSONL/{LANG}.dev_casimedicos.jsonl'
# ]
def juntar_jsons(path, langs=['es', 'en']):
    for lang in langs:
        with open(f'{path}/{lang}_completo_casimedicos.jsonl', 'w', encoding='utf-8') as outfile:
            for file in glob.glob(os.path.join(f'{path}/', f'{lang}*.jsonl')):
                print(file)
                with open(file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
    print(f'Done: {path}')

def balance_answers(path, langs=['es', 'en']):
    np.random.seed(42) # Seed for reproducibility
    files_to_convert = []
    for lang in langs:
        files_to_convert += glob.glob(os.path.join(f'{path}/original/', f'{lang}*.jsonl'))

        for set_path in files_to_convert:
            instances = []
            with open(set_path, 'r', encoding='utf-8') as file:
                instances = [json.loads(row) for row in file]

            new_file_path = f'{path}/balanced/{os.path.basename(set_path)}'

            with open(new_file_path, 'w', encoding='utf-8') as file:

                for instance in instances:
                    id = instance['id']
                    id_esp = instance['question_id_specific']
                    type = instance['type']
                    question = instance['full_question']
                    correct_option_id = instance['correct_option']
                    correct_option = instance['options'][str(correct_option_id)]
                    incorrect_options = [instance['options'][index] for index, option in instance['options'].items() if int(index) != correct_option_id]

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-1',
                        'question_id_specific': id_esp,
                        'type': type,
                        'full_question': question,
                        'options': {
                            "1": correct_option,
                            "2": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                            "3": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                            "4": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                            "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                        },
                        'correct_option': 1
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-2',
                        'question_id_specific': id_esp,
                        'type': type,
                        'full_question': question,
                        'options': {
                            "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                            "2": correct_option,
                            "3": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                            "4": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                            "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                        },
                        'correct_option': 2
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-3',
                        'question_id_specific': id_esp,
                        'type': type,
                        'full_question': question,
                        'options': {
                            "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                            "2": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                            "3": correct_option,
                            "4": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                            "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                        },
                        'correct_option': 3
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-4',
                        'question_id_specific': id_esp,
                        'type': type,
                        'full_question': question,
                        'options': {
                            "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                            "2": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                            "3": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                            "4": correct_option,
                            "5": incorrect_options[3] if isinstance(incorrect_options[3], str) else ""
                        },
                        'correct_option': 4
                    }, ensure_ascii=False) + '\n')

                    np.random.shuffle(incorrect_options)
                    file.write(json.dumps({
                        'id': f'{id}-5',
                        'question_id_specific': id_esp,
                        'type': type,
                        'full_question': question,
                        'options': {
                            "1": incorrect_options[0] if isinstance(incorrect_options[0], str) else "",
                            "2": incorrect_options[1] if isinstance(incorrect_options[1], str) else "",
                            "3": incorrect_options[2] if isinstance(incorrect_options[2], str) else "",
                            "4": incorrect_options[3] if isinstance(incorrect_options[3], str) else "",
                            "5": correct_option
                        },
                        'correct_option': 5
                    }, ensure_ascii=False) + '\n')

            print(f"File {new_file_path} created.")

if __name__=='__main__':
    path = '/evaluador/data'
    langs = ['it', 'fr']
    balance_answers(path, langs)
    # juntar_jsons(f'{path}/original', langs)
    # juntar_jsons(f'{path}/balanced', langs)
