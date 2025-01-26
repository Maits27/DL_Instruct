import json

mipath='.'
with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('mipath='):
            mipath = linea.split('=')[1]
            break

params_file_path = f"{mipath}/evaluador/src/configuration/params.json"
with open(params_file_path, 'r', encoding='utf-8') as f:
    script_params = json.load(f)

for language in ["en", "es"]:
    for step in ["train", "dev", "test"]:
        fichero_final = []
        fichero_final_2 = []
        with open(f"{mipath}/evaluador/data/casimedicos/original/{language}_{step}_casimedicos.jsonl", 'r', encoding='utf-8') as file:
            lines = [json.loads(row) for row in file]
            for line in lines:
                instancia = {
                    "text": f"{script_params['prompts'][language]['Q']}:{line['id']}_{line['question_id_specific']}\n"+
                            f"{script_params['prompts'][language]['QT']}: {line['type']}\n"+
                            f"{script_params['prompts'][language]['C']}: {line['full_question']}\n",
                    "options": f"1.- {line['options']['1']}\n" +
                            f"2.- {line['options']['2']}\n" +
                            f"3.- {line['options']['3']}\n" +
                            f"4.- {line['options']['4']}\n" +
                            f"5.- {line['options']['5']}",
                    "label": line['correct_option']
                }
                fichero_final.append(instancia)
                if language=="es":
                    instancia = {
                        "text": f"{script_params['prompts']['en']['Q']}:{line['id']}_{line['question_id_specific']}\n" +
                                f"{script_params['prompts']['en']['QT']}: {line['type']}\n" +
                                f"{script_params['prompts']['en']['C']}: {line['full_question']}\n",
                        "options": f"1.- {line['options']['1']}\n" +
                                f"2.- {line['options']['2']}\n" +
                                f"3.- {line['options']['3']}\n" +
                                f"4.- {line['options']['4']}\n" +
                                f"5.- {line['options']['5']}",
                        "label": line['correct_option']
                    }
                    fichero_final_2.append(instancia)
        with open(f"{mipath}/evaluador/data/casimedicos/original/{language}_{step}_casimedicos.json", 'w', encoding='utf-8') as file2:
            json.dump(fichero_final, file2, ensure_ascii=False, indent=4)
        if language=="es":
            with open(f"{mipath}/evaluador/data/casimedicos/original/es_pen_{step}_casimedicos.json", 'w',
                      encoding='utf-8') as file2:
                json.dump(fichero_final_2, file2, ensure_ascii=False, indent=4)



