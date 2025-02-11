import json
from pathlib import Path
import openpyxl
import pandas as pd

mipath=''

with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('mipath='):
            mipath = linea.split('=')[1]
            break
def json_to_dataframe(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        return pd.json_normalize(data)

def combinarJSON(init, langs=['es', 'en', 'it', 'fr'], steps=['train', 'test', 'dev'], ex=''):
    datos_combinados = []
    initPath = Path(init)

    for params_json_path in list(initPath.glob(f'**/params{ex}.json')):
        results_json_path = (params_json_path.parent / 'metrics.json')

        with open(params_json_path, 'r', encoding='utf-8') as file:
            params = json.load(file)

        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)


        for lang in langs:
            for prompt_l in ["_pen", ""]:
                accuracies, recalls, f1s, precisions, matthews = [], [], [], [], []
                if not (lang == 'en' and prompt_l == '_pen'):
                    for step in steps:
                        accuracies.append(data[f"{lang}_{step}{prompt_l}"]["eval_accuracy"])
                        recalls.append(data[f"{lang}_{step}{prompt_l}"]["eval_macro_recall"])
                        precisions.append(data[f"{lang}_{step}{prompt_l}"]["eval_macro_precision"])
                        f1s.append(data[f"{lang}_{step}{prompt_l}"]["eval_macro_f1"])
                        if "matthews" in data[f"{lang}_{step}{prompt_l}"]:
                            matthews.append(data[f"{lang}_{step}{prompt_l}"]["matthews"])

                    p_l = 'en' if prompt_l != '' else lang
                    print(params['train_path'])
                    datos_json = {
                        "datos": params['train_path'].split('/')[-2] if ex!="FA" else ex,
                        "model": params['model'],
                        "prompt": f"{params['prompt']} {p_l}",
                        "lang": lang,
                        "accuracy": sum(accuracies) / len(accuracies),
                        "recall": sum(recalls) / len(recalls),
                        "precision": sum(precisions) / len(precisions),
                        "f1": sum(f1s) / len(f1s),
                        "matthews": None if len(matthews) == 0 else sum(matthews) / len(matthews)
                    }

                    datos_combinados.append(datos_json)

    with open(f'{initPath}/todasLasMetricas.json', 'w', encoding='utf-8') as file:
        json.dump(datos_combinados, file, indent=2, ensure_ascii=False)

    return json_to_dataframe(f'{initPath}/todasLasMetricas.json')


def sacarCSV(initPath, output_csv, output_excel, langs=['es', 'en', 'it', 'fr'], steps=['train', 'test', 'dev'], ex=''):
    all_dataframes = combinarJSON(initPath, langs=langs, steps=steps, ex=ex)
    all_dataframes.to_csv(f'{initPath}/{output_csv}', index=False)
    csv_a_excel(initPath, output_csv, output_excel)

def csv_a_excel(initPath, output_csv, output_excel):
    # Lee el archivo CSV
    df = pd.read_csv(f'{initPath}/{output_csv}')

    # Guarda el DataFrame en un archivo de Excel
    df.to_excel(f'{initPath}/{output_excel}', index=False)


if __name__ == "__main__":
    nombreFA = "output_debutV2_shuffle42"
    nombre = "output_original_shuffle42"
    nombreMCQA = "output_MCQA_shuffle42NoIt"
    # sacarCSV(f'{mipath}/evaluador/data/OUT_CASIMEDICOS_balanced', f'{nombre}.csv', f'{nombre}.xlsx')
    sacarCSV(f'{mipath}/evaluador/data/CASIMEDICOS_O', f'{nombre}.csv',
             f'{nombre}.xlsx', ['es', 'en'], ['dev'])
    # sacarCSV(f'{mipath}/evaluador/data/FA_O/NOINSTRUCT/data_debut_v2', f'{nombreFA}.csv',f'{nombreFA}.xlsx', ['es'], ['dev'], ex='FA')
    # sacarCSV(f'{mipath}/evaluador/data/MCQA_O/NOINSTRUCT', f'{nombreMCQA}.csv',
    #          f'{nombreMCQA}.xlsx', ['en'], ['dev'], ex='_MCQA')