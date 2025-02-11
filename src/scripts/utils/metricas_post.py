import json, os, glob
from pathlib import Path
import pandas as pd
from sklearn.metrics import matthews_corrcoef

mipath=''

with open('../../../hf/hf.txt', 'r') as archivo:
    for linea in archivo:
        linea = linea.strip()
        if linea.startswith('mipath='):
            mipath = linea.split('=')[1]
            break
            
idiomas = ['es', 'en']
steps = ['train', 'test', 'dev']

def calcular_media_accuracy(json_path):
    # Leer el archivo JSON
    with open(json_path, 'r') as file:
        data = json.load(file)

    accuracies = []
    for idioma in idiomas:
        for step in steps:
            accuracies.append(data[f"{idioma}_{step}"]["eval_accuracy"])

    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f'All: {mean_accuracy}')
    data["mean_accuracy"] = mean_accuracy


    for idioma in ['es', 'en']:
        accuracies = []
        for step in steps:
            accuracies.append(data[f"{idioma}_{step}"]["eval_accuracy"])
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f'{idioma}: {mean_accuracy}')
        data[f"mean_accuracy_{idioma}"] = mean_accuracy

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def matthews_metric(labels, predictions):
    return matthews_corrcoef(labels, predictions)

def sacar_preds_labels(path):
    predicciones = []
    labels = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            predicciones.append(data['prediction'])
            labels.append(data['correct_option'])
    return labels, predicciones


def calcular_metricas(initP):
    # Usamos pathlib para recorrer todos los archivos .jsonl
    for archivo_jsonl in Path(initP).glob('**/*.jsonl'):
        labels, predicciones = sacar_preds_labels(archivo_jsonl)

        folder_path = archivo_jsonl.parent
        file_name = archivo_jsonl.name

        metrics_file = folder_path / "metrics.json"

        lang, prompt = file_name.split("_")[2], file_name.split("_")[-1].split(".")[0]
        prompt = "_dev" if lang == prompt else "_dev_pen"

        print(metrics_file)

        with open(metrics_file, 'r+', encoding='utf-8') as file:
            print(file)
            metricas = json.load(file)
            print(metricas)

            metricas[f'es{prompt}']['matthews'] = matthews_metric(labels, predicciones)
            #
            file.seek(0)  # Asegurarse de sobrescribir desde el principio
            json.dump(metricas, file, indent=4)
            file.truncate()  # Eliminar cualquier contenido residual si la nueva json es más pequeña





# # Reemplaza 'ruta/al/archivo.json' con el path del archivo JSON
# calcular_media_accuracy(f'{mipath}/evaluador/src/output_p1/metrics.json')
# # Reemplaza 'ruta/al/archivo.json' con el path del archivo JSON
# calcular_media_accuracy(f'{mipath}/evaluador/src/output_p2/metrics.json')
if __name__=='__main__':
    fa_path = f"{mipath}/evaluador/data/FA_O/outputFA/data_debut_v2"
    calcular_metricas(fa_path)
    # initPath = Path(f'{mipath}/evaluador/data/OUT_CASIMEDICOS')
    #
    # for archivo_json in list(initPath.glob('**/metrics.json')):
    #     print(archivo_json)
    #     calcular_media_accuracy(archivo_json)

