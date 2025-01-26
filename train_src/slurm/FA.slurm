#!/usr/bin/env bash
#SBATCH --job-name='FA'
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --output=./out/FAout_logs.log
#SBATCH --error=./out/FAout_errors.err

source ../../../hf/hf.txt
# Behar dugun igurune birtuala aktibatzen dugu
export PATH="$instructenv":"$PATH"

SRC_PATH="$mipath/evaluador/train_src"
OUTPUT_PATH="$SRC_PATH/outputFA/"
cleaner_script_path="$SRC_PATH/scripts/utils/experiment_cleaner.py"

# huggingface-cli logout
declare -a pairs=(
    "Llama meta-llama/Llama-3.1-8B"
    "GemmaGrande google/gemma-2-9b"
    "Mistral mistralai/Mistral-7B-v0.3"
    "BIGLlama meta-llama/Llama-3.1-70B"
)
#declare -a pairs=(
#    "BIGLlama meta-llama/Llama-3.3-70B-Instruct"
#)
# Gure python scripta deitzen dugu
#python casimedicosQAInstruct.py
for pair in "${pairs[@]}"; do
  IFS=' ' read -r n m <<< "$pair"
  for datos in "original" "balanced"
  do
    for l in "es" "en" "es_pen"
    do
      for lr in 5e-5 75e-6 25e-6
      do
        for wd in 0.01 0.001
        do
          for warm in 0.0 0.05
          do
            json_path="$SRC_PATH/configuration/params.json"
            cp "$SRC_PATH/configuration/plantilla.json" "$json_path"
            sed -i.bak "s#MI_PATH#$mipath#" $json_path
            sed -i.bak "s#VAR_LR#$lr#" $json_path
            sed -i.bak "s#MODEL_ID_NAME#$n#" $json_path
            sed -i.bak "s#CARPETA_NOMBRE#$datos#" $json_path
            sed -i.bak "s#MODEL_ID#$m#" $json_path
            sed -i.bak "s#VAR_WD#$wd#" $json_path
            sed -i.bak "s#VAR_LANG#$l#" $json_path
            sed -i.bak "s#VAR_WARM#$warm#" $json_path

            python "$SRC_PATH/scripts/train_FA.py"
          done
        done
      done
      cp "$json_path" "$OUTPUT_PATH/${datos}_${l}/$n/output_${lr}_${wd}_$warm"
      python "$cleaner_script_path" -i "$OUTPUT_PATH/${datos}_${l}/$n/output_${lr}_${wd}_$warm"
    done
  done
done
