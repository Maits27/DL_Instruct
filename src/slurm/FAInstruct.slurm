#!/usr/bin/env bash
#SBATCH --job-name='FA_instruct'
#SBATCH --cpus-per-task=1
#SBATCH --mem=250GB
#SBATCH --gres=gpu:3
#SBATCH --mail-user=murruela002@ikasle.ehu.eus
#SBATCH --output=./out/out_logs_FA.log
#SBATCH --error=./out/out_errors_FA.err

source ../../../hf/hf.txt
# Behar dugun igurune birtuala aktibatzen dugu
export PATH="$instructenv2":"$PATH"

SRC_PATH="$mipath/evaluador/src"
OUTPUT_PATH="$SRC_PATH/outputFA/"

# huggingface-cli logout
#declare -a pairs=(
#    "Llama meta-llama/Meta-Llama-3.1-8B-Instruct"
#    "Gemma google/gemma-2-2b-it"
#    "GemmaGrande google/gemma-2-9b-it"
#    "Mistral7B mistralai/Mistral-7B-Instruct-v0.3"
#    "BIGLlama meta-llama/Llama-3.3-70B-Instruct"
#    "LlamaNoIt meta-llama/Llama-3.1-8B"
#    "GemmaGrandeNoIt google/gemma-2-9b"
#    "MistralNoIt mistralai/Mistral-7B-v0.3"
#)
declare -a pairs=(
    "GemmaGrandeNoIt google/gemma-2-9b"
    "BIGLlamaNoIt meta-llama/Llama-3.1-70B"
)
#declare -a pairs=(
#    "BIGLlama meta-llama/Llama-3.3-70B-Instruct"
#)
# Gure python scripta deitzen dugu
#python casimedicosQAInstruct.py
for pair in "${pairs[@]}"; do
  IFS=' ' read -r n m <<< "$pair"
  for p in "2" "3"
  do
    for datos in "data_debut_v2"
    do
      for l in "es"
      do
        for s in "test"
        do
          for t in true false
          do
            json_path="$SRC_PATH/configuration/paramsFA.json"
            cp "$SRC_PATH/configuration/plantilla_FA.json" "$json_path"
            sed -i.bak "s#MI_PATH#$mipath#" $json_path
            sed -i.bak "s#FA_PATH#$fapath#" $json_path
            sed -i.bak "s#PROMPT_SOLO_ENG#$t#" $json_path
            sed -i.bak "s#CARPETA_NOMBRE#$datos#" $json_path
            sed -i.bak "s#MODEL_ID_NAME#$n#" $json_path
            sed -i.bak "s#MODEL_ID#$m#" $json_path
            sed -i.bak "s#AC_PROMPT#$p#" $json_path
            sed -i.bak "s#VAR_LANG#$l#" $json_path
            sed -i.bak "s#VAR_STEP#$s#" $json_path

            python "$SRC_PATH/scripts/batch_FAInstruct.py"
          done
        done
        cp "$json_path" "$OUTPUT_PATH/$datos/$n/output_p$p"
      done
    done
  done
done
