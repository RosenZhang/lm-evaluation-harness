#!/bin/bash
mkdir -p test_output

model_values=("bigscience/mt0-base", "bigscience/mt0-small", "bigscience/mt0-large", "bigscience/mt0-xl", "bigscience/mt0-xxl")
config_values=("malayalam_mixsentiment" "sentimix_spaeng" "tamil_mixsentiment")
prompt_values=(1 2 3 4 5)
numshot_values=(0 1 3 5)

for model_arg in "${model_values[@]}"; do
  model_arg_short="${model_arg##*/}"
      for config_arg in "${config_values[@]}"; do
        for numshot_arg in "${numshot_values[@]}"; do
          for prompt_arg in "${prompt_values[@]}"; do
            rm -r lm_eval/tasks/codemix_nlu/__pycache__
            echo "running ${config_arg} ${model_arg_short} promptid-${prompt_arg} numshot-${numshot_arg}_results"
            python main.py \
              --model hf-seq2seq \
              --model_args "pretrained=${model_arg}" \
              --tasks "${config_arg}_p${prompt_arg}" \
              --output_path "test_output/${config_arg}_${model_arg_short}_p${prompt_arg}_${numshot_arg}shot_results" \
              --num_fewshot ${numshot_arg} \
              --device cuda:0
            rm -r lm_eval/tasks/codemix_nlu/__pycache__
          done
        done
      done
done
