#!/bin/bash
lms=("bert" "distilbert" "roberta" "xlnet")
max_len=512
lr=1e-5
n_epochs=10

tasks=("Amazon-Google" "Abt-Buy" "Beer" "Itunes-Amazon" "Walmart-Amazon" "Fodors-Zagats" "DBLP-ACM" 
       "DBLP-Scholar" "Company" "Dirty/DBLP-ACM" "Dirty/DBLP-Scholar" "Dirty/Walmart-Amazon" 
       "Dirty/Itunes-Amazon" "WDC/Computer-Title-XL" "WDC/Computer-Desc-XL" "WDC/Computer-Brand-XL" 
       "WDC/Computer-Spec-XL" "WDC/Computer-Title-L" "WDC/Computer-Title-M" "WDC/Computer-Title-S" 
       "WDC/Camera-Title-XL" "WDC/Camera-Title-L" "WDC/Camera-Title-M" "WDC/Camera-Title-S" 
       "WDC/Watch-Title-XL" "WDC/Watch-Title-L" "WDC/Watch-Title-M" "WDC/Watch-Title-S" 
       "WDC/Shoe-Title-XL" "WDC/Shoe-Title-L" "WDC/Shoe-Title-M" "WDC/Shoe-Title-S" "WDC/All-Title-XL" 
       "WDC/All-Title-L" "WDC/All-Title-M" "WDC/All-Title-S" "N/Amazon-Google" "N/DBLP-ACM" 
       "N/Abt-Buy" "N/Walmart-Amazon" "N/Itunes-Amazon" "DI2KG/Monitor" "DI2KG/Camera")




# The research paper recommends a batch size of 4 for the "Itunes-Amazon" task and 16 otherwise.
for lm in "${lms[@]}"; do
  for task in "${tasks[@]}"; do
    if [[ "$task" == *"Itunes-Amazon"* ]]; then
      batch_size=4
    else
      batch_size=16
    fi

    echo "================================================================================================================================"
    echo "Running task: $task with language model: $lm"
    echo "batch_size=$batch_size, max_len=$max_len, lr=$lr, n_epochs=$n_epochs, lm=$lm"
    echo "Timestamp: $(date)"
    echo "================================================================================================================================"

    CUDA_VISIBLE_DEVICES=0 python train.py \
      --task "$task" \
      --batch_size "$batch_size" \
      --max_len "$max_len" \
      --lr "$lr" \
      --n_epochs "$n_epochs" \
      --finetuning \
      --split \
      --lm "$lm"
  done
done