# Gemma 3 + LoRA results (written automatically 2026-10-19 22:22 PDT)

LoRA rank 16, 1 epoch, adapter: `/scratch/wuhungma/lora/r16_bf16_ep1/adapter`.
Same prompt, clips and 0.5 threshold as zero-shot Gemma. Slurm jobs: train 65960416, val 65960417, test 65967998.

## Job status
```
        65960416       lora_r16    COMPLETED   08:11:53 
      65960417_0    gemma_probs    COMPLETED   00:15:33 
      65960417_1    gemma_probs    COMPLETED   00:15:33 
      65960417_2    gemma_probs    COMPLETED   00:15:33 
      65960417_3    gemma_probs    COMPLETED   00:15:33 
      65967998_0    gemma_probs    COMPLETED   00:17:53 
      65967998_1    gemma_probs    COMPLETED   00:17:55 
      65967998_2    gemma_probs    COMPLETED   00:17:51 
      65967998_3    gemma_probs    COMPLETED   00:17:50 
```
done: 370 steps, adapter saved to /scratch/wuhungma/lora/r16_bf16_ep1/adapter

## val: LoRA
Rows: 555, errors: 0
```
split=val  threshold=0.5
dataset  method             n  bal_acc  recall    prec      F1  AUROC
edf      single camera     78    91.0%   90.9%   80.0%   85.1%  0.968
edf      camera vote       41    93.8%   90.9%   90.9%   90.9%  0.970
mcfd     single camera    193    93.2%   88.9%   95.5%   92.1%  0.988
mcfd     camera vote       26   100.0%  100.0%  100.0%  100.0%  1.000
up_fall  single camera    284    92.6%   95.2%   84.6%   89.6%  0.976
up_fall  camera vote      142    92.1%   94.2%   84.5%   89.1%  0.982
```

## test: LoRA
Rows: 694, errors: 0
```
split=test  threshold=0.5
dataset  method             n  bal_acc  recall    prec      F1  AUROC
edf      single camera     74    95.5%  100.0%   78.3%   87.8%  0.964
edf      camera vote       39    96.7%  100.0%   81.8%   90.0%  0.993
mcfd     single camera    200    92.5%   94.4%   85.0%   89.5%  0.957
mcfd     camera vote       25    93.8%  100.0%   81.8%   90.0%  1.000
up_fall  single camera    420    94.4%   96.4%   89.5%   92.8%  0.991
up_fall  camera vote      210    96.4%  100.0%   90.3%   94.9%  0.991
```

## For comparison: zero-shot Gemma (same clips)
```
split=val  threshold=0.5
dataset  method             n  bal_acc  recall    prec      F1  AUROC
edf      single camera     78    75.1%   59.1%   72.2%   65.0%  0.872
edf      camera vote       41    78.5%   63.6%   77.8%   70.0%  0.892
mcfd     single camera    193    68.5%   75.0%   54.0%   62.8%  0.793
mcfd     camera vote       26    74.2%   77.8%   58.3%   66.7%  0.810
up_fall  single camera    284    59.3%   19.2%   95.2%   32.0%  0.804
up_fall  camera vote      142    52.9%    5.8%  100.0%   10.9%  0.863

split=test  threshold=0.5
dataset  method             n  bal_acc  recall    prec      F1  AUROC
edf      single camera     74    79.8%   66.7%   75.0%   70.6%  0.860
edf      camera vote       39    87.2%   77.8%   87.5%   82.4%  0.913
mcfd     single camera    200    73.0%   81.9%   56.2%   66.7%  0.817
mcfd     camera vote       25    78.8%   88.9%   61.5%   72.7%  0.903
up_fall  single camera    420    62.3%   25.0%   97.7%   39.8%  0.844
up_fall  camera vote      210    57.7%   15.5%  100.0%   26.8%  0.943
```
