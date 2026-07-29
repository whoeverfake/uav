@echo off
REM Attention-RNN: Bahdanau attention + GRU for spatial-temporal modeling
cd /d c:\Users\whoever\Desktop\hust\sci\uav\proposed
set "PYTHONHOME="
set "PYTHONPATH=c:\Users\whoever\Desktop\hust\sci\uav\proposed"
..\.venv\Scripts\python.exe -u train\train.py --critic_attn attention_rnn --experiment_name attention_rnn > train_attention_rnn.log 2>&1
echo TRAINING_DONE EXITCODE=%errorlevel% >> train_attention_rnn.log
