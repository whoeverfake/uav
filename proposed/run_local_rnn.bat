@echo off
REM Local Spatial Self-Attention + RNN: original intended innovation
cd /d c:\Users\whoever\Desktop\hust\sci\uav\proposed
set "PYTHONHOME="
set "PYTHONPATH=c:\Users\whoever\Desktop\hust\sci\uav\proposed"
..\.venv\Scripts\python.exe -u train\train.py --critic_attn local_rnn --experiment_name local_rnn > train_local_rnn.log 2>&1
echo TRAINING_DONE EXITCODE=%errorlevel% >> train_local_rnn.log
