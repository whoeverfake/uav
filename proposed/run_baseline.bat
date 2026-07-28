@echo off
REM Bahdanau-attention critic baseline. Identical to run_train.bat except
REM --critic_attn bahdanau, so the reward curves isolate the attention change.
cd /d c:\Users\whoever\Desktop\hust\sci\uav\proposed
set "PYTHONHOME="
set "PYTHONPATH=c:\Users\whoever\Desktop\hust\sci\uav\proposed"
..\.venv\Scripts\python.exe -u train\train.py --critic_attn bahdanau --experiment_name bahdanau_baseline > train_bahdanau.log 2>&1
echo TRAINING_DONE EXITCODE=%errorlevel% >> train_bahdanau.log
