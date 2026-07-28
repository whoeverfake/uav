@echo off
cd /d c:\Users\whoever\Desktop\hust\sci\uav\proposed
set "PYTHONHOME="
set "PYTHONPATH=c:\Users\whoever\Desktop\hust\sci\uav\proposed"
..\.venv\Scripts\python.exe -u train\train.py > train_full.log 2>&1
echo TRAINING_DONE EXITCODE=%errorlevel% >> train_full.log
