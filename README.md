## Test task for 'Machine Learning in Production' course.

Coding task:
 - Step 1:
   - python script for training:\
     [/test_nlp/train_model.py](https://github.com/oshi192/test_task_prjctr/blob/main/test_nlp/train_model.py)
   - trained model with metrics:
     [/test_nlp/test_model.index]()
   - Instruction how to reproduce:
     - first install neccessary libraries:
       ```antlrv4
        pip3 install -r requirements.txt
       ```
     - second execute script to train and save model:
       ```antlrv4
       python3 ./test_nlp/train_model.py
       ```
- Step 2:
   - python script for training:\
     [/test_nlp/train_model.py](https://github.com/oshi192/test_task_prjctr/blob/main/test_nlp/train_model.py)
   - trained model with metrics:
     [/test_nlp/test_model.index]()
   - Instruction how to reproduce:
     - first install neccessary libraries:
       ```antlrv4
        pip3 install -r requirements.txt
       ```
     - second start the server with uvicorn:
       ```antlrv4
       uvicorn main:app --reload
       ```
