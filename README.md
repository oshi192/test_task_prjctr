## Test task for 'Machine Learning in Production' course.

Coding task:
 - Step 1:
   - path to python script for training:\
     [/test_nlp/train_model.py](https://github.com/oshi192/test_task_prjctr/blob/main/test_nlp/train_model.py)
   - trained model with metrics:
     [/test_nlp/data/test_model.h5]()
   - Instruction how to reproduce:
     - install neccessary libraries:
       ```antlrv4
        pip3 install -r requirements.txt
       ```
     - execute script for training and saving the model:
       ```antlrv4
       python3 ./test_nlp/train_model.py
       ```
       model will be saved to file _/test_nlp/data/test_model.h5_ \
       and tokenizer to _/test_nlp/data/tokenizer.pickle_
- Step 2:
   - path to python script for training: \
     [main.py](https://github.com/oshi192/test_task_prjctr/blob/main/main.py)
   - Instruction how to reproduce:
     - first install neccessary libraries:
       ```antlrv4
        pip3 install -r requirements.txt
       ```
     - second start the server with uvicorn:
       ```antlrv4
       uvicorn main:app --reload
       ```
       it will start api server on localhost with single endpoint
     - go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the endpoint
