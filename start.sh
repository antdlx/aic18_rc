#!/bin/bash 
cd capsule-mrc81/capsuleNet-mrc/
python3 run81.py --mode 'test' --input '/search/work/input/data'
python3 run84.py --mode 'test' --input '/search/work/input/data'

cd ../../QA_Test
python3 config_cla_v646.py --mode 'test' --input '/search/work/input/data'
python3 config_cla_v60.py --mode 'test' --input '/search/work/input/data'
python3 config_ans_v20.py --mode 'test' --input '/search/work/input/data'

cd ..
python3 vote_ser_new_word.py --mode 'predict' --input 'test' --predict_file '/search/work/output/result'
