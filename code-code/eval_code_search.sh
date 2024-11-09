code2codesearch_path=code-code

declare -a langs=("C++" "Java" "Python" "C#" "Javascript" "PHP" "C")
declare -a models=("codebert" "roberta")
declare -a datatypes=("snippet" "program")

function run() {
for datatype in "${datatypes[@]}"; do
    echo datatype
    data_folder=${data_path}${datatype}_level/
    for model in "${models[@]}"; do
        exp_name=${datatype}
        echo exp_name
        for lang1 in "${langs[@]}"; do
            echo $lang1
            python3 ${evaluator_path}evaluator.py -a ${data_folder}$lang1/test.jsonl  -p ${data_folder}$lang1/predictions.jsonl 
        done
    done
done
}

exp=code2code
exp_path=${code2codesearch_path}
data_path=xlcost_data/retrieval/code2code_search/
evaluator_path=${exp_path}/

run;
