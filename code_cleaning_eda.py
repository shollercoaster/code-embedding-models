from datasets import load_dataset, Dataset
import pandas as pd
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
from statistics import mean, median, stdev
from transformers import AutoTokenizer
import json
import subprocess

data_files = {
"train": ['python/final/jsonl/train/python_train_0.jsonl.gz',
'python/final/jsonl/train/python_train_1.jsonl.gz'
]
}

raw_dataset = load_dataset("json", data_files=data_files)

def split_dataset(raw_dataset: Dataset) -> tuple:
    '''
    Loads raw dataset object and creates training, validation and testing splits,
    and retaining only code and comment columns.
    :param s: raw dataset for language
    :returns: pandas dataframes of train, validation and test splits.
    '''
    # Convert the dataset to a pandas DataFrame for easy manipulation
    df = pd.DataFrame(raw_dataset['train'])

    # Rename the 'code' and 'docstring' columns
    df = df[['code', 'docstring']].rename(columns={'code': 'method', 'docstring': 'comment'})

    # Split the dataset into train, validation, and test sets
    train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
    test_val_df = df.drop(train_df.index)

    # Split the remaining 20% into validation and test sets (50% each of the remaining data)
    val_df = test_val_df.sample(frac=0.5, random_state=42)  # 10% for validation
    test_df = test_val_df.drop(val_df.index)  # 10% for testing

    return train_df, val_df, test_df

def is_ascii(s):
    '''
    Determines if the given string contains only ascii characters

    :param s: the string to check
    :returns: whether or not the given string contains only ascii characters
    '''
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_inline_pairs(method):
    '''
    Get all pairs of inline comments and corresponding code snippets

    :param method: the method to retrieve the pairs of comments and corresponding
    code snippets from
    :returns: all pairs of comments and corresponding code snippets
    '''
    pairs = [[]]

    comment = False
    bracket = False
    indent_lvl = -1
    lines = method.split("\n")
    for line in lines:
        if "//" in line and not bracket and not "://" in line:
            pairs[-1].append(line)
            if '\t' in line:
                indent_lvl = line.count('\t')
            else:
                indent_lvl = line.split("//")[0].count(' ')
            comment = True
            bracket = False
        elif comment:
            if '{' in line and not bracket:
                bracket = True
                pairs[-1].append(line)
            elif '}' in line:
                line_indent = -1
                if '\t' in line:
                    line_indent = line.count('\t')
                else:
                    line_indent = line.split("//")[0].count(' ')
                if indent_lvl == line_indent:
                    pairs[-1].append(line)
                if not bracket:
                    pairs.append([])
                    comment = False
                    bracket = False
            elif line.isspace() or line == '' and not bracket:
                pairs.append([])
                comment = False
            else:
                pairs[-1].append(line)

    # Convert pairs into proper format of (code snippet, inline comment) dataframe
    code_snippets   = []
    comments        = []
    for pair in pairs:
        if pair and len(pair) < 5:
            code    = []
            comment = []
            skip = False
            for line in pair:
                if "TODO" in line: break
                if "//" in line:
                    comment.append(line.replace('//', ''))
                else:
                    code.append(line)
            if len(code) > 1 and len(comment) > 0:
                        code_snippets.append('\n'.join(code))
                        comments.append('\n'.join(comment))

    pairs = pd.DataFrame(zip(code_snippets, comments), columns = ["method", "comment"])
    return pairs


def add_inline(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Helper function to go through all methods in a given dataframe and add all
    pairs of inline comments and corresponding code snippets

    :param df: the dataframe to retrieve and add all pairs of inline comments
    and corresponding code snippets to
    :returns: a new dataframe with the newly added pairs of inline comments and
    corresponding code snippets
    '''
    new_df = df[df['method'].str.contains("//")]
    all_pairs = []
    for method in tqdm(new_df.method.values):
        pairs = get_inline_pairs(method)
        all_pairs.append(pairs)

    df_pairs = pd.concat([pairs for pairs in all_pairs])
    return pd.concat([df, df_pairs])

def remove_html_tags(comment: str) -> str:
    '''
    Remove any HTML tags from a given comment

    :param comment: the comment to remove any HTML tags from
    :returns: the comment with any HTML tags removed
    '''
    return BeautifulSoup(comment, "html.parser").get_text()

def primary_cleaning(train_df, val_df, test_df) -> None:
    '''
    Convert to lower case, remove extra whitespace, remove empty comments, and remove duplicates.
    '''
    train_df = train_df.applymap(lambda x: ' '.join(x.split()).lower())
    val_df = val_df.applymap(lambda x: ' '.join(x.split()).lower())
    test_df = test_df.applymap(lambda x: ' '.join(x.split()).lower())

    train_df = train_df[~(train_df['comment'] == '')]
    val_df = val_df[~(val_df['comment'] == '')]
    test_df = test_df[~(test_df['comment'] == '')]

    train_df = train_df[~train_df['comment'].duplicated()]
    val_df = val_df[~val_df['comment'].duplicated()]
    test_df = test_df[~test_df['comment'].duplicated()]

"""
DATA EXPLORATION
"""

def get_counter(df: pd.DataFrame, tokenizer: AutoTokenizer, col: str) -> Counter:
    '''
    Get the counts for each token in a given pandas dataframe column

    :param df: the pandas dataframe to get the counts of tokens from
    :param tokenizer: the tokenizer to use for tokenizing the rows in the pandas
    dataframe
    :param col: the column to grab rows from when tokenizing
    :returns: the counts of each token in the given pandas dataframe
    column
    '''
    toks = []
    for i, row in df.iterrows():
        toks.extend(tokenizer.tokenize(row[col]))

    cnt = Counter()
    for tok in toks:
        cnt[tok] += 1
    return cnt

def filter_len(
    row: pd.Series, tokenizer: AutoTokenizer, method_len: int, comment_len: int
    ) -> bool:
    '''
    Determine if a given panda dataframe row has a method or comment that has
    more tokens than max length

    :param row: the row to check if it has a method or comment that is too long
    :param tokenizer: the tokenizer to tokenize a method or comment
    :param method_len: the max number of tokens a method can have
    :param comment_len: the max number of tokens a comment can have
    :returns: whether or not the given row have a method or comment that have
    more tokens than a max length
    '''
    return len(tokenizer.tokenize(row.method)) < method_len and len(tokenizer.tokenize(row.comment)) < comment_len


"""
TRAINING
"""
def convert_dataset_to_codexglue_structure(train_df, val_df, test_df):
    train_df['code_tokens'] = train_df.method.apply(lambda x: x.split())
    train_df['docstring_tokens'] = train_df.comment.apply(lambda x: x.split())
    with open('python/train.jsonl','w') as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    val_df['code_tokens'] = val_df.method.apply(lambda x: x.split())
    val_df['docstring_tokens'] = val_df.comment.apply(lambda x: x.split())
    with open('python/valid.jsonl','w') as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    test_df['code_tokens'] = test_df.method.apply(lambda x: x.split())
    test_df['docstring_tokens'] = test_df.comment.apply(lambda x: x.split())
    with open('python/test.jsonl','w') as f:
        for _, row in test_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

"""
FUNCTION CALLS
"""
train_df, val_df, test_df = split_dataset(raw_dataset)

print("lengths of datasets before processing: ", len(train_df), len(val_df), len(test_df))

train_df = train_df[train_df['method'].apply(lambda x: is_ascii(x))]
val_df = val_df[val_df['method'].apply(lambda x: is_ascii(x))]
test_df = test_df[test_df['method'].apply(lambda x: is_ascii(x))]

train_df = train_df[train_df['comment'].apply(lambda x: is_ascii(x))]
val_df = val_df[val_df['comment'].apply(lambda x: is_ascii(x))]
test_df = test_df[test_df['comment'].apply(lambda x: is_ascii(x))]

train_df = add_inline(train_df)
val_df = add_inline(val_df)
test_df = add_inline(test_df)

train_df.comment = train_df.comment.apply(remove_html_tags)
val_df.comment = val_df.comment.apply(remove_html_tags)
test_df.comment = test_df.comment.apply(remove_html_tags)

primary_cleaning(train_df, val_df, test_df)

tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
method_cnt = get_counter(train_df, tokenizer, 'method')
comment_cnt = get_counter(train_df, tokenizer, 'comment')
method_lens = train_df.method.apply(lambda x: len(tokenizer.tokenize(x))).values
comment_lens = train_df.comment.apply(lambda x: len(tokenizer.tokenize(x))).values
max_method_len = int(np.quantile(method_lens, 0.95))
max_comment_len = int(np.quantile(comment_lens, 0.95))

train_df = train_df[train_df.apply(
    lambda row: filter_len(
        row, tokenizer, max_method_len,
        max_comment_len
    ), axis = 1
)]
val_df = val_df[val_df.apply(
    lambda row: filter_len(
        row, tokenizer, max_method_len,
        max_comment_len
    ), axis = 1
)]
test_df = test_df[test_df.apply(
    lambda row: filter_len(
        row, tokenizer, max_method_len,
        max_comment_len
    ), axis = 1
)]

print("lengths of datasets after processing: ", len(train_df), len(val_df), len(test_df))

convert_dataset_to_codexglue_structure(train_df, val_df, test_df)

"""
TRAINING CALLS
"""

lang = 'python' # programming language
lr = 5e-5
batch_size = 8 # change depending on the GPU Colab gives you
beam_size = 10
source_length = 256
target_length = max_comment_len
data_dir = '.'
output_dir = f'model/{lang}'
train_file = f'{data_dir}/{lang}/train.jsonl'
dev_file = f'{data_dir}/{lang}/valid.jsonl'
epochs = 10
pretrained_model = 'microsoft/codebert-base'

# The command to be executed
command = [
    "python", "run.py",
    "--do_train",
    "--do_eval",
    "--do_lower_case",
    "--model_type", "roberta",
    "--model_name_or_path", pretrained_model,
    "--train_filename", train_file,
    "--dev_filename", dev_file,
    "--output_dir", output_dir,
    "--max_source_length", str(source_length),
    "--max_target_length", str(target_length),
    "--beam_size", str(beam_size),
    "--train_batch_size", str(batch_size),
    "--eval_batch_size", str(batch_size),
    "--learning_rate", str(lr),
    "--num_train_epochs", str(epochs)
]

# Run the command
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Capture the output and errors (optional)
stdout = result.stdout.decode('utf-8')
stderr = result.stderr.decode('utf-8')

print("Output:", stdout)
print("Errors:", stderr)
