import re

# get misc info about tokenizer (i just made these numbers up, theyre for security / optimization in REAN code)
max_token_size = 12
average_token_length = 8

def tokenize_segment(text, con_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"):
    """
    this function serves as the tokenizer for training and using both the word2vec, and REAN.
    breaks up string given into a list of tokens (substrings).
    to detokenize just use "".join() on the resulting list / or alternativly use tthe next function
    
    Args:
        text (str): the sequence to be tokenized
        con_chars (str): constant chars - a string of all the chars to NOT break up, and keep the same in the same cells / tokens in the resulting list
    
    Returns:
        list[str]: a list of tokens, representing text arg
    """
    
    # dunno chatgpt wrote it
    pattern = fr"""
        [{re.escape(con_chars)}]+        # Match one or more constant characters (words)
        (?:[\s]+)?                       # Optionally include following whitespace
      | [^\s{re.escape(con_chars)}]      # Match any single character not in constant characters or whitespace (punctuation)
        (?:[\s]+)?                       # Optionally include following whitespace
    """
    
    tokens = re.findall(pattern, text, re.VERBOSE)

    return tokens

def detokenize_segment(tokens):
    """
    a function to input tokens, and join them back into a readable sequence
    
    Args:
        tokens (list[str]): sequence outputted by the tokenize_segment function, to be turned back intto normal text
    
    Returns:
        (str): tokens sequence as normal string
    """
    
    return "".join(tokens)