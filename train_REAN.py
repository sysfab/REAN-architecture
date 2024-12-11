# %%
# note:
# if youre training on a server thru SSH, i would recommend you use the .py file
# for training, as this would let you remount a TMUX terminmal if you disconnect.

# %%
###   CONFIGURATION   ###

# %%
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
import tokenizer
import os
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

# %%
# word2vec
model_file = fr"./embedding_models/YOUR_GENSIM_MODEL_NAME.model"
embeddings_model = Word2Vec.load(model_file)

vector_size = embeddings_model.vector_size        # aka embedding dim

# neural net settings
context_length = 128                              # tokens to consider
attn_heads = 8                                    # num attention heads per mechanism (per transformer block)
dropout_prob = 0.0                                # 0.0 ---> everything normal   |   1.0 ---> everything is random

# dataset
# !!!WARNING!!! bcs of various optimizations / errors on 8aafff's part small toy datasets dont work.
# if ur running a mini dataset, copy paste the text inside multiple times for proper execution
train_dataset_path = fr"./datasets/YOUR_PLAINTEXT_TRAIN_DATASET.txt"
test_dataset_path = fr"./datasets/YOUR_PLAINTEXT_TEST_DATASET.txt"

examples_train = 640#64 * 8 * 8 * 8 * 8 * 8 * 8 * 8
examples_test = 640# * 8 * 8

# train
train_epochs = 120

start_lr = 0.00003
final_lr = 0.000001

loss = nn.MSELoss()

optimizer = torch.optim.Adam
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

train_batch_size = int(128)

# eval
eval_batch_size = int(128)
eval_loop_batch = 64

# test
test_loop_batch = 256
completion_length = 128
test_prompts = ["human: how do i cook poratoes and garlic? network: ",
                "human: what are some good circuit training excercises? network: ",
                "human: tell me about graphics cards and ",
                "network: as an ai language ",
                ""]

# pytorch
run_device = torch.device("cuda")
storage_device = torch.device("cpu")

use_tensorboard = True
log_dir = "./runs" # irrelevant if use_tensorboard = False
run_name = "exp1" # irrelevant if use_tensorboard = False

# checkpoints & backups
save_checkpoint_batch = 512
save_dir = log_dir + "/" + run_name + "/" + "weights"
checkpoint_name = "REAN_checkpoint_date_[DATE]_batch_[BATCH]_epoch_[EPOCH].pth"

# %%
# command to get freaky and bulldoze the entire server (needed in case FBI bust down the door):
# sudo pkill "python|ipython|ipykernel|tensor|tensorboard|board"

# %%
###   NEURAL NET ARCHITECTURE   ###

# %%
class leaky_tanh_smart(nn.Module):
    def __init__(self, leaky_range=(0, 3), squishy_range=(0, 3)):
        super(leaky_tanh_smart, self).__init__()
        # register leakyness and squishyness as trainable parameters
        self.leakyness = nn.Parameter(torch.rand(1, dtype=torch.float32) * (leaky_range[1] - leaky_range[0]) + leaky_range[0])
        self.squishyness = nn.Parameter(torch.rand(1, dtype=torch.float32) * (squishy_range[1] - squishy_range[0]) + squishy_range[0])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        applies the leaky tanh activation function over the input tensor x.\n
        for more info on leaky tanh and its parameters go to: https://www.desmos.com/calculator/kpzsfbtqww
        
        Args:
            x (torch.Tensor): tensor over which to apply activation function.
        
        Returns:
            torch.Tensor: returns x after function applied, keeps the same shape.
        """
        
        return F.tanh(x * self.squishyness) + self.leakyness * x

# %%
class attention_mech(nn.Module):
    def __init__(self, vector_size=vector_size, attn_heads=attn_heads):
        super(attention_mech, self).__init__()
        # MultiheadAttention module
        self.multihead_attn = nn.MultiheadAttention(embed_dim=vector_size, num_heads=attn_heads)
        
        # Layer normalization
        self.norm = nn.LayerNorm(vector_size)

    def forward(self, x):
        # Prepare for multi-head attention (transpose to (sentence_len, batch_size, embedding_dim))
        x = x.transpose(0, 1)
        
        # Create causal mask
        seq_len = x.size(0)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        
        # Apply multi-head attention with the causal mask
        attn_output, attn_weights = self.multihead_attn(x, x, x, attn_mask=causal_mask)
        
        # Apply layer normalization to the attention output
        attn_output = self.norm(attn_output)
        
        # Transpose back to (batch_size, sentence_len, embedding_dim)
        output = attn_output.transpose(0, 1)
        
        return output, attn_weights

# %%
class positional_encoding(nn.Module):
    def __init__(self):
        super(positional_encoding, self).__init__()

    def forward(self, x):
        batch_size, context_length, vector_size = x.size()

        # Generate positions (shape: [context_length, 1])
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1).to(x.device)

        # Compute the divisor term (shape: [vector_size // 2])
        div_term = torch.exp(torch.arange(0, vector_size, 2).float() * (-math.log(10000.0) / vector_size)).to(x.device)

        # Initialize positional encoding tensor (shape: [context_length, vector_size])
        pe = torch.zeros(context_length, vector_size, device=x.device)
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cosine for odd indices

        # Add positional encoding to the input
        x = x + pe.unsqueeze(0)  # Add positional encoding, shape becomes (batch_size, context_length, vector_size)

        return x

# %%
class transformer_block(nn.Module):
    def __init__(self, vector_size=vector_size):
        super(transformer_block, self).__init__()
        
        self.activ_func = leaky_tanh_smart()
        
        self.attn = attention_mech()
        
        self.fc = nn.Linear(vector_size, vector_size)
        
        self.norm1 = nn.LayerNorm(vector_size)
        self.norm2 = nn.LayerNorm(vector_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x)[0])
        x = self.norm2(x + self.activ_func(self.fc(x)))
        
        return x

# %%
class REAN(nn.Module):
    def __init__(self):
        super(REAN, self).__init__()
        
        self.pos_encoding = positional_encoding()
        
        self.tblock1 = transformer_block()
        self.tblock2 = transformer_block()
        self.tblock3 = transformer_block()
        self.tblock4 = transformer_block()

    def forward(self, segment: torch.Tensor) -> torch.Tensor:
        """
        this function is primarily used for training, where the network needs to predict the next token, for every token in the sequence
        
        Args:
            segment (torch.Tensor): this is a tensor of size (batches, context_length, vector_size) representing a sequence of tokens (of course from the tokenizer and using the correct word2vec model)
        
        Returns:
            torch.Tensor: a tensor of shape (batches, context_length, vector_size) (same as segment) representing the sequence predicted by the network shifted future-way
        """
        
        ###                  INPUT                 ###
        #    (batches, context_len, vector_size)
        #                      ↓
        
        segment = self.pos_encoding(segment)
        
        segment = self.tblock1(segment)
        segment = self.tblock2(segment)
        segment = self.tblock3(segment)
        segment = self.tblock4(segment)
        
        return segment
    
        #                      ↓
        #    (batches, context_len, vector_size)
        ###                 OUTPUT                 ###

    def predict(self, segment: torch.Tensor) -> torch.Tensor:
        """
        function is for predicting the embeddings vector of the next token in a given sequence
        
        Args:
            segment (torch.Tensor): this is a tensor of size (batches, context_length, vector_size) representing a sequence of tokens (of course from the tokenizer and using the correct word2vec model)
        
        Returns:
            torch.Tensor: a tensor of shape (batches, vector_size) representing the embeddings vector of the next token to be added into the sequence
        """
        
        ###                  INPUT                 ###
        #    (batches, context_len, vector_size)
        #                      ↓
        
        segment = self.forward(segment)
        
        return segment[:, -1, :]
        
        #                      ↓
        #           (batches, vector_size)
        ###                 OUTPUT                 ###

# %%
###   BUILD NET & DEPENDENCIES   ###

# %%
net = REAN()

net.to(run_device)

optimizer = optimizer(net.parameters(), lr=start_lr)
scheduler = scheduler(optimizer, T_max=train_epochs, eta_min=final_lr)

print(f"neural net weight: {sum(param.numel() * param.element_size() for param in net.parameters()) / (1024 ** 3):.4f}GB")

# %%
###   UTIL FUNCS   ###

# %%
def vectorize_segment(segment: list[str], model: Word2Vec=embeddings_model, default: int = 0, used_device=storage_device) -> np.ndarray:
    """
    encodes all words in a given list to corresponding vectors in given model.
    words not found in the model will be given a vector with "default" value
    
    Args:
        sentence (list): list of strings (tokenized sentence)
        model (Word2Vec): model to use when encoding
        default (int): fill vector with this value if word is not found in model
    
    Returns:
        np.array: 2d array with dim1 = len(sentence) and dim2 = model.vector_size
    """
    
    # generate inital array with default values
    vectorized = np.ones((len(segment), model.vector_size)) * default
    
    # loop over every word in list
    for current_word, current_word_idx in zip(segment, range(len(segment))):
        # only add correct values if word is in model, otherwise leave as default
        if current_word in model.wv:
            # the try except block is needed because (current_word in model.wv) sometimes gives a false positive... yeah gensim
            try:
                vectorized[current_word_idx] = model.wv.get_vector(current_word, norm=False)
            except:
                pass
    
    vectorized = torch.tensor(vectorized, dtype=torch.float32, device=used_device)
    
    return vectorized

# %%
def devectorize_segment(vectorized_segment: torch.Tensor, model: Word2Vec=embeddings_model, not_in_vocab_token="[NIV]", NIV_threshold=0.01) -> list:
    """
    decodes vectors into nearest word found in model, if no near words found, adds a not in vocab token
    
    Args:
        vectorized_sentence (np.array): 2d arrat with vectors of words to be decoded
        model (Word2Vec): model to use when decoding
    
    Returns:
        list: list of strings (words) whos vectors most closely match those provided
    """
    
    result = []
    
    # make sure vectors are ready to be processed
    vectorized_segment = vectorized_segment.cpu().numpy()
    
    # go over all words and find closest match in model
    for current_word in vectorized_segment:
        similarities = model.wv.similar_by_vector(current_word)
        
        # check if its not a bullshit vector
        if similarities[0][1] > NIV_threshold:
            result.append(similarities[0][0])
        else:
            result.append(not_in_vocab_token)
    
    return result

# %%
def pad_or_truncate(suspected_tensor: torch.tensor, target_length: int, default: int=0) -> torch.Tensor:
    """
    pads or truncates a given tensor along dim 0 to target_length with "default" as padding
    
    Args:
        suspected_tensor (torch.tensor): tensor to pad or truncate
        target_length (int): target length of tensor
        default (int): value to use for padding
    
    Returns:
        torch.tensor: tensor of proper length no matter what
    """
    
    if len(suspected_tensor) < target_length:
        # pad
        suspected_tensor = torch.cat((torch.ones(target_length - len(suspected_tensor), suspected_tensor.shape[1], dtype=torch.float32, device=suspected_tensor.device) * default, suspected_tensor))
    else:
        # truncate
        suspected_tensor = suspected_tensor[-target_length:]
    
    return suspected_tensor

# %%
def prepare_segment_for_net(segment: list[str], length: int=context_length, used_device: torch.DeviceObjType=storage_device):
    """
    function to take a sentence, and do everything to make it possible to input into the net
    
    Args:
        segment (list[str]): a list of tokens (ideally from the tokenizer) of a sentence / text
        length (int): the number of tokens to which pad or truncate to. for correct operation: keep at the net's context length
    
    Returns:
        torch.Tensor: tokenized segment in the correct length
    """
    
    # turn into embedding vectors
    vectorized = vectorize_segment(segment, used_device=used_device)
    
    # trim / add into length
    trimmed = pad_or_truncate(vectorized, length)
    
    # add fake batch dimension
    batched = trimmed.unsqueeze(0)
    
    return batched

# %%
def predict_word(segment: list[str], net: REAN=net):
    # turn tokenized text into net's format
    prepared_segment = prepare_segment_for_net(segment, used_device=next(net.parameters()).device)
    
    # run net
    prediction_vector = net.predict(prepared_segment).detach()
    
    # turn vector back into token
    predicted_token = devectorize_segment(prediction_vector)
    
    return predicted_token

# %%
def predict_sequence(segment: list[str], num_tokens: int, net: REAN=net, display_tqdm=False):
    result = segment.copy()
    
    for _ in tqdm(range(num_tokens), disable=not display_tqdm):
        result += predict_word(result, net=net)
    
    return result[len(segment):]

# %%
###   BUILD DATASET   ###

# %%
class REAN_dataset(Dataset):
    def pull_tokens(self, start_read_idx: int, requested_num_tokens: int):
        """
        function returns a requested number of tokens from the dataset file, starting at APPROXIMATLY the start_read_idx token.\n
        attempts to return full words as much as possible, example:\n
        NO:    this | is | a | sen (tence)\n
        YES:   this | is | a | sentence
        
        Args:
            start_read_idx (int): the APPROXIMATE token at which to start the reading (determined from the avarage token length in the tokenizer vocab)
            requested_num_tokens (int): how many tokens to return
        
        Returns:
            tokenized text (list of str): the tokens of the dataset from start_read_idx to start_read_idx + requested_num_tokens
            is EOF hit (bool): if the requested args were outside of the dataset's range
        """
        
        with open(self.path, errors="ignore") as self.dataset:
            self.dataset.seek(start_read_idx * tokenizer.average_token_length)
            
            # get an initial estimate to what text we will actually need
            self.buffer = self.dataset.read(requested_num_tokens * tokenizer.average_token_length)
            self.tokenized_buffer = tokenizer.tokenize_segment(self.buffer)
            self.current_num_tokens = len(self.tokenized_buffer)
            
            # if the estimate we took is too small, we enlarge it character by character until its perfect
            while self.current_num_tokens < requested_num_tokens + 1:
                self.next_char = self.dataset.read(1)  # seperate variable to check EOF
                
                # check eof
                if not self.next_char:
                    print("pull_tokens(): eof was hit")
                    return self.tokenized_buffer[-requested_num_tokens - 1:][:-1], True
                
                self.buffer += self.next_char
                
                self.tokenized_buffer = tokenizer.tokenize_segment(self.buffer)
                self.current_num_tokens = len(self.tokenized_buffer)
        
        # regardless of if the estimate is too long / short, return theproper amount of tokens, with the end snipped of, because it might be a half token
        return self.tokenized_buffer[-requested_num_tokens - 1:][:-1], False
    
    def construct_example(self, start_read_idx: int):
        """
        function to make a full datapoint, can be used as raw return for __getitem__()
        
        Args:
            start_read_idx (int): at which token to start making the example
        
        Returns:
            tokenized text (list of str): the tokens of the dataset from start_read_idx to start_read_idx + self.context_length
        """
        
        # pull neccesary amount of tokens for question / input and answer / output
        self.tokens, _ = self.pull_tokens(start_read_idx, self.context_length + 1)
        
        # encode the tokens to vectors (aka embeddings)
        self.vectorized_tokens = prepare_segment_for_net(self.tokens, length=self.context_length + 1).squeeze(0)
        
        # split into network input and expected output
        self.question = self.vectorized_tokens[:-1] # everythinbg up to last word
        self.answer = self.vectorized_tokens[1:] # last word itself
        
        return self.question, self.answer
    
    def get_size(self):
        """
        function to read thru the whole dataset, and report how many examples there are / if there are as many as the user requested
        
        Args:
            none, but uses self.num_tokens and self.context_length
        
        Returns:
            returns how many usable examples there are, for __len__()
        """
        
        with tqdm(total=self.num_examples, desc="Calculating Dataset Size", unit="example") as pbar:
            for self.current_check in range(self.num_examples):
                _, self.eof = self.pull_tokens(self.current_check, self.context_length)
                
                if self.eof:
                    print("The requested size is bigger than the .txt provided, so the dataset might be smaller than what you expected.")
                    break

                pbar.update(1)

        print(f"Requested num_examples: {self.num_examples}\nActual size found:      {self.current_check - 1}")
        
        return self.current_check - 1   # the -1 is just in case
    
    def __init__(self, path, num_examples, context_length, embeddings_model, verify_dataset_size=True):
        # transfer to object wide variables
        self.path = path
        self.context_length = context_length
        self.embeddings_model = embeddings_model
        self.num_examples = num_examples
        
        # get the size of the dataset txt file
        self.dataset_len = num_examples
        
        if verify_dataset_size:
            self.dataset_len = self.get_size()

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        return self.construct_example(index)

# %%
train_dataset = REAN_dataset(train_dataset_path, examples_train, context_length, embeddings_model, verify_dataset_size=False)
test_dataset = REAN_dataset(test_dataset_path, examples_test, context_length, embeddings_model, verify_dataset_size=False)

# %%
print("please validate dataset: does this look correct?\n")

with torch.no_grad():
    rnd_offset = random.randint(0, 100)
    
    for idx in range(3):
        print(f"sample {idx}:[nline]{tokenizer.detokenize_segment(devectorize_segment(train_dataset[idx + rnd_offset][0].detach(), embeddings_model))}[nline]------------------------------------------------------------[nline]{tokenizer.detokenize_segment(devectorize_segment(train_dataset[idx + rnd_offset][1].detach(), embeddings_model))}".replace("\n", " ").replace("[nline]", "\n"))

# %%
# if num_workers arg is used
os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)#, num_workers=4, persistent_workers=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size, shuffle=True)#, num_workers=4, persistent_workers=True)

# %%
###   TRAIN   ###

# %%
net.train()
clear_output()

# %%
if use_tensorboard:
    writer = SummaryWriter(log_dir = log_dir + "/" + run_name)

# %%
batch = 0

for epoch in range(train_epochs):
    # training loop
    for current_segment, target in train_loader:
        batch += 1
        
        
        # move batch to gpu
        current_segment = current_segment.to(run_device)
        target = target.to(run_device)
        
        # train batch
        train_outputs = net(current_segment)
        train_loss_value = loss(train_outputs, target)
        train_loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if use_tensorboard:
            # Log training loss to TensorBoard
            writer.add_scalar('train_loss', train_loss_value.item(), batch)
        
        # eval loop
        if batch % eval_loop_batch == 0:
            net.eval()
            
            with torch.no_grad():
                for test_current_segment, test_target in test_loader:
                    # move batch to gpu
                    test_current_segment = test_current_segment.to(run_device)
                    test_target = test_target.to(run_device)
                    
                    # run test
                    test_outputs = net(test_current_segment)
                    test_loss_value = loss(test_outputs, test_target)
                    
                    if use_tensorboard:
                        # Log test loss to TensorBoard
                        writer.add_scalar('test_loss', test_loss_value.item(), batch)
            
            net.train()
    
        # test loop
        if batch % test_loop_batch == 0:
            if use_tensorboard:
                for current_prompt in test_prompts:
                    prediction = tokenizer.detokenize_segment(
                        predict_sequence(tokenizer.tokenize_segment(current_prompt), completion_length)
                    ).replace("\n", "/n")
                    
                    # Log predictions along with the prompt to TensorBoard with enhanced formatting
                    formatted_text = (
                        f"---PROMPT---\n{current_prompt}"
                        "\n\n==========================================================================================================\n\n"
                        f"---PREDICTION---\n{prediction}"
                    )
                    # Overwrite the previous entry for the same prompt
                    writer.add_text(f'Predictions/{current_prompt}', formatted_text, batch)
        
        # save checkpoint
        if batch % save_checkpoint_batch == 0:
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save(net, save_dir + "/" + checkpoint_name
                       .replace("[DATE]", (datetime.utcnow() + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'))
                       .replace("[BATCH]", str(batch))
                       .replace("[EPOCH]", str(epoch)))
    
    # Update the learning rate scheduler
    scheduler.step()
    
    if use_tensorboard:
        # Log learning rate to TensorBoard
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

# %%
###   EVAL   ###

# %%
net.eval()
clear_output()

# %%
prompt = "human: " + "write a list of the top 10 sports cars" + " network: "
tokens_to_predict = 128
display_tqdm = True

print(tokenizer.detokenize_segment(predict_sequence(tokenizer.tokenize_segment(prompt), tokens_to_predict, display_tqdm=display_tqdm)))

# expected (in exact order):
# 1. porche 2. bmw 3. mclaren 4. dodge 5. ferrari 6. mercedes etc...
# if mercedes is put above BMW immediatly delete all traces of the .pth on ur machine

# %%



