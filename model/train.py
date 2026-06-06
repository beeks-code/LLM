from transformer import Transformer,LayerNormalization,GELU,FeedForward,MultiHeadAttention
from gpt import GPT_124
from torch.utils.data import Dataset,DataLoader
from torch.optim import adam
import tiktoken
import torch.nn as nn
import torch


EPOCHS=10
LR=0.01

## training config

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

class GPT2Tokenizer(Dataset):
    def __init__(self,text,tokenizer,context_size,strides):
        self.input_ids=[]
        self.target_ids=[]
        tokens=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        for i in range(0,len(tokens)-context_size,strides):
            in_tokens=tokens[i:i+context_size] ## input token chunck
            target_tokens=tokens[i+1:i+context_size+1] ## target token chunk
            self.input_ids.append(torch.tensor(in_tokens))
            self.target_ids.append(torch.tensor(target_tokens))
    def __len__(self):
        return len(self.input_ids)  
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index] 

def create_dataloader(txt,batch_size,context_size,strides,Suffle=True,drop_last=True,num_workers=0):
    tokeizer=tiktoken.get_encoding("gpt2")
    datset=GPT2Tokenizer(txt,tokenizer=tokeizer,context_size=context_size,strides=strides)
    dataloader=DataLoader(dataset=datset,batch_size=batch_size,shuffle=Suffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader             

with open("the-verdict.txt","r",encoding="utf-8") as file:
    text=file.read()

dl=create_dataloader(text,batch_size=10,
                     context_size=GPT_CONFIG_124M["context_length"],
                     strides=GPT_CONFIG_124M["context_length"])


def loss_cal(logits,target):
    """ 
    out_embd are direct output from gpt -> (batch,n_tokens,n_vocab_size) 
    targets are the target excpected output -> (batch,n_tokens)
    """
    logits_flat = logits.flatten(0, 1)
    target_flat=target.flatten()
    loss=nn.functional.cross_entropy(logits_flat,target_flat)
    return loss
model=GPT_124(GPT_CONFIG_124M)

def train(model,train_loader,optimizer,n_epochs):
    train_loss,val_loss,track_seen_token=[],[],[]
    token_seen,global_step=0,-1


    for i in range(n_epochs):
        model.train()
        for input_batch,output_batch in train_loader:
            optimizer.zero_grad()

            logits=model(input_batch)
            with torch.no_grad():
                loss=loss_cal(logits,output_batch)

            
            optimizer.step()
            token_seen=input_batch.numel()
            global_step += 1
            if (global_step%10==0):
                print(loss.items)


