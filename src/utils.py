import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import spacy
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]  

class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self, root_dir, captions_df, transform=None, vocab = None, freq_threshold=1):
        self.root_dir = root_dir
        self.transform = transform

        self.df = captions_df
        
        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions.tolist())
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx] + ".jpg"
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)

def show_image(img, title = None):
    img = img.numpy().transpose((1,2,0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class CapsCollate:

    def __init__(self, pad_idx, batch_first = False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        try:
            print(batch.shape)
            imgs = [item[0].unsqueeze(0) for item in batch]
            imgs = torch.cat(imgs, dim = 0)
            targets = [item[1] for item in batch]
            targets = pad_sequence(targets, batch_first = self.batch_first, padding_value = self.pad_idx)
            return imgs, targets
        except Exception as e:
            print(e)
            print(batch)

def show_tensor_image(img, title = None):

    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406

    img = img.numpy().transpose((1,2,0))

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        #self.fine_tune()
    
    def forward(self,images):
        features = self.resnet(images) #2040, 7
        features = features.permute(0,2,3,1) #2040, 7, 7, 2048
        features = features.view(features.size(0),-1,features.size(-1)) #2040, 49, 2048
        return features

#TODO: change to show attend and tell
class Attention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim):
        super(Attention,self).__init__()
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        self.A = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self,features,hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))

        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)

        alpha = F.softmax(attention_scores, dim=1)

        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)
        return alpha, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob = 0.3):
        super(DecoderRNN,self).__init__()

        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.device = device

        self.dropout = nn.Dropout(drop_prob)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        self.embedding = nn.Embedding(vocab_size,embed_size)

        self.init_h = nn.Linear(encoder_dim,decoder_dim)
        self.init_c = nn.Linear(encoder_dim,decoder_dim)
        self.f_beta = nn.Linear(decoder_dim,encoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim, decoder_dim, bias=True)

        self.fc = nn.Linear(decoder_dim,vocab_size)
        #self.init_weights()

    def forward(self, features, captions):
        batch_size = features.size(0)
        num_features = features.size(1)
        vocab_size = self.vocab_size

        #embed the captions
        embeddings = self.embedding(captions)

        #initialize the hidden state and cell state
        h,c = self.init_hidden_state(features)

        #get the sequence length to iterate
        seq_length = len(captions[0])-1

        #initialize the preditions
        predictions = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)

        #initialize the attention weights
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)

        for s in range(seq_length):
            alphas_new, context = self.attention(features, h)
            lstm_input = torch.cat((embeddings[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fc(self.dropout(h))

            predictions[:, s] = output
            alphas[:, s] = alphas_new

        return predictions, alphas

    """def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)"""

    def generate_caption(self, features, max_len = 20, vocab = None):
        batch_size = features.size(0)
        h, c = self.init_hiden_state(features)

        alphas_list = []

        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1).to(self.device)

        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alphas, context = self.attention(features, h)

            alphas_list.append(alphas.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fc(self.drop(h))
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim=1)

            captions.append(predicted_word_idx.item())

            if vocab.itos[predicted_word_idx.item()] == '<EOS>':
                break

            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        return [vocab.itos[idx] for idx in captions], alphas_list
    
    def init_hidden_state(self, features):
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        return h,c

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob= 0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob = drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        predictions, alphas = self.decoder(features, captions)
        return predictions, alphas

def save_model(model, num_epochs):
    torch.save(model.state_dict(), 'model-{}.ckpt'.format(model.num_epochs))
    torch.save(model.optimizer.state_dict(), 'optimizer-{}.ckpt'.format(model.num_epochs))