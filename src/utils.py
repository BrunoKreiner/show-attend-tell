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
    """
    Generate Vocabulary Dictionaries. One is index to string and one is string to index to map values back and forth.

    Args:
        freq_threshold (int): minimum frequency threshold for words to be added to vocab
    """
    def __init__(self, freq_threshold):
        """
        Set the pre-reserved tokens and the frequency threshold for words to be added to the vocab

        Args:
            freq_threshold (int): frequency threshold for words to be added to vocab
        """
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        """
        Tokenize a string of text into a list of tokens with spacy and convert to lowercase

        Args:
            text (string): text to be converted into a list of tokens

        Returns:
            list<string>: list of tokenized text
        """
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        """
        build vocabulary from a list of sentences (all captions in flickr8k).
        A words frequency is counted in a Counter() object and if it reaches the frequency threshold it is added to the vocab.

        Args:
            sentence_list (list<string>): List of sentences to build vocab from
        """
        frequencies = Counter()
        idx = 4 #start index from 4 as 0,1,2,3 are reserved for pre-defined tokens <PAD>,<SOS>,<EOS>,<UNK>
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """
        Convert a text into a list of numerical tokens based on the vocab dictionary
        First use spacy tokenizer function to tokenize the whole text, then convert each token to their string-to-index (stoi) vocabulary value

        Args:
            text (string): text to be converted to a list of tokens 

        Returns:
            list<int>: list of tokens from stoi vocab
        """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]  

class FlickrDataset(Dataset):
    """
    Flickr8k Dataset for Pytorch Dataloader.
    
    Args:
        root_dir (string): path to the root directory of the dataset
        captions_df (pandas dataframe): dataframe containing the captions and image names
        transform (torchvision.transforms) (optional, default = None): transforms to be applied to the image
        vocab (Vocabulary) (optional, default = None): vocabulary object to convert text to numerical tokens
        freq_threshold (int) (optional, default = 1): frequency threshold for words to be added to vocab
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
        """
        Return the length of the dataset

        Returns:
            int: length of dataset
        """
        return len(self.df)
    
    def __getitem__(self,idx):
        """
        Get an item from the dataset at a given index

        Args:
            idx (int): idx given by the dataloader calling the dataset

        Returns:
            img (torch.tensor): image tensor
            caption_vec (torch.tensor): caption tensor (tokenized)
        """
        caption = self.captions[idx]
        img_name = self.imgs[idx] + ".jpg"
        img_location = os.path.join(self.root_dir, img_name)
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
    """
    Plots an image from a torch tensor

    Args:
        img (torch.tensor): _description_
        title (string) (optional, default = None): title for the plot
    """
    img = img.numpy().transpose((1,2,0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class CapsCollate:
    """
    Custom collate function for the dataloader to pad the captions to the same length

    Args:
        pad_idx (int): index of the padding token in the vocabulary
        batch_first (bool) (optional, default = False): if True, the batch dimension is the first dimension
    """

    def __init__(self, pad_idx = 0, batch_first = True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        """
        function called by the dataloader for each batch

        Args:
            batch (list of tuples of img and caption tensors): batch

        Returns:
            list of tuples of img and caption tensors: prepared batch
        """
        try:
            images_ = list()
            captions_ = list()

            for b in batch:
                images_.append(b[0])
                captions_.append(b[1])

            images_ = torch.stack(images_, dim=0)
            captions_ = pad_sequence(captions_, batch_first=self.batch_first, padding_value= self.pad_idx)

            return images_, captions_
        
        except Exception as e:
            print(e)
            print(batch)

def show_tensor_image(img, title = None):
    """
    Show image from a tensor but reverse the normalization from flickr8k using the mean and std

    Args:
        img (torch.tensor): _description_
        title (string) (optional, default = None): title for the plot
    """

    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406

    img = img.numpy().transpose((1,2,0))

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class EncoderCNN(nn.Module):
    """
    EncoderCNN class for the encoder part of the model
    Uses pretrained resnet50 from torchvision.models.
    The last layer is removed and a new layer is added to get the output of the shape (batch_size, embed_size)
    """
    def __init__(self, filter_size = 14, resnet_type = "resnet50"):
        super(EncoderCNN,self).__init__()
        if resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif resnet_type == "resnet101":
            resnet= models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((filter_size,filter_size))
        #self.fine_tune()
    
    def forward(self,images):
        """
        forwards the images through the encoder,


        Args:
            batch of images (torch.tensor): image(s) to be encoded into feature vectors

        Returns:
            features (torch.tensor): tensor of encoded features out of the batch of images
        """
        features = self.resnet(images)# <batch size>, 1024, 14, 14 | features from resnet, 2048 filters, 7x7
        features = self.adaptive_pool(features) #<batch size>, 2040, filter size, filter size | adaptive avg pool to get the same size for all images
        
        # flatten the features for attention mechanism. flattened vector of 14x14 (196 values) are used for calculating attention weights.
        features = features.permute(0,2,3,1) #<batch size>, 7, 7, 2048 
        features = features.view(features.size(0),-1,features.size(-1)) #<batch size>, 196, 2048
        return features

class Attention(nn.Module):
    """
    Implements Attention mechanism for decoder RNN.
    Uses the encoder features and the decoder hidden state to calculate the attention weights.
    Uses a linear layer to transform the feature vector into the attention dimension.
    Uses a linear layer to transform the decoder hidden state into the attention dimension.
    Uses tanh activation function on the sum of the transformed feature vector and the transformed decoder hidden state.
    Uses another linear layer to transform the output of the tanh activation function into the attention scores.
    Uses softmax function on those attention scores to get the alphas.
    Uses the alphas to calculate the attention weights by multiplying with the features.

    Args:
        encoder_dim (int): dimension of feature vectors from the encoder
        decoder_dim (int): dimension of the decoder hidden state
        attention_dim (int): dimension of the attention vector
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention,self).__init__()
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        self.A = nn.Linear(attention_dim, 1)

        #self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, features, hidden_state):
        """
        Forwards features and last hidden state to calculate and return the new attention weights and alphas.
        First the 

        Args:
            features (torch.tensor): features from the encoder
            hidden_state (torch.tensor): last hidden state from the decoder. To quote the paper: 
                "The weight alpha_i of each annotation vector a_i
                is computed by an attention model f_attemtopm for which we use
                a multilayer perceptron conditioned on the previous hidden
                state ht-1." 
                Equations 4 and 5 in the paper. 

        Returns:
            tuple of alphas and attention weights: returns the newly calculated alphas and attention weights 
        """
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))

        attention_scores = self.A(combined_states) #attention score per pixel in condensed filters (196, 1)  
        attention_scores = attention_scores.squeeze(2) #squeeze to get (196,) for softmax
        alpha = F.softmax(attention_scores, dim=1)

        attention_weights = features * alpha.unsqueeze(2) #deterministic soft attention 
        attention_weights = attention_weights.sum(dim=1) #sum across i = 1 to 196 to get the attention weights [batch_size, encoder_dim (2048)]
        
        return alpha, attention_weights

class DecoderRNN(nn.Module):
    """
    Decode EncoderCNN output and use attention weights to predict the next word in the caption in a for loop going from 0 to max_length of the longest caption in batch.

    Args:
        embed_size (int): size of the word embeddings, same as linear layer but uses lookup instead of a matrix vector multiplication.
        vocab_size (int): size of the vocabulary.
        attention_dim (int): dimension of the attention vector.
        encoder_dim (int): dimension of feature vectors from the encoder.
        decoder_dim (int): dimension of the decoder hidden state.
        device (torch.device): device to run the model on.
        drop_prob (float) (optional, default = 0.3): dropout probability, for each word in the generated caption, the last layer goes through a fully connected layer with the set dropout probability.
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob = 0.3):
        super(DecoderRNN,self).__init__()

        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.device = device

        self.dropout = nn.Dropout(drop_prob)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        #use embedding layer to get the word embeddings for the original captions for teacher-forcing
        #links : 
        #   - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        #   - https://discuss.pytorch.org/t/how-nn-embedding-trained/32533
        #   - https://stackoverflow.com/a/51668224/5625096 #good answer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)

        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(self, features, captions):
        """
        Decodes features from encoder and predicts the next words in the caption in a loop.
        First initializes the hidden state and cell state of the LSTM cell.
        Then initializes the predictions and attention weights.
        Then iterates through the captions and predicts the next word in the caption.
        First the attention weights are calculated using the attention mechanism.
        Then the LSTM cell is used to predict the next word in the caption using the the caption embeddings (?) and the attention weights.
        Then the hidden state and cell state are updated.
        Then the output is calculated using dropout and a fully connected layer.
        This output gets added to the list of all predictions
        The alphas get also added to a list of all alphas.
        
        Args:
            features (torch.tensor): features from the encoder
            captions (torch.tensor): captions from the dataset
        """
        batch_size = features.size(0)
        num_features = features.size(1)
        vocab_size = self.vocab_size

        #embed the captions
        embeddings = self.embedding(captions)

        #initialize the hidden state and cell state
        h, c = self.init_hidden_state(features)
        
        #get the sequence length to iterate
        seq_length = [len(caption) - 1 for caption in captions  - 1]
        seq_length = max(seq_length)

        #initialize the preditions
        predictions = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)

        #initialize the attention weights
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)

        for s in range(seq_length):
            alphas_new, context = self.attention(features, h)
            
            #context = self.sigmoid(self.f_beta(h[:batch_size]))
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
        """
        Generates a caption until the end of sentence token is reached or the max length is reached.

        Returns:
            tuple: tuple of predicted captions and alphas per prediction step
        """
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)

        alphas_list = []

        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1).to(self.device)

        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alphas, context = self.attention(features, h)

            alphas_list.append(alphas.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
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
        return h, c

class EncoderDecoder(nn.Module):
    """
    Implements container for the whole model.
    Keeps encoder and decoder together.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob= 0.3, resnet_type = "resnet50"):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(resnet_type=resnet_type)
        self.decoder = DecoderRNN(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob = drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        predictions, alphas = self.decoder(features, captions)
        return predictions, alphas

def save_model(model, num_epochs):
    torch.save(model.state_dict(), 'model-{}.ckpt'.format(model.num_epochs))
    torch.save(model.optimizer.state_dict(), 'optimizer-{}.ckpt'.format(model.num_epochs))