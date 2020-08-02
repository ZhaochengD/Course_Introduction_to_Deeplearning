import torch
import random
import time
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.autograd import Variable

import Levenshtein as Lev

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modelpath = '../param/M_46.pth'

tf_rate = 0.5
PARPATH = '../param/M_{}.pth'
OUTPATH = '../result/out.csv'
VALLOSS = [1000]
EPOCH = 50

def load_data():
    speech_train = np.load('../data/train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('../data/dev_new.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('../data/test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('../data/train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('../data/dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid

def transform_letter_to_index(transcript, letter2index):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above, ------- here I use letter2index
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter_to_index_list = []
    for line in transcript:
        mapping = [letter2index['<sos>']]
        mapping = [letter2index[i] for i in b' '.join(filter(None, line)).decode('UTF-8')]
        mapping.append(letter2index['<eos>'])
        letter_to_index_list.append(mapping)
    return letter_to_index_list

def create_dictionaries():
    index2letter = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
    letter2index = dict(zip(index2letter, [i for i in range(len(index2letter))]))
    return letter2index, index2letter

letter2index, index2letter = create_dictionaries()
vocab_size = len(letter2index)

class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))
        
def collate_train(batch_data):
    data = [torch.Tensor(i[0]) for i in batch_data]
    target = [torch.LongTensor(i[1][1:]) for i in batch_data]

    padded_data = pad_sequence(data,batch_first = True)
    padded_target = pad_sequence(target,batch_first = True)

    data_lens = []
    target_lens = []
    for i in batch_data:
      data_lens.append(len(i[0])) 
      target_lens.append(len(i[1][1:])) 

    return (padded_data),torch.LongTensor(data_lens),(padded_target), torch.LongTensor(target_lens)

def collate_test(batch_data):
    data = [torch.Tensor(i) for i in batch_data]
    padded_data = pad_sequence(data,batch_first=True)
    data_lens = []
    for i in batch_data:
      data_lens.append(len(i)) 
    return padded_data,torch.LongTensor(data_lens)

speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()

class LockDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, x): # L, N, C
        if self.p == 0 or not self.training:
            return x
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(1 - self.p)
        mask = Variable(mask.div_(1 - self.p), requires_grad=False)
        mask = mask.expand_as(x)
        return x * mask
    
class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted 
        '''
        # query.shape, key.shape, values.shape, reduced_length 
        # torch.Size([2, 128]) torch.Size([179, 2, 128]) torch.Size([179, 2, 128]) tensor([179, 109])
        
        # key = self.activate(key)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        query = query.unsqueeze(2)
        # Output shape of bmm: (batch_size, max_len, 1)
        #print('aaaaaaaaaaaa', key.shape, query.shape)
        energy = torch.bmm(key, query).squeeze(2)
        
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(device)
        # print('mask', mask)
        energy.masked_fill_(mask, -1e9)
        attention = nn.functional.softmax(energy,dim=1)
        
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        # print('context', context.shape, context)
        return context, attention

class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x):
        return self.blstm(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        
        self.cnn = torch.nn.Conv1d(input_dim,input_dim,kernel_size = 3,stride = 1,padding =1)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size = hidden_dim, num_layers=1, bidirectional=True)

        ### Add code to define the blocks of pBLSTMs! ###

        self.pblstm = []
        self.pblstm.append(pBLSTM(hidden_dim * 4,hidden_dim))
        self.pblstm.append(pBLSTM(hidden_dim * 4,hidden_dim))
        self.pblstm.append(pBLSTM(hidden_dim * 4,hidden_dim))

        self.pblstm = torch.nn.ModuleList(self.pblstm)

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

        self.lockeddrop = LockDropout(0.5)

    def forward(self, x, lens):
        outputs = x.to(device)
        lens = lens.to(device)
        
        outputs = outputs.transpose(1,2)
        #print('-----', outputs.shape)
        outputs = self.cnn(outputs)
        outputs = outputs.transpose(1,2)
        outputs = pack_padded_sequence(outputs, lengths=lens, enforce_sorted=False)
        outputs, _ = self.lstm(outputs)
        for l in range(len(self.pblstm)):
          outputs, _ = pad_packed_sequence(outputs)
          outputs = outputs.permute(1, 0, 2)
          new_T = outputs.shape[1] // 2
          lens = lens // 2
          new_H = outputs.shape[2] * 2
          if outputs.shape[1] % 2 == 1:
                  outputs = outputs[:, :-1, :].reshape(outputs.shape[0], new_T, new_H)
          else:
                  outputs = outputs.reshape(outputs.shape[0], new_T, new_H)

          outputs = self.lockeddrop(outputs.permute(1, 0, 2))
          outputs = pack_padded_sequence(outputs,lengths=lens,enforce_sorted=False)
          outputs, hidden = self.pblstm[l](outputs)

        linear_input, lens = pad_packed_sequence(outputs)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return lens, hidden, keys, value

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_size=128, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=embedding_size + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
        self.lockdrop1 = LockDropout(0.5)
        self.lockdrop2 = LockDropout(0.5)

        self.query_network = nn.Linear(embedding_size, key_size)
        self.isAttended = isAttended
        self.attention = Attention()

        self.fc = nn.Linear(key_size * 2, embedding_size)
        self.tanh = nn.Hardtanh(inplace = True)
        self.character_prob = nn.Linear(embedding_size, vocab_size)
        self.character_prob.weight = self.embedding.weight
        # self.fc = nn.Linear(key_size + value_size, ) 
    def forward(self, key, values, reduced_length, text=None,hidden=None, isTrain=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        teacher_force_rate = 0 if not isTrain else tf_rate

#        batch_size = key.shape[1]
        # print('----------------------text shape', text.shape)
        if (isTrain == True):
            # print('text.shape', text.shape, text)
            max_len =  text.shape[1]
            # print('max_len', max_len)
            embeddings = self.embedding(text)
        else:
            max_len = 250
        #print('------text-----', text)


        predictions = []
        hidden_states = [hidden, None]
#        print('aaaaaaaaaaaaa', hidden_states[0][0].shape)        
        # initialize context
        all_attention = []
        
        prediction = torch.zeros(text.shape[0],1).fill_(33).to(device)
        #print('----------------------max_len', max_len)
        for i in range(max_len):

            teacher_force = True if np.random.random_sample() < teacher_force_rate else False
            if(isTrain):
                if random.random() > tf_rate:
                    teacher_force = False
                else:
                    teacher_force = True

                if not teacher_force:
                    char_embed = self.embedding(prediction.argmax(dim = 1))
                else:
                    if i == 0:
                        char_embed = self.embedding(prediction.argmax(dim = 1))
                    else:
                        char_embed = embeddings[:, i - 1, :]
            else:
                # for validation
                char_embed = self.embedding(prediction.argmax(dim = 1))
#                else:
#                    char_embed = embeddings[:, i - 1, :]
            #print('aaaaaaaaaaaaaa', char_embed.shape)
            char_embed = char_embed.to(device)
            query = self.query_network(char_embed)
            context, attention = self.attention(query, key, values, reduced_length)
#             print('----------------1', attention)
            inp = torch.cat([char_embed, context], dim=1)
            inp = self.lockdrop1(inp.unsqueeze(1)).squeeze(1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            inp_2 = self.lockdrop2(inp_2.unsqueeze(1)).squeeze(1)
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]

#            print('bbbbbbbbbbbbbbb', output.shape)
            query = self.query_network(hidden_states[1][0])
            #print('query.shape, key.shape, values.shape, reduced_length', query.shape, key.shape, values.shape, reduced_length)
            # torch.Size([2, 128]) torch.Size([94, 2, 128]) torch.Size([94, 2, 128]) tensor([94, 60])
            context, attention = self.attention(query, key, values, reduced_length) # here value is context
            # print('----------------2', attention)
            all_attention.append(attention)
            #(self, query, key, value, lens)
            #prediction = self.character_prob(torch.cat([output, values[i,:,:]], dim=1))
            output = self.fc(torch.cat([output, context], dim=1))
            output = self.tanh(output)
            prediction = self.character_prob(output)
            # print('one time prediction shape', prediction.shape)
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)#, np.array(attention)[:, 0, :reduced_length[0]]

def init_weights(m):
    if type(m) == nn.LSTM:
        nn.init.orthogonal_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)
    if type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        #m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight)

class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.encoder.apply(init_weights)
        self.decoder = Decoder(vocab_size, hidden_dim*2, isAttended=True)
        self.decoder.apply(init_weights)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True):
        reduced_length, hidden, key, value = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, reduced_length, text_input, hidden=None)
        else:
            predictions = self.decoder(key, value, reduced_length, text=None, isTrain=False,hidden=None)
        return predictions


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(device)
    start = time.time()
    eval_loss = []
    for batchIdx, (batchx, batchx_len, batchy, batchy_len) in enumerate(train_loader):
#        print('----------batch--------', batchx, batchy)
        
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        batchx, batchx_len, batchy, batchy_len = batchx.permute(1, 0, 2).to(device),batchx_len.to(device), batchy.long().to(device), batchy_len.to(device)
        predictions = model(batchx, batchx_len, batchy)
        # if batchIdx == 1:
        #    print(greedy_search([predictions], index2letter))
        #print('--------predict', greedy_search([predictions], index2letter))
        #print('--------truth', batchy)
#        print('--------predict', predictions)
#        print('--------truth', batchy)
        # mask = torch.arange(predictions.size(1)).repeat(batchy_len.size(0),1).to(device) >= batchy_len.contiguous().view(-1,1).long()
        mask = torch.arange(batchy.size(1)).unsqueeze(0).to(device) >= batchy_len.unsqueeze(1)
        # print(mask)
        predictions = predictions.permute(0,2,1)
        loss = criterion(predictions, batchy)
        loss.masked_fill_(mask, 0.0)
#        print('-------loss',loss.shape, torch.sum(loss) / batchy_len.sum())
#        print()
        loss_eval = loss.clone()
        loss = torch.sum(loss) / batchy_len.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        eval_loss.append(torch.exp(torch.sum(loss_eval) / torch.sum((~mask).int())))
        #if batchIdx == 1:
        if device == 'cuda': 
            torch.cuda.empty_cache() 
            del batchx
            del batchx_len
            del batchy
            del batchy_len
            del predictions
            del loss
            del mask
        end = time.time()
    print('------train loss in epoch {} is------'.format(epoch), sum(eval_loss) / (batchIdx+1))

def val(model, val_loader, criterion, epoch):
    with torch.no_grad():
        model.eval()
        model.to(device)
        start = time.time()
        eval_loss = []
        for batchIdx, (batchx, batchx_len, batchy, batchy_len) in enumerate(val_loader):
            
            torch.autograd.set_detect_anomaly(True)
            batchx, batchx_len, batchy, batchy_len = batchx.permute(1, 0, 2).to(device),batchx_len.to(device), batchy.long().to(device), batchy_len.to(device)
            predictions = model(batchx, batchx_len, batchy)
            mask = torch.arange(predictions.size(1)).repeat(batchy_len.size(0),1).to(device) >= batchy_len.contiguous().view(-1,1).long()
            mask = mask.to(device)
            predictions = predictions.permute(0,2,1)
            loss = criterion(predictions, batchy)
            loss.masked_fill_(mask, 0)
            loss_eval = loss.clone()
            loss = torch.sum(loss) / batchy_len.sum()
            eval_loss.append(torch.exp(torch.sum(loss_eval) / torch.sum((~mask).int())))
            if device == 'cuda':
                torch.cuda.empty_cache() 
                del batchx
                del batchx_len
                del batchy
                del batchy_len
                del predictions
                del loss
                del mask
            #if batchIdx == 1:
            #    break
        end = time.time()
        VALLOSS.append(sum(eval_loss) / (batchIdx+1))
        print('------validation loss in epoch {} is------'.format(epoch), sum(eval_loss) / (batchIdx+1))

def test(model, test_loader):
    with torch.no_grad():
        model.eval()
        model.to(device)
        start = time.time()
        predict_list = []
        for batchIdx, (batchx, batchx_len) in enumerate(test_loader):
            # print('---------------------', batchIdx, batchx.shape)
            batchx, batchx_len = batchx.to(device),batchx_len.to(device)
            predictions = model(batchx, batchx_len, isTrain=False)

            # print('------------',  predictions.shape)
            predict_list.append(predictions)
            #if batchIdx == 1:
            #    break
            if device == 'cuda':
                torch.cuda.empty_cache() 
                del batchx
                del batchx_len
                del batchy
                del batchy_len
                del predictions
            if batchIdx == 3:
                break
        end = time.time()
        return predict_list

def distance(s1, s2):
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def greedy_search(batchli, idx2chr): #(B, L, C)
    #batchli = [i.detach().numpy() for i in batchli]
    out = []
    for probs in batchli:
    #    print('aaaaa', probs.shape)
        for prob in probs:
    #        print('bbbb', prob.shape)
            s = []
            for step in prob:
    #            print('cccc', step)
                #             idx = torch.multinomial(step, 1)[0]
                idx = torch.argmax(step)
                c = idx2chr[int(idx)]
    #            print('------------', c)
                if c == '<eos>':
                    break
                s.append(c)
    #        print(s)
            out.append("".join(s))        
    return out

def process_output(predict_list):
    for li in predict_list:
        [2, 250, 35]

def sampler(inputx, tau = 1.0, temperature = 10):
    noise = torch.rand(inputx.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = Variable(noise).to(device)
    x = (inputx + noise) / tau + temperature
    x = F.softmax(x.view(inputx.size(0), -1))
    return x.view_as(inputx)


LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=256)
    #vmodel.load_state_dict(torch.load(modelpath))
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='none')
    nepochs = EPOCH
    batch_size = 64 if device == 'cuda' else 2
    if device == 'cuda': 
        torch.cuda.empty_cache() 

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, letter2index)
    character_text_valid = transform_letter_to_index(transcript_valid, letter2index)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_test)

    for epoch in range(nepochs):
        train(model, train_loader, criterion, optimizer, epoch)
        old_val_loss = VALLOSS[-1]
        
        val(model, val_loader, criterion, epoch)
        new_val_loss = VALLOSS[-1]
        if new_val_loss < old_val_loss:
            torch.save(model.state_dict(), PARPATH.format(epoch))
            
    prediction = test(model, train_loader)
    #print('len---------prediction----------', len(prediction))
    #print('shape---------prediction----------', prediction[0].shape)
#    print(prediction)
    return prediction

if __name__ == '__main__':
    A = main()

    k = greedy_search(A, index2letter)
    #print(k)
    b = zip([i for i in range(len(k))],k)
    #print(b)
    c = np.array(tuple(b))
    df = pd.DataFrame(c, columns=['id', 'Predicted'])
    df.to_csv(OUTPATH, index = False)
