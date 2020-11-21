#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation


# In[42]:


seq_len = 80
batch_size = 256
embed_size = 400
hidden_size = 1150
nlayers = 3
lr = 0.1
weight_decay = 1e-5


# In[4]:


# load all that we need

dataset = np.load('../dataset/wiki.train.npy', allow_pickle=True)
fixtures_pred = np.load('../fixtures/prediction.npz', allow_pickle=True)  # dev
fixtures_gen = np.load('../fixtures/generation.npy', allow_pickle=True)  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz', allow_pickle=True)  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy', allow_pickle=True)  # test
vocab = np.load('../dataset/vocab.npy', allow_pickle=True)


# In[29]:


len(vocab)


# In[431]:


# data loader

class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        
    def __iter__(self):
        index = 0
        np.random.shuffle(self.dataset)
        self.text = np.concatenate(self.dataset, axis=0)
        n_seq = len(self.text) // self.seq_len
        self.text = self.text[:n_seq * self.seq_len]
        self.data = torch.tensor(self.text).view(-1,seq_len)
        self.datalen = self.data.size(0)
        while index < self.datalen:
            if index + self.batch_size < self.datalen:
                yield self.data[index:index + self.batch_size, :-1].long(), self.data[index:index + self.batch_size, 1:].long()
            else:
                yield self.data[index:, :-1], self.data[index:, 1:].long()      


# In[ ]:


class 


# In[432]:


# model

class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """
    def __init__(self, vocab_size,embed_size,hidden_size, nlayers):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers=nlayers
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.LSTM(input_size = embed_size,hidden_size=hidden_size,num_layers=nlayers)
        self.scoring = nn.Linear(hidden_size,vocab_size)

    def forward(self, x): # inputshape is (N, L)
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        batch_size = x.size(0)
        out = x.permute(1, 0)
        out = self.embedding(out)
        hidden = None
        out,hidden = self.rnn(out,hidden)
        out = out.view(-1, self.hidden_size)
        out = self.scoring(out)
        out = out.view(-1, batch_size,self.vocab_size)
        return out#.permute(1,0,2)


# In[433]:


# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            
            targets = targets.permute(1,0)
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        self.optimizer.zero_grad()
        outputs = self.model(inputs.long())
        loss = self.criterion(outputs.reshape(-1,outputs.size(2)),targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.025)
        self.optimizer.step()
        return loss.item()
    
    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


# In[434]:


class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here
            
            :param inp:
            :return: a np.ndarray of logits
        """
        result = model(torch.from_numpy(inp).long())
        return result[-1,:,:].float().detach().numpy()

        
    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """   
        generated_words = []
        scores = model(torch.from_numpy(inp).long()) # 1 x V
        scores = scores.permute(1, 0, 2)
        _,current_word = torch.max(scores[:, -1, :],dim=1) # N x 1 x V -> N x 1 x 1
        generated_words.append(current_word)
        if forward > 1:
            for i in range(forward-1):
                scores = model(current_word.long().unsqueeze(1))
                scores = scores.permute(1, 0, 2)
                _,current_word = torch.max(scores[:, -1, :],dim=1) # 1
                generated_words.append(current_word)
        return torch.stack(generated_words).transpose(1,0)
        


# In[435]:


# TODO: define other hyperparameters here

NUM_EPOCHS = 5
BATCH_SIZE = 40


# In[436]:


run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)


# In[437]:


model = LanguageModel(len(vocab), embed_size, hidden_size, nlayers)
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


# In[438]:


best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch "+str(epoch)+" with NLL: "+ str(best_nll))
        trainer.save()
    


# In[ ]:


# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()


# In[ ]:


# see generated output
print (trainer.generated[-1]) # get last generated output


# In[ ]:





# In[ ]:




