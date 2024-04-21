import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_para, model_eval_sts

import math

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

MAX_DATA = 9000

class MNR(nn.Module):

    def __init__(self):
        super(MNR, self).__init__()

    def forward(self, seq_a, seq_b, label):
 
        matrix = torch.zeros(seq_a.shape[0], seq_b.shape[0])


        loss_sum = torch.zeros(1)

        for i in range(seq_a.shape[0]):
            for j in range(seq_b.shape[0]):
                matrix[i, j] = F.cosine_similarity(seq_a[i, :], seq_b[j, :], dim=0)

        for i in range(seq_a.shape[0]):
            loss_sum += F.binary_cross_entropy_with_logits(matrix[i][i].float(), label[i].float(), reduction="sum")

        #loss_sum = F.cross_entropy(matrix, label, reduction="sum")

        loss = torch.tensor([loss_sum.item()], requires_grad=True)

        return loss

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.sentiment = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.paraphrase = torch.nn.Linear(BERT_HIDDEN_SIZE, 2)
        self.similarity = torch.nn.Linear(BERT_HIDDEN_SIZE, 8)
        #raise NotImplementedError


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        output = self.bert(input_ids, attention_mask)
        output = output['pooler_output']
        return output
        raise NotImplementedError


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        outputs = self.bert(input_ids, attention_mask)
        outputs = outputs['pooler_output']
        final = self.sentiment(outputs)
        return final
        raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO

        """
        output1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']
        output2 = self.bert(input_ids_2, attention_mask_2)['pooler_output']
        
        output = torch.empty(output1.shape[0], output1.shape[1])
        for i in range(0, output1.shape[0]):
            for j in range(0, output1.shape[1]):
                output[i][j] = math.sqrt(output1[i][j]*output1[i][j] + output2[i][j]*output2[i][j])
        
        output = self.paraphrase(output)
        return output
        """

        output1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']
        output2 = self.bert(input_ids_2, attention_mask_2)['pooler_output']
        
        #output = torch.cat([output1,output2],dim=-1)

        #output = self.paraphrase(output).squeeze(1)

        #print(output)
        #print(output.sigmoid())

        output = F.cosine_similarity(output1, output2)
        return output

        raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        output1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']
        output2 = self.bert(input_ids_2, attention_mask_2)['pooler_output']

        """
        #distance method
        output = torch.empty(output1.shape[0], output1.shape[1])
        for i in range(0, output1.shape[0]):
            for j in range(0, output1.shape[1]):
                output[i][j] = math.sqrt(output1[i][j]*output1[i][j] + output2[i][j]*output2[i][j])
        
        output = self.similarity(output)
        
        """
        #cosine method
        output = F.cosine_similarity(output1, output2)
        output = (output + 1) * 2.5
        

        #print(output.shape)
        return output
        raise NotImplementedError

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-sst-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            print(logits)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            print(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    
    best_dev_acc = 0
    for epoch in range(args.epochs):
        
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}',disable=TQDM_DISABLE):
            b_ids1, b_ids2, b_mask1, b_mask2, b_labels = (batch['token_ids_1'], batch['token_ids_2'], batch['attention_mask_1'], batch['attention_mask_2'], batch['labels'])
            b_ids1 = b_ids1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask1 = b_mask1.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = F.binary_cross_entropy_with_logits(logits, b_labels.float().view(-1), reduction='sum') / args.batch_size
            loss = torch.tensor([loss.item()], requires_grad=True)

            loss.backward()
            optimizer.step()
            num_batches += 1
            train_loss += loss.item()
        train_loss = train_loss / (num_batches)
        
        train_acc, *_ = model_eval_para(para_train_dataloader, model, device)
        dev_acc, *_ = model_eval_para(para_dev_dataloader, model, device)

        if best_dev_acc < dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")
    
    best_dev_acc = 0
    
    for epoch in range(args.epochs):
        
        model.train()
        num_batches = 0
        train_loss = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}',disable=TQDM_DISABLE):
            b_ids1, b_ids2, b_mask1, b_mask2, b_labels = (batch['token_ids_1'], batch['token_ids_2'], batch['attention_mask_1'], batch['attention_mask_2'], batch['labels'])
            b_ids1 = b_ids1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask1 = b_mask1.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = F.mse_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss = torch.tensor([loss.item()], requires_grad=True)

            loss.backward()
            optimizer.step()
            num_batches += 1
            train_loss += loss.item()

        train_loss = train_loss / (num_batches)
        
        train_acc, *_ = model_eval_sts(sts_train_dataloader, model, device)
        dev_acc, *_ = model_eval_para(sts_dev_dataloader, model, device)

        if best_dev_acc < dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)


        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)

def multitask_testing(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)


    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    
    sst_dev_data_set = SentenceClassificationDataset(sst_dev_data, args)

    para_dev_data_portion = para_dev_data[:64]

    para_dev_data_set = SentencePairDataset(para_dev_data_portion, args)
    sts_dev_data_set = SentencePairDataset(sts_dev_data, args)
    sst_dev_dataloader = DataLoader(sst_dev_data_set, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data_set.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data_set, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data_set.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data_set, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data_set.collate_fn)

    for epoch in range(args.epochs):
        model.train()

        if args.max_data > -1:
            sst_train_len = len(sst_train_data)
            para_train_len = len(para_train_data)
            sts_train_len = len(sts_train_data)

            if sst_train_len > args.max_data:
                sst_part_prob = random.randrange(0,sst_train_len-args.max_data)
                sst_train_data_portion = sst_train_data[sst_part_prob:sst_part_prob+args.max_data]
            else:
                sst_train_data_portion = sst_train_data[:]

            if para_train_len > args.max_data:
                para_part_prob = random.randrange(0,para_train_len-args.max_data)
                para_train_data_portion = para_train_data[para_part_prob:para_part_prob+args.max_data]
            else:
                para_train_data_portion = para_train_data[:]
                para_dev_data_portion = para_dev_data[:]


            if sts_train_len > args.max_data:
                sts_part_prob = random.randrange(0,sts_train_len-args.max_data)
                sts_train_data_portion = sts_train_data[sts_part_prob:sts_part_prob+args.max_data]
            else:
                sts_train_data_portion = sts_train_data[:]
            
            
            

        sst_train_data_set = SentenceClassificationDataset(sst_train_data_portion, args)
        

        para_train_data_set = SentencePairDataset(para_train_data_portion, args)
        

        sts_train_data_set = SentencePairDataset(sts_train_data_portion, args)
        

        sst_train_dataloader = DataLoader(sst_train_data_set, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data_set.collate_fn)
        
    
        para_train_dataloader = DataLoader(para_train_data_set, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data_set.collate_fn)
        

        sts_train_dataloader = DataLoader(sts_train_data_set, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data_set.collate_fn)
        
        train_loss = 0
        batches = 0
        sst_train_dataloader_it = iter(sst_train_dataloader)
        para_train_dataloader_it = iter(para_train_dataloader)
        sts_train_dataloader_it = iter(sts_train_dataloader)
        batch_num = int(args.max_data/args.batch_size)

        tot_sst_loss = 0
        tot_para_loss = 0
        tot_sts_loss = 0

        loss_mnr_fn = MNR()

        for batch_num in tqdm(range(batch_num), desc=f'train-{epoch}', disable=TQDM_DISABLE):

            try:
                nextSST = sst_train_dataloader_it.__next__()
            except:
                sst_train_dataloader_it = iter(sst_train_dataloader)
                nextSST = sst_train_dataloader_it.__next__()

            try: 
                nextPara = para_train_dataloader_it.__next__()
            except:
                para_train_dataloader_it = iter(para_train_dataloader)
                nextPara = para_train_dataloader_it.__next__()

            try:
                nextSTS = sts_train_dataloader_it.__next__()
            except:
                sts_train_dataloader_it = iter(sts_train_dataloader)
                nextSTS = sts_train_dataloader_it.__next__()

            b_sst_ids, b_sst_mask, b_sst_labels = (nextSST['token_ids'],
                                       nextSST['attention_mask'], nextSST['labels'])
            b_para_ids1, b_para_ids2, b_para_mask1, b_para_mask2, b_para_labels = (nextPara['token_ids_1'], nextPara['token_ids_2'], nextPara['attention_mask_1'], nextPara['attention_mask_2'], nextPara['labels'])
            b_sts_ids1, b_sts_ids2, b_sts_mask1, b_sts_mask2, b_sts_labels = (nextSTS['token_ids_1'], nextSTS['token_ids_2'], nextSTS['attention_mask_1'], nextSTS['attention_mask_2'], nextSTS['labels'])

            b_sst_ids.to(device)
            b_sst_mask.to(device)
            b_sst_labels.to(device)
            b_para_ids1.to(device)
            b_para_ids2.to(device)
            b_para_mask1.to(device)
            b_para_mask2.to(device)
            b_para_labels.to(device)
            b_sts_ids1.to(device)
            b_sts_ids2.to(device)
            b_sts_mask1.to(device)
            b_sts_mask2.to(device)
            b_sts_labels.to(device)

            optimizer.zero_grad()

            sstLogit = model.predict_sentiment(b_sst_ids,b_sst_mask)
            paraLogit = model.predict_paraphrase(b_para_ids1, b_para_mask1, b_para_ids2, b_para_mask2)
            stsLogit = model.predict_similarity(b_sts_ids1, b_sts_mask1, b_sts_ids2, b_sts_mask2)

            sstLoss = F.cross_entropy(sstLogit, b_sst_labels.view(-1), reduction='sum') / args.batch_size
            paraLoss = F.binary_cross_entropy_with_logits(paraLogit, b_para_labels.float().view(-1),reduction='sum') / args.batch_size
            stsLoss = F.mse_loss(stsLogit, b_sts_labels.view(-1),reduction='sum') / args.batch_size

            #print(paraLogit.sigmoid().round().flatten())
            #print(b_para_labels)
            #print(stsLogit.flatten())
            #print(b_sts_labels)

            paraMNRLoss = loss_mnr_fn(model(b_para_ids1,b_para_mask1),model(b_para_ids2,b_para_mask2), b_para_labels.view(-1))/args.batch_size
            stsMNRLoss = loss_mnr_fn(model(b_sts_ids1,b_sts_mask1),model(b_sts_ids2,b_sts_mask2), b_sts_labels.view(-1))/args.batch_size



            loss = (sstLoss + paraLoss + stsLoss + paraMNRLoss + stsMNRLoss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tot_sst_loss += sstLoss.item()
            tot_para_loss += paraLoss.item()
            tot_sts_loss += stsLoss.item()
            batches+=1
        
        train_loss = train_loss/batches/3

        sst_train_acc, *_ = model_eval_sst(sst_train_dataloader, model, device)
        sst_dev_acc, *_ = model_eval_sst(sst_dev_dataloader,model,device)
        para_train_acc, *_ = model_eval_para(para_train_dataloader,model,device)
        para_dev_acc, *_ = model_eval_para(para_dev_dataloader,model,device)
        sts_train_acc, *_ = model_eval_sts(sts_train_dataloader,model,device)
        sts_dev_acc, *_ = model_eval_sts(sts_dev_dataloader,model,device)

        #para_train_acc = 0
        #para_dev_acc = 0

        print(f"General Acc : SST :: {sst_train_acc :.3f} : PARA :: {para_train_acc :.3f} : STS :: {sts_train_acc :.3f}")

        avg_dev_acc = (sst_dev_acc+para_dev_acc+sts_dev_acc)/3
        print(f"Dev Acc : SST :: {sst_dev_acc :.3f} : PARA :: {para_dev_acc :.3f} : STS :: {sts_dev_acc :.3f} : AVG :: {avg_dev_acc :.3f}")

        if avg_dev_acc > best_dev_acc:
            best_dev_acc = avg_dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f} sst loss :: {tot_sst_loss/batches :.3f} para loss :: {tot_para_loss/batches :.3f} sts loss :: {tot_sts_loss/batches :.3f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--max_data", type=int, default=6016)
    parser.add_argument("--mode", type=str, choices=("multitask_finetune", "normal"), default="normal")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    #test_sts(args)
    if args.mode == "multitask_finetune":
        multitask_testing(args)
    else:
        train_multitask(args)
    #test_model(args)
