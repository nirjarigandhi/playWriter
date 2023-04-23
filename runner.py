from Decode_only_transformer import *

from random import shuffle
from matplotlib import pyplot as plt
import gc

print(f'{torch.cuda.is_available()}')
torch.cuda.empty_cache()

data_dictionary = torch.load('/home/vijay/Documents/413/playWriter/dataset/data.txt')

train_dataset = torch.Tensor(data_dictionary['train'])
test_dataset = torch.Tensor(data_dictionary['test'])
valid_dataset = torch.Tensor(data_dictionary['valid'])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Training", flush=True)
# x = train_dataset[0:50, :49, :].to(device)
# y = train_dataset[0:50, 49:, :].to(device)
# transformer = Transformer(6, 8, 512, 50, 1000, 6, 8, 8, 512, 50, 50, 10086, None).to(device)
# loss_func = nn.CrossEntropyLoss()

def shuffle_dataset(dataset: torch.Tensor):
    """Shuffle the data set batch wise when in the format (batch, sequence, embedding)"""
    batch_size = dataset.shape[0]

    return dataset[torch.randperm(batch_size)]


def split_dataset(dataset, batch_size):
    # Split the dataset into batches
    # Return the batches
    return torch.split(dataset, batch_size, 0)

def split_batch_answers(dataset, sentence_length: int):
    """DataSet must be one member of the batch"""
    answers = dataset[:, 1:,:]
    tests = dataset[:, :sentence_length - 1, :]
    return tests, answers

mask = torch.ones((49, 49))
mask[:,:] = -float("inf")
mask = torch.triu(mask, 1) # Don't Look ahead. This needs to be the size of (seq, seq) because of the attention matrix inside the folders
mask = mask.to(device)
 

model = DecodeTransformer(3, 8, 10086, 768, 70, 1000, mask).to(device)


# y = split_dataset(train_dataset, 50)
# first = y[0]
# tests, answers = split_batch_answers(first, 50)
# tests = tests.to(device)
# answers = answers.to(device)
def learn(x):
  return pow(512, -0.5) * min(pow(x+ 0.0001, -0.5), x * pow(4000, -0.5))

# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=1, betas=(0.9, 0.98), eps=pow(10, -9))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,learn)

# for i in range(10000):
#     results = model.forward(tests)
#     pre_logits= model.pre_softmax
#     loss = loss_func(pre_logits.reshape((pre_logits.shape[0] * pre_logits.shape[1], pre_logits.shape[2])).float(), torch.argmax(answers, 2).flatten())
#     #loss.retain_grad()
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     scheduler.step()
#     print(f'the loss is {loss.item()}')



def train(epoch: int, batches: int , train_data: torch.Tensor, valid_data: torch.Tensor, learn: callable, device, model: DecodeTransformer, offset: int = 0, list_of_accuracy = [], list_of_vaccuracy =[], list_of_epoch=[], list_of_loss =[]):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=1, betas=(0.9, 0.98), eps=pow(10, -9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,learn)
    list_of_accuracy = list_of_accuracy
    list_of_vaccuracy = list_of_vaccuracy
    list_of_epoch = list_of_epoch
    list_of_loss = list_of_loss
    for i in range(offset, epoch + offset):
        shuffle = shuffle_dataset(train_data)

        batch_elements= split_dataset(shuffle, batches)
        for j in range(len(batch_elements)):
            train_test, train_answers = split_batch_answers(batch_elements[j], 50)
            train_test = train_test.to(device)
            train_answers = train_answers.to(device)
            results = model.forward(train_test)
            loss = loss_func(model.pre_logits.reshape((model.pre_logits.shape[0] * model.pre_logits.shape[1], model.pre_logits.shape[2])).float(), torch.argmax(train_answers, 2).flatten())
            #loss.retain_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            del train_test
            del train_answers
            torch.cuda.empty_cache()
            print(f"The loss is {loss.item()}")
        
        if i % 10 == 0:
            tacc = get_accuracy(model, train_data, device)
            vacc = get_accuracy(model, valid_dataset, device)
            list_of_accuracy.append(tacc)
            list_of_vaccuracy.append(vacc)
            list_of_epoch.append(i)
            list_of_loss.append(loss.item())
            torch.save({"epoch": list_of_epoch, "train_acc": list_of_accuracy, "valid_acc": list_of_vaccuracy, "list_of_loss": list_of_loss}, f'/home/vijay/Documents/413/playWriter/training_saves/saves_({i}).txt')
            print(f"The loss is {loss.item()},  [Train Acc {tacc}%], [Valid Acc {vacc}%]")

        if i % 100 == 0:
            torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss.item()}, f'/home/vijay/Documents/413/playWriter/checkpoint/states_({i}).txt')
    
    return list_of_accuracy, list_of_vaccuracy, list_of_loss, list_of_epoch


def get_accuracy(model: DecodeTransformer, train_dataset: torch.Tensor, device):
    set = train_dataset.clone()
    tests, answers = split_batch_answers(set, 50)
    tests = tests.to(device)
    argmax_answers = torch.argmax(answers, 2).flatten()
    argmax_answers = argmax_answers.to(device)
    results = model.forward(tests)
    argmax_results = torch.argmax(results, 2).flatten()

    hold = list(argmax_answers == argmax_results)

    del tests
    del argmax_answers
    del results
    del argmax_results
    gc.collect()
    torch.cuda.empty_cache()

    return (hold.count(True) / len(hold)) * 100



    
list_of_accuracy, list_of_vaccuracy, list_of_loss, list_of_epoch = train(301, 50, train_dataset, valid_dataset, learn, device, model)







    


