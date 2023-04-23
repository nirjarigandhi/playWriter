from Decode_only_transformer import *



import gc

print(f'{torch.cuda.is_available()}')
torch.cuda.empty_cache()
def createDataset(filenames, dataset_size=50):
    
        # Create a dataset from a file
    
        # Open the file
        main_dataset = []
        curr_dataset = []
        unique_words = set()
        for filename in filenames:
            f = open(filename, 'r')
            for line in f:
                # Read the line
                # Split the line into a list
                line = line.strip().lower()
                line = line.split(' ')
                for word in line:
                    if len(curr_dataset) == dataset_size:
                        main_dataset.append(curr_dataset)
                        curr_dataset = []
                    if word != '':
                        curr_dataset.append(word)
                        unique_words.add(word)
                # Create a dataset from the list
                # Add the dataset to the list
            f.close()
        ## Add decoder <start> 
        unique_words.add("<start>")
        main_dataset.append(curr_dataset)
        return main_dataset, unique_words

def createOneHotDataset(dataset, unique_words):
    # Create a one hot dataset from a dataset
    # Get the unique words
    # Create a one hot dataset
    # Return the one hot dataset
    unique_words = list(unique_words)
    one_hot_dataset = []
    for data in dataset:
        one_hot_data = []
        for word in data:
            one_hot_word = [0] * len(unique_words)
            one_hot_word[unique_words.index(word)] = 1
            one_hot_data.append(one_hot_word)
        one_hot_dataset.append(one_hot_data)
    return one_hot_dataset

dataset1, unique_words = createDataset(['play1.txt', 'play2.txt'])
unique_words = list(unique_words)
start_token = unique_words.index("<start>")
dataset1 = createOneHotDataset(dataset1, unique_words)
train_len = int(len(dataset1) * 0.8)
train_dataset = dataset1[:train_len]
test_len = int(len(dataset1) * 0.1)
test_dataset = dataset1[train_len:test_len+train_len]
valid_dataset = dataset1[test_len+train_len:]

train_dataset = torch.Tensor(train_dataset)
valid_dataset.pop()
test_dataset = torch.Tensor(test_dataset)
valid_dataset = torch.Tensor(valid_dataset)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Training", flush=True)
# x = train_dataset[0:50, :49, :].to(device)
# y = train_dataset[0:50, 49:, :].to(device)
# transformer = Transformer(6, 8, 512, 50, 1000, 6, 8, 8, 512, 50, 50, 10086, None).to(device)
loss_func = nn.CrossEntropyLoss()

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



def train(epoch: int, batches: int , train_data: torch.Tensor, valid_data: torch.Tensor, learn: callable, device, model: DecodeTransformer):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=1, betas=(0.9, 0.98), eps=pow(10, -9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,learn)
    for i in range(epoch):
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
        torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss.item()}, f'/home/vijay/Documents/413/playWriter/checkpoint/things_({i}).txt')

    
train(20, 50, train_dataset, valid_dataset, learn, device, model)






    



