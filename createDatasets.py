from Transformer import *

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
print("Training", flush=True)
# x = train_dataset[0:50, :49, :].to(device)
# y = train_dataset[0:50, 49:, :].to(device)
transformer = Transformer(6, 8, 512, 50, 1000, 6, 8, 8, 512, 50, 50, 10086, None).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(recurse=True))
optimizer.zero_grad()
# for i in range(2000):
#     de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), 10086).unsqueeze(0) * torch.ones((50, 1, 10086)) #Make a tensor of shape (batch_size, 1, 10086)
#     results = transformer.forward(x, de_in.to(device))
#     #print(f'Shape of output {(transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()).shape}, shape of y {torch.argmax(y.squeeze(1), 1).shape} ', flush=True)

#     #loss = loss_func(torch.argmax(y.squeeze(1), 1).float(), (transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()))
#     loss = loss_func((transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()), torch.argmax(y.squeeze(1), 1).long() )
#     loss.retain_grad()
#     loss.backward()
#     print(loss.item())
#     optimizer.step()
#     optimizer.zero_grad()



def shuffle_dataset(dataset: torch.Tensor):
    """Shuffle the data set batch wise when in the format (batch, sequence, embedding)"""
    batch_size = dataset.shape[0]

    return dataset[torch.randperm(batch_size)]


def split_dataset(dataset, batch_size):
    # Split the dataset into batches
    # Return the batches
    return torch.split(dataset, batch_size, 0)

def split_batch_encoder_and_answers(batch):
    # use teaching forcing to split the dataset into batches
    # Return the batches
    batch_encoder = []
    batch_decoder = []
    for i in range(1, batch.shape[1]):
        batch_encoder.append(batch[:,0:i,:])
        batch_decoder.append(batch[:,i,:].unsqueeze(1))
    
    return batch_encoder, batch_decoder

epoch = 10

def train(epoch: int, dataset: torch.Tensor, batch_size: int, model, device, start_token: int, raw_embedding_size: int):
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(recurse=True))
    de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), raw_embedding_size).unsqueeze(0) * torch.ones((batch_size, 1, raw_embedding_size)) #Make a tensor of shape (batch_size, 1, 10086) start token for the decoder
    de_in = de_in.to(device)
    optimizer.zero_grad()
    list_of_loss = []
    for i in range(epoch):
        data = shuffle_dataset(dataset).to(device) #shuffle the data
        batches = list(split_dataset(data, batch_size)) # returns a list of tensor objects [batch_size, sentence_length, embedding_size]
        batches.pop() #remove the last uneven one
        for batch in batches:
            batch_encoder, batch_answers = split_batch_encoder_and_answers(batch)
            for i in range(len(batch_encoder)):
                results = model.forward(batch_encoder[i], de_in)
                loss = loss_func((model.pre_logit.reshape((model.pre_logit.shape[0] * model.pre_logit.shape[1], model.pre_logit.shape[2])).float()), torch.argmax(batch_answers[i].squeeze(1), 1).long() )
                loss.retain_grad()
                loss.backward()
                print(loss.item())
                optimizer.step()
                optimizer.zero_grad()
    

    return list_of_loss



