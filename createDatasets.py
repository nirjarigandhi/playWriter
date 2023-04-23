from Transformer import *


from random import shuffle
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
dataset1.pop()
shuffle(dataset1)
train_len = int(len(dataset1) * 0.8)
train_dataset = dataset1[:train_len]
test_len = int(len(dataset1) * 0.1)
test_dataset = dataset1[train_len:test_len+train_len]
valid_dataset = dataset1[test_len+train_len:]

train_dataset = torch.Tensor(train_dataset)
test_dataset = torch.Tensor(test_dataset)
valid_dataset = torch.Tensor(valid_dataset)

torch.save({'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset, 'start_token': start_token, "unique_words_list": unique_words, 'train_len': train_len, 'test_len': test_len}, '/home/vijay/Documents/413/playWriter/dataset/data.txt')



# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Training", flush=True)
# x = train_dataset[0:50, :49, :].to(device)
# y = train_dataset[0:50, 49:, :].to(device)
# transformer = Transformer(6, 8, 512, 50, 1000, 6, 8, 8, 512, 50, 50, 10086, None).to(device)
# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(transformer.parameters(recurse=True))
# optimizer.zero_grad()
# for i in range(2000):
#     de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), 10086).unsqueeze(0) * torch.ones((50, 1, 10086)) #Make a tensor of shape (batch_size, 1, 10086)
#     results = transformer.forward(x, de_in.to(device))
# #     #print(f'Shape of output {(transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()).shape}, shape of y {torch.argmax(y.squeeze(1), 1).shape} ', flush=True)

# #     #loss = loss_func(torch.argmax(y.squeeze(1), 1).float(), (transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()))
#     loss = loss_func((transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()), torch.argmax(y.squeeze(1), 1).long() )
#     loss.retain_grad()
#     loss.backward()
#     print(loss.item())
#     optimizer.step()
#     optimizer.zero_grad()



# def shuffle_dataset(dataset: torch.Tensor):
#     """Shuffle the data set batch wise when in the format (batch, sequence, embedding)"""
#     batch_size = dataset.shape[0]

#     return dataset[torch.randperm(batch_size)]


# def split_dataset(dataset, batch_size):
#     # Split the dataset into batches
#     # Return the batches
#     return torch.split(dataset, batch_size, 0)

# def split_batch_encoder_and_answers(batch):
#     # use teaching forcing to split the dataset into batches
#     # Return the batches
#     batch_encoder = []
#     batch_decoder = []
#     for i in range(1, batch.shape[1]):
#         batch_encoder.append(batch[:,0:i,:])
#         batch_decoder.append(batch[:,i,:].unsqueeze(1))
    
#     return batch_encoder, batch_decoder


# def split_differnently(batch, encoder_context_amount):
#     encoder_amount_prompts = batch[:, 0:encoder_context_amount, :] # the sentence length given to the encoder is 5 for positional encoding
#     answers = []
#     for i in range(encoder_context_amount, batch.shape[1]):
#         answers.append(batch[:, i, :].unsqueeze(1))

#     return encoder_amount_prompts, answers



# def get_accuracy(data_set, model, raw_embedding_size, start_token):
#   count_true = 0
#   total = 0
#   shuffled = shuffle_dataset(data_set).to(device)
#   encoder_inputs, expected_answers = split_batch_encoder_and_answers(shuffled)
#   de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), raw_embedding_size).unsqueeze(0) * torch.ones((data_set.shape[0], 1, raw_embedding_size))
#   de_in = de_in.to(device)
#   for i in range(len(encoder_inputs)):
#     results = model.forward(encoder_inputs[i], de_in)
#     collapsed1 = torch.argmax(results, 2)
#     collapsed2 = torch.argmax(expected_answers[i], 2)
    
#     intermediate = collapsed1 == collapsed2
#     intermediate = list(intermediate.flatten())
#     count_true += intermediate.count(True)
#     total += len(intermediate)

  
#   return count_true / total


# def new_accuracy(data_set, model, raw_embedding_size, start_token):
#     count_true = 0
#     total = 0
#     shuffled = shuffle_dataset(data_set).to(device)
#     encoder_inputs, expected_answers = split_differnently(shuffled, 45)
#     de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), raw_embedding_size).unsqueeze(0) * torch.ones((data_set.shape[0], 1, raw_embedding_size))
#     de_in = de_in.to(device)
#     for i in range(len(expected_answers)):
#         results = model.forward(encoder_inputs, de_in)[:,-1,:].unsqueeze(1)
#         de_in = torch.concat((de_in, expected_answers[i]), 1)
#         collapsed1 = torch.argmax(results, 2)
#         collapsed2 = torch.argmax(expected_answers[i], 2)

#         intermediate = collapsed1 == collapsed2
#         intermediate = list(intermediate.flatten())
#         count_true += intermediate.count(True)
#         total += len(intermediate)
    
#     return count_true / total

    

# def learn(x):
#   return pow(512, -0.5) * min(pow(x+ 0.0001, -0.5), x * pow(4000, -0.5))

# def train(epoch: int, dataset: torch.Tensor, va_dataset: torch.Tensor, batch_size: int, model, device, start_token: int, raw_embedding_size: int):
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=1, betas=(0.9, 0.98), eps=pow(10, -9))
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,learn)
#     de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), raw_embedding_size).unsqueeze(0) * torch.ones((batch_size, 1, raw_embedding_size)) #Make a tensor of shape (batch_size, 1, 10086) start token for the decoder
#     de_in = de_in.to(device)
#     optimizer.zero_grad()
#     list_of_train_accuracy = []
#     list_of_loss = []
#     list_va = []
#     for i in range(epoch):
#         data = shuffle_dataset(dataset).to(device) #shuffle the data
#         batches = list(split_dataset(data, batch_size)) # returns a list of tensor objects [batch_size, sentence_length, embedding_size]
#         batches.pop() #remove the last uneven one
#         for batch in batches:
#             batch_encoder, batch_answers = split_batch_encoder_and_answers(batch)
#             for i in range(len(batch_encoder)):
#                 results = model.forward(batch_encoder[i], de_in)
#                 loss = loss_func((model.pre_logit.reshape((model.pre_logit.shape[0] * model.pre_logit.shape[1], model.pre_logit.shape[2])).float()), torch.argmax(batch_answers[i].squeeze(1), 1).long() )
#                 loss.retain_grad()
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 scheduler.step()
#                 print(f'The training accuracy is , and loss is {loss.item()} \n ')
#         # list_of_loss.append(loss.item())
#         # ta = get_accuracy(dataset, model, raw_embedding_size, start_token)
#         # va = get_accuracy(va_dataset, model, raw_embedding_size, start_token)
#         print(f'The training accuracy is , and loss is {loss.item()} \n validation accuracy ')
#         torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss.item()}, '/home/vijay/Documents/413/playWriter/checkpoint/things.txt')
#         # list_of_train_accuracy.append(ta)
#         # list_va.append(va)
        
#         del data, batches, batch_encoder, batch_answers
#         gc.collect()
#         torch.cuda.empty_cache()

    

#     return list_of_train_accuracy, list_of_loss, list_va



# def train_differently(epoch: int, dataset: torch.Tensor, va_dataset: torch.Tensor, batch_size: int, model, device, start_token: int, raw_embedding_size: int):
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=0.01)
#     de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), raw_embedding_size).unsqueeze(0) * torch.ones((batch_size, 1, raw_embedding_size)) #Make a tensor of shape (batch_size, 1, 10086) start token for the decoder
#     de_in = de_in.to(device)
#     optimizer.zero_grad()
#     list_of_train_accuracy = []
#     list_of_loss = []
#     list_va = []
#     for i in range(epoch):
#         data = shuffle_dataset(dataset).to(device) #shuffle the data
#         batches = list(split_dataset(data, batch_size)) # returns a list of tensor objects [batch_size, sentence_length, embedding_size]
#         batches.pop() #remove the last uneven one
#         for batch in batches:
#             batch_encoder, batch_answers = split_differnently(batch, 45)
#             for i in range(len(batch_answers)):
#                 results = model.forward(batch_encoder, de_in)[:,-1,:].unsqueeze(1)
#                 de_in = torch.concat((de_in, batch_answers[i]), 1)
#                 loss = loss_func((model.pre_logit[:,-1,:].unsqueeze(1).reshape((model.pre_logit[:,-1,:].unsqueeze(1).shape[0] * model.pre_logit[:,-1,:].unsqueeze(1).shape[1], model.pre_logit[:,-1,:].unsqueeze(1).shape[2])).float()), torch.argmax(batch_answers[i].squeeze(1), 1).long() )
#                 loss.retain_grad()
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 gc.collect()
#                 torch.cuda.empty_cache()
#         list_of_loss.append(loss.item())
#         ta = new_accuracy(dataset, model, raw_embedding_size, start_token)
#         va = new_accuracy(va_dataset, model, raw_embedding_size, start_token)
#         print(f'The training accuracy is {ta}, and loss is {loss.item()} \n validation accuracy {va}')
#         list_of_train_accuracy.append(ta)
#         list_va.append(va)
        
#         del data, batches, batch_encoder, batch_answers
#         gc.collect()
#         torch.cuda.empty_cache()
    
#     return list_of_train_accuracy, list_of_loss, list_va

# class NewModel(nn.Module):

#   def __init__(self, one_hot_embedding_size, device):
#     super(NewModel, self).__init__()
#     self.one_hot_embedding_size = one_hot_embedding_size
#     self.model = torch.nn.Transformer(batch_first=True)
#     self.input_warp = torch.rand((1, one_hot_embedding_size, 512), requires_grad=True, device=device)
#     torch.nn.Parameter(self.input_warp)
#     self.posencode = PositionalEmbeddings()
#     self.device = device
#     self.out_matrix = torch.rand((1, 512, one_hot_embedding_size), requires_grad=True, device=device)
#     torch.nn.Parameter(self.out_matrix)
  
#   def forward(self, encoder_input, decoder_input):
#     encoder_input = self.posencode.add_posencoding(torch.matmul(encoder_input.to(self.device), self.input_warp.to(self.device)), 512, encoder_input.shape[1], encoder_input.shape[2]).to(self.device)
#     decoder_input = self.posencode.add_posencoding(torch.matmul(decoder_input.to(self.device), self.input_warp.to(self.device)), 512, decoder_input.shape[1], decoder_input.shape[2]).to(self.device)
#     model = self.model.to(self.device)

#     results = self.model.forward(encoder_input, decoder_input)

#     self.pre_logit = torch.matmul(results, self.out_matrix)

#     return self.pre_logit




# acc, general_loss, va = train(200, train_dataset[:4, :, :], valid_dataset[:4, :, :], 50, transformer, device, start_token, 10086)