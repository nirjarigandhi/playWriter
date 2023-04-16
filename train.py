from Transformer import *
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train(model, train_dataset, valid_dataset, epochs=100, batch_size=50, learning_rate=0.001, device='cpu'):
    # Train the model
    # Create the optimizer
    # Create the loss function
    # Train the model
    # Return the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # Train
        model.train()
        for batch in train_dataset:
            optimizer.zero_grad()
            output = model(batch)

            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
        # Validate
        model.eval()
        with torch.no_grad():
            for batch in valid_dataset:
                output = model(batch)
                loss = loss_fn(output, batch)
                print(loss)
    return model

transformer = Transformer(6, 8, 512, 50, 1000, 6, 8, 8, 512, 50, 50, 10086, None).to(device)
de_in = torch.nn.functional.one_hot(torch.Tensor([start_token]).long(), 10086).unsqueeze(0) * torch.ones((50, 1, 10086)) #Make a tensor of shape (batch_size, 1, 10086)
# loss = loss_func((transformer.pre_logit.reshape((transformer.pre_logit.shape[0] * transformer.pre_logit.shape[1], transformer.pre_logit.shape[2])).float()), torch.argmax(y.squeeze(1), 1).long() )