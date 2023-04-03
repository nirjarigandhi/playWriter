
def createDataset(filenames, dataset_size=50):
    
        # Create a dataset from a file
    
        # Open the file
        main_dataset = []
        curr_dataset = []
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
                    else:
                        curr_dataset.append(word)
                # Create a dataset from the list
                # Add the dataset to the list
            f.close()
        main_dataset.append(curr_dataset)
        return main_dataset

dataset1 = createDataset(['play1.txt', 'play2.txt'])
train_len = int(len(dataset1) * 0.8)
train_dataset = dataset1[:train_len]
test_len = int(len(dataset1) * 0.1)
test_dataset = dataset1[train_len:test_len]
valid_dataset = dataset1[test_len+train_len:]