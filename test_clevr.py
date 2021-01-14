from data import AoCelvrDataSet

if __name__ == '__main__':

    data = AoCelvrDataSet(True)
    train_data = data.get_data(False)
    print(len(train_data))
    #print(train_data)
