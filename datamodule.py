
from datasets import load_dataset, concatenate_datasets

class DataModule:

  def __init__(self, tokenizer, args, idx= 0):
    self.args= args
    self.tokenizer= tokenizer
    
    print(self.args.train_data_path)
    self.dataset= load_dataset('csv', data_files= {'train': self.args.train_data_path, 'dev': self.args.dev_data_path})
    self.train, self.dev= self.dataset['train'], self.dataset['dev']

    # for except case 
    self.train= self.train.filter(lambda example: example[self.args.label_columns] != -1)
    self.dev= self.dev.filter(lambda example: example[self.args.label_columns] != -1)

    if self.args.label_columns == 'label_sentiment_1':
      self.pos_train= self.train.filter(lambda example: example[self.args.label_columns] == 0).shard(num_shards= 4, index= 0)
      self.other_train= self.train.filter(lambda example: example[self.args.label_columns] != 0)

      print(self.pos_train)
      print(self.other_train)
      self.train= concatenate_datasets([self.pos_train, self.other_train])
      self.train= self.train.shuffle(seed= self.args.seed)

    self.preprocess_train= self._preprocess(self.train)
    self.preprocess_dev= self._preprocess(self.dev)
  
  def _preprocess(self, dataset):
    remove_columns= dataset.column_names
    print(remove_columns)
    return dataset.map(
        self._tokenize,
        remove_columns= remove_columns,
        keep_in_memory= True,
        desc= 'preprocessing data'
    )
  
  def _tokenize(self, example):
    tokenized_example= self.tokenizer(
      example['sentence'],
      truncation= True,
      padding= 'max_length',
      max_length= self.args.max_seq_length,
      return_token_type_ids= False,
      return_tensors= 'pt'
    )
    
    result= dict()
    for key, value in tokenized_example.items():
      result[key]= value.squeeze()
    result['labels']= example[self.args.label_columns]

    return result
  
