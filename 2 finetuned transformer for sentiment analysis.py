import pandas as pd
df = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
# print(df.shape)

x = list(df['message'])
# print(x)
y = list(df['label'])
# print(y)

y = list(pd.get_dummies(y,drop_first=True)['spam'])
# print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)
# print(X_train)


from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


train_encoding = tokenizer(X_train,truncation=True,padding=True)
test_encoding = tokenizer(X_test,truncation=True,padding=True)
# print(train_encoding)


import tensorflow as tf
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encoding),y_train))
# print(train_dataset)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encoding),y_test))
# print(test_dataset)


from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
training_args = TFTrainingArguments(
    output_dir=r'/home/paras/PycharmProjects/pythonprojects/A.I/DL/Transformers/results',
    num_train_epochs = 2,
    per_device_train_batch_size= 4,
    per_device_eval_batch_size=8,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir=r'/home/paras/PycharmProjects/pythonprojects/A.I/DL/Transformers/log_dir',
    logging_steps=10
)


with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
trainer = TFTrainer(
    model = model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
print(trainer.evaluate(test_dataset))
print(trainer.predict(test_dataset)[1])
trainer.save_model('senti_model')