# Quickstart

The Transformer that has led to a renaissance in tools for natural
language processing acheiving state-of-the-art results across almost
all benchmarks.  The mission of `transformers` is to make it easy for
anyone _to access, extend, and apply_ these models for their own tasks.

This guide walks you through two broad categories of tasks.

* *Text Understanding*, e.g. [sentiment classification](), [question answering](), [sentence labeling](), and [information extraction]().
* *Text Generation*, e.g. [conditional generation](), [text summarization](), and [machine translation](). 


The library is written such a user does not need to be aware of the internals of the model. However,
you do need to be aware of three key objects: 

* *Tokenizer*; Maps a sequence of words into a sequence of indices. 
* *Model*; Maps a sequence of indices into a sequence of feature embeddings. 
* *Head*; Maps a sequence of feature embeddings to a prediction.


You can pair a most Model's with a task-specific Head. The Model is
 quite large and _pre-trained_ on a massive volume of text; whereas
 the Head is small and can be _fine-tuned_ on a tiny annotated
 dataset.

For the quickstart, we will focus on using `transformers` with community [shared
models](www.huggingface.co/models). For mode details on training,
see [training]().


<!-- The two most common use cases for the library are tasks involving text understanding and -->
<!-- text generation. -->


<!-- Here are two examples showcasing a few `Bert` and `GPT2` classes and pre-trained models. -->

<!-- See full API reference for examples for each model class. -->

## Text Understanding




<!-- Let's start by preparing a tokenized input (a list of token
embeddings indices to be fed to Bert) from a text string using
`BertTokenizer` -->

This section demonstrates how to use  transformers for text
understanding. We focus on sequence classification, i.e.
predicting a class label based on a sentence. Specifically we try to
predict if a movie review has negative or positive sentiment with a
value between 1 and 5.

We begin by specifying a specific model. We use
a classic model in this family, known as [Bert]().

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Can use any model on huggingface.co. This model has been trained for sentiment.
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Download pre-trained Tokenizer, Model, and Head.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set for evaluation on GPU.
model.eval().cuda()
```

Next, we utilize the tokenizer to convert an input sentence to a format that
the model can use.


```python

# For this model, text format start with CLS and splits sentences with [SEP]
text = "[CLS] Muppets Take Manhattan is a great movie . [SEP]"

# Tokenize input
indexed_tokens = tokenizer.encode(text)

# Convert to PyTorch tensor format. 
tokens_tensor = torch.tensor([indexed_tokens]).cuda()
```

Finally, we pass this tensor to our model and mead to predict its output.

```
# Predict the sentiment (1-5) for the input sentence. 
model_output = model(tokens_tensor)[0]
print(model_output.argmax(-1)+1)
```

That's it. In 10 lines, we have quite powerful model for this task.
Similar methods can be used for a range of other text understanding
tasks. These include: 

* AutoModelForTokenClassification
* AutoModelForSequenceClassification
* AutoModelForQuestionAnswering
* AutoModelForTokenClassification

If you want access to the raw transformer outputs, you can also utilize.

* AutoModel

For more advanced usage skip ahead to

* Tokenization
* Training
* Deployment
* Model Sharing


### Text Generation

This section demonstrates how to use transformers for text
generation. We focus on langauge modeling, i.e.  predicting the next
word in a sequence based on the previous words. Specifically we try to
complete a sentence given only its beginning.  We use
a large model of this class, known as [GPT-2]().


```python
import torch
from transformers import AutoTokenizer, AutoModelForLMHeadModel

# Can use any model on huggingface.co. This model has been trained for language modeling.
model_name = 'gpt2'

# Download pre-trained Tokenizer, Model, and Head.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForLMHeadModel.from_pretrained(model_name)

# Set for evaluation on GPU.
model.eval().cuda()
```

Next, we utilize the tokenizer to convert an input sentence to a format that
the model can use.


```python
# Encode a text inputs
text = "Who was Jim Henson ? Jim Henson was a"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens]).cuda()
```

Finally, we pass this to our model and decoder it to the next word.

```python

# Run the model.
outputs = model(tokens_tensor)[0]

# Convert to a word. 
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
```

If we want to generate a sequence of output words we can continue running
in a loop.

```
generated = tokenizer.encode(text)
context = torch.tensor([generated])
past = None

for i in range(100):
    output, past = model(context, past=past)
    token = torch.argmax(output[..., -1, :])

    generated += [token.tolist()]
    context = token.unsqueeze(0)

sequence = tokenizer.decode(generated)
```

This is the basic form of text generation but similar models and head
can be used for a range of other text understanding tasks. These
include:

* AutoModelForMaskedLM
* AutoModelForLMHeadModel

For more advanced usage skip ahead to

* Tokenization
* Training
* Deployment
* Model Sharing
* Sequential generation


