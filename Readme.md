<h1>Service Ticket Classification</h1>

Sorting IT Support tickets manually so as to route the ticket to the relevant department can be time consuming and it would be ideal if they can be automated.

The objectives are:

1. Identification of Service Issue Type and automated assignment to the right service department to increase SLA for resolution

2. Use the latest development in the NLP and deep learning area to improvise the model with a human in the loop by providing regular feedback. 

The dataset used in this report was obtained from Kaggle (https://www.kaggle.com/datasets/aniketg11/supportticketsclassification). The dataset contains 48,549 rows of data and 9 features. 
Out of which in the title column, 712 does not have any text. Rows with no text in the title are dropped. Dropping these rows would not have a big impact as these were only 1.4% of the total dataset.

<h3>Preprocessing</h3>

Since the dataset does not provide metadata on the categorical numerics in the ticket (columns C to I above), these will not be applied directly to our model. 
The approach shifted to using the transformer BERTopic to cluster each row of the data by its topics. The clustering is based on cosine similarity. 
The dataset was then labeled accordingly with broad-based clusters including: 'Hardware', 'HR Support', 'Access', 'Miscellaneous', 'Storage', 'Purchase', 'Internal Project', 'Administrative rights'. 
This modeled dataset contained 47837 rows and 2 columns (ticket text and category). 

Prior to training, the rows in the dataset were shuffled and the 4 majority classes (Hardware, HR Support, Access, Miscellaneous) were randomly downsampled to 5000 rows each. 
This was done to reduce the imbalanced distribution of categories and reduce overfitting to the majority classes, as well as to reduce training time with a smaller dataset. 
The testing dataset consisted of 1000 randomly selected rows, which will be set aside separate from the training process. 
The remaining data was split for training and validation with a 80:20 ratio. A vectorization layer was first adapted based on the text in the training set, and then applied to all text to normalize, 
split, and map strings to integers. Before training, the vectorised input data were fed to an embedding layer and their corresponding labels were one-hot-encoded.

<h3>RNN and LSTM</h3>

A recurrent neural network (RNN) was trained as the baseline. RNN was selected for this text classification task as it can process sequential data of variable input length by maintaining an internal state memory, 
capturing the context and dependencies between words. Basic RNNs with the below configuration were trained (Table 1). The Adam optimizer with a learning rate of 0.0001 was used as it generally performs well for text classification. 
Categorical cross entropy (CCE) loss function and a final softmax layer with 8 units were applied as this is a multi-class classification task with 8 possible labels.

Table 1: Summary of Sequence Model Configuration & Results

![Screenshot from 2024-02-18 14-49-29](https://github.com/chanhanxiang/ticket_classification/assets/107524953/eb09531d-e1f1-4661-938c-00081ed2d190)


In RNN 1, the testing accuracy of the RNN model was 78.43%. This test accuracy will be used as a baseline for subsequent models. 
While the training & validation losses were still downtrending after 7 epochs, the rate of improvement in the validation accuracy is much lower than that of the training accuracy and variance of the model was increasing sizeably especially after the fourth epoch. 
Hence, running more epochs will likely lead to overfitting. For RNN 2, I will attempt to improve the accuracy by using a more complex model, increasing the number of weights in the dense layer from 32 to 128. 
To reduce risk of overfitting, fewer epochs (7 to 6) will be applied. 
After training, RNN 2 performed similarly to RNN 1, with a slightly higher validation accuracy and lower variance between training and validation accuracy, but with a lower testing accuracy of 76.99%.

![RNN](https://github.com/chanhanxiang/ticket_classification/assets/107524953/f23b68eb-cecf-44f5-83b7-934dfe9697d6)

A possible reason for the mediocre testing accuracy is the vanishing gradient problem in RNNs. As some tickets have large lengths of text, the gradient signal from earlier time steps may be lost or ‘forgotten’. 
To address this, the Long Short-Term Memory (LSTM) model will be applied. LSTMs are a variation of and improve on RNNs by keeping a separate long-term memory cell (cell-state) to preserve memory from earlier time steps.

LSTM models with the above configuration were trained (Table 1). LSTM 1 performed better than RNN, with higher test accuracy of 79.6% and significantly lower training time. 
This can be attributed to LSTMs better being able to learn long-term dependencies as alluded to in the previous paragraph. However, LSTM 1 showed sizable variance and slight overfitting. 
To address this, dropout layers were applied in LSTM 2, which resulted in similar test accuracy but a lower variance, thus reducing overfitting.

![LSTM](https://github.com/chanhanxiang/ticket_classification/assets/107524953/8c977fb9-5238-40e2-989a-78c4b5dbe29f)


To further improve on the model accuracy, a bi-directional LSTM (Bi-LSTM) was applied (Table 2). BiLSTM processes input text in both forward and backward directions simultaneously, allowing it to better capture and retain long-term dependencies than standard LSTMs, which can be particularly helpful for tickets with longer text sequences. When applied to the dataset, Bi-LSTM 1 outperformed LSTM with a testing accuracy of 83.52%. However, there is some degree of variance observed, suggesting slight overfitting. In Bi-LSTM 2, the number of weights in the LSTM layer and Dense layer were increased allow it to learn more complex relationships while number of epochs were reduced from 10 to 6 to reduce risk of overfitting. This improved on the previous model by producing a lower variance, reducing overfitting while increasing in test accuracy to 84.13%.

![BiLSTM](https://github.com/chanhanxiang/ticket_classification/assets/107524953/bbfe4220-38ce-480a-8f07-7755c6b1818c)

Additionally, hybrid models involving bidirectional LSTM or GRU and convolutional layers (CNN) were used. CNNs are good at detecting local patterns in data, such as n-grams or phrases that are important for text classification, while LSTM or GRUs model the context and dependencies between words. By combining them, the model can learn both local and global features of the text, thereby improving the overall performance of the model. Various hybrid combinations were trialed and the architecture with the best results were outlined in Table 1. The Bi-LSTM+CNN model improved on Bi-LSTM 2 with a lower variance while maintaining a similar test accuracy. Whereas the best result was obtained with the Bi-GRU+CNN model with the highest test accuracy of 84.62%, and low train-test variance below 1%, as well as good training speed.

![Classification](https://github.com/chanhanxiang/ticket_classification/assets/107524953/497ef1fc-565b-4e34-9936-5220c38717ef)

<h3>Distilbert</h3>

The tokenizer and model maybe first imported from transformers. To run Distilbert model, DistilbertTokenizer and TFDistilBertModel were selected as the tokenizer and model respectively. For the model, ‘distilbert-base-uncased' was used as the text data did not contain capitalised words. Max length was set at 200 to strike a balance between faster training time and covering optimal length of each string. Batch size is set to 16, a common figure used for text processing. Two input layers were fed into the model: input_ids and attention_mask. Input_id is to give each word an unique identifier, while attention_mask indicates padding status for the tokens - 0 if padded, 1 if otherwise. Token_type_ids was not applied as it is used for padding in the prediction of the next sentence, and out dataset consist of single sentence datatype.

For the optimiser, Adam was used as the sample involves a Multi-Class classification. Sparse Categorical Cross Entropy was applied. Learning rate was set at 5e-05. If faster learning rates (to the power of -04 or -03), there is an observed tendency for the train and validation (labelled as test henceforth) sets diverging. Initially, 5000 samples were extracted out of the entire dataset and run for 5 epochs. Maximum accuracy reaches 0.38 while optimal loss reaches 1.60, but based on the trend line as shown below it can be improved much more:

![Distilbert](https://github.com/chanhanxiang/ticket_classification/assets/107524953/9be9053d-2dcb-4f23-9889-65c7bc8bf6e3)

Subsequently, the entire dataset was applied and split into the train and test sets, 15 epochs was applied. Optimal accuracy (around 0.7) and loss (around 0.8) is obtained at around the 8th epoch. Similar levels of training time and accuracy/loss performance have been observed if AutoTokenizer/TFAutoModel is used (Refer to: Distilbert_full_using_AutoTokenizer.ipynb). If BertTokenizer/TFBertModel is used, training time per epoch is approximately twice the duration as compared to DistilBertTokenizer/TFDistilBertModel, while there is no perceptible decrease in performance, in line with established precedents. 
