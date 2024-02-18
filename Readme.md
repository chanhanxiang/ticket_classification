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

<h3>Sequence models</h3>

A recurrent neural network (RNN) was trained as the baseline. RNN was selected for this text classification task as it can process sequential data of variable input length by maintaining an internal state memory, 
capturing the context and dependencies between words. Basic RNNs with the below configuration were trained (Table 1). The Adam optimizer with a learning rate of 0.0001 was used as it generally performs well for text classification. 
Categorical cross entropy (CCE) loss function and a final softmax layer with 8 units were applied as this is a multi-class classification task with 8 possible labels.

Table 1: Summary of Sequence Model Configuration & Results

![Screenshot from 2024-02-18 14-49-29](https://github.com/chanhanxiang/tokenise_exercise/assets/107524953/d2e967c8-93e4-4b2b-b61d-1f3db470783e)


In RNN 1, the testing accuracy of the RNN model was 78.43%. This test accuracy will be used as a baseline for subsequent models. 
While the training & validation losses were still downtrending after 7 epochs, the rate of improvement in the validation accuracy is much lower than that of the training accuracy and variance of the model was increasing sizeably especially after the fourth epoch. 
Hence, running more epochs will likely lead to overfitting. For RNN 2, I will attempt to improve the accuracy by using a more complex model, increasing the number of weights in the dense layer from 32 to 128. 
To reduce risk of overfitting, fewer epochs (7 to 6) will be applied. 
After training, RNN 2 performed similarly to RNN 1, with a slightly higher validation accuracy and lower variance between training and validation accuracy, but with a lower testing accuracy of 76.99%.

A possible reason for the mediocre testing accuracy is the vanishing gradient problem in RNNs. As some tickets have large lengths of text, the gradient signal from earlier time steps may be lost or ‘forgotten’. 
To address this, the Long Short-Term Memory (LSTM) model will be applied. LSTMs are a variation of and improve on RNNs by keeping a separate long-term memory cell (cell-state) to preserve memory from earlier time steps.
LSTM models with the above configuration were trained (Table 1). LSTM 1 performed better than RNN, with higher test accuracy of 79.6% and significantly lower training time. 
This can be attributed to LSTMs better being able to learn long-term dependencies as alluded to in the previous paragraph. However, LSTM 1 showed sizable variance and slight overfitting. 
To address this, dropout layers were applied in LSTM 2, which resulted in similar test accuracy but a lower variance, thus reducing overfitting.
