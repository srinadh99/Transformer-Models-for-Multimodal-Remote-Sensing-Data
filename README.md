# Transformers-Models-for-Multimodal-Remote-Sensing-Data

For any queries, please contact at srinadhml99@gmail.com

# Details
In this work, I studied the performance of a new Multimodal Fusion Transformer (MFT) for Remote Sensing (RS) Image Classification proposed recently. The link for the original MFT paper is here https://arxiv.org/pdf/2203.16952.pdf. 

Transformer-based models are widely used in several image-processing applications due to their promising performance over Convolutional Neural Networks (CNNs). However, there are certain challenges while adapting transformer models for the Hyperspectral Image (HSI) classification. In RS image classification, using images from multiple sources and exploiting complementary information is getting more attention these days with the increased availability of multi-sensor data. But considering the patch-level processing in the transformers and the number of spectral bands in the HSI data or the fused data, the number of model parameters is becoming a major issue. In MFT work, a new transformer-based architecture with novel attention and tokenization mechanisms is proposed to obtain complementary information from other modalities through external CLS token. They showed CLS token extracted from other multimodal data is generalizing well compared to the random CLS token used in general transformer models.

In this work, I specifically studied the performance of the MFT model with vaying the CLS token. Specifically, I used the CLS token to be 'random', processed through 'channel' and 'pixel' tokenization methods. I considered the 'Trneto' dataset alone to validate the performance of the MFT model.
The performance is interms on training loss and confusion matrix. I have updated the codes also so that we can just call the 'token_type' as either 'channel' or 'pixel' instaed of writing two separate codes like in the original work.

# Our Next Contributions
We will update soon .....

# Acknowledgment
We considered part of our codes from the following source.
1. https://github.com/AnkurDeria/MFT
