# 2nd Place - Summary of Approach

**Author:** Devon Stanfield
**Rank:** 2
**Votes:** 125

---

I joined the competition late, played around with some ideas, couldn't get a good model going, but I learned a lot though. As it neared the end of the competition, I figured I'd submit a simple fine-tuned model for the sake of having a submission.

I looked up cassava pre-trained models and found [https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2](url).
To setup the model on the TPU, I referred to tensorflow's tpu setup guides. 

Training: [https://www.kaggle.com/devonstanfield/cassava-train](url)
My image preprocessing was minor, just a quick resize and rescale. Most of the functions in the data pipeline were to handle TFRecords. I also extended the training labels from 5 to 6 as to match the cassava model's output which included a "background" label.
During training, I used "EarlyStopping" and "ReduceLROnPlateau". 
The callbacks would cutoff training around ~30 epochs.
I saved the model weights: [https://www.kaggle.com/devonstanfield/cassava-model](url) to a dataset, since we weren't allowed to perform inference on the TPU. I also saved the cached version of the tfhub layer to a dataset: [https://www.kaggle.com/devonstanfield/cassava-layer](url) since we weren't allowed the internet during inference.

Inference: [https://www.kaggle.com/devonstanfield/cassava-infer](url)
Before inference, I loaded the previously cached version of the model and the saved weights. The preprocessing pipeline was the same as training, mostly dealing with TFRecords. After passing the data through the model, I saved the results.

Overall, I am surprised I placed 2nd. My approach was pretty much by-the-book basic. This was demonstrated by my model placing 660th in the public leaderboard with a score of 0.9025, landing right in the middle of a bunch of other people. I assumed they did the same thing I did. Go figure :)
