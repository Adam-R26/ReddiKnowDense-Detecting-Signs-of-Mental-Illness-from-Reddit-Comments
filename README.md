# ReddiKnowDense: Detecting Signs of Mental Illness from Reddit Comments
### Summary
- This project develops an <b>end-to-end machine learning pipeline</b> that leverages the advanced capabilities of RoBERTa, a large language model, for text classification.
- The model is <b>fine-tuned using two distinct classification heads</b>: a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), labeled as RoBERTa-MLP and RoBERTa-CNN, respectively.
- This approach allows for a <b>comparative analysis of these methods with GPT 3.5's in-context learning capabilities</b>, focusing on <b>identifying potential mental illness indicators in Reddit comments</b>, a novel real world problem.
- Along the way, the <b>power behind the transformer architecture</b> will be discussed in a way to make it understandable to the masses. My goal here is to <b>help people struggling to get to grips with modern NLP techinques, gain a more intuitive understanding of why these methods are so effective</b> rather than just using them as a so called black box.
- However, we will start with the approach to give you and idea the project workflow before delving into this explanation section.

NB: This project also explores the <b>advanced machine learning concept of self-training, where a model's own predictions are recycled as training data to enhance performance</b>. This method is particularly effective with small datasets, where the classification decision boundary is ambiguous. By continuously retraining with its predictions, the model fine-tunes this boundary, improving its accuracy. I will omit this part of the project from this repo to avoid it becoming too convoluted but may make another one in future detailing this method and its effectivness in this domain.

NB: You will notice <b>I have left the API keys I used to connect to the Reddit API within the scripts</b>. Of course these keys have been deleted now and so will not work if you try to use them, but serve as an example of the keys you will need when registering for API usage. Note that in a commercical setting these keys would need to be stored as environment variables to mitigate security risks. This project was purely academic for me, hence I just used them within the script.

### Approach
#### Data Acquisition and Pre-Processing
- 

