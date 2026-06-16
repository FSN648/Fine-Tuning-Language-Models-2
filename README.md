# In-Context Learning and Parameter-Efficient Fine-Tuning of LLaMA 3 8B for Natural Language Inference

This project evaluates LLaMA 3 8B on a Natural Language Inference (NLI) classification task using two complementary strategies: prompting-based in-context learning (no parameter updates) and QLoRA-based fine-tuning. Results are compared against an encoder-based baseline (RoBERTa-large) on the same task and dataset.

## Part 1: In-Context Learning (ICL)

### Zero-Shot Prompting
A zero-shot prompt is constructed and evaluated on the MultiNLI dataset without providing any labeled examples. The prompt text, decoding hyperparameters (e.g., temperature), and the reasoning behind these choices are documented and the resulting performance is reported.

### One-Shot Prompting
To assess the effect of in-context demonstrations, the zero-shot prompt is extended with a single labeled example drawn from the training set. The strategy used for selecting this example, and the rationale for the one-shot approach, are described alongside the resulting performance.

## Part 2: Fine-Tuning with QLoRA

### Generative Fine-Tuning
LLaMA 3 8B is fine-tuned on the NLI task using QLoRA, with the model trained to generate the predicted class label as text output. After training, the LoRA adapter weights are merged with the base model weights, and the merged model is evaluated on the test set. Training hyperparameters and methodology are documented in detail.

### Classification Head Fine-Tuning
As a second QLoRA-based approach, a linear classification layer is added on top of the base model (using `LlamaForSequenceClassification`) and fine-tuned via QLoRA. The placement of the linear layer within the model architecture, along with the chosen hyperparameters and the rationale behind this design, are described.

## Results

Final results across all approaches (zero-shot, one-shot, QLoRA generative fine-tuning, QLoRA classification head fine-tuning) are compared in a results table alongside the encoder baseline (RoBERTa-large, evaluated separately), reporting accuracy, training time, and number of trainable parameters for each method.
