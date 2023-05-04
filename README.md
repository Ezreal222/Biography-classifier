# Biography-classifier

### Using Naïve Bayes text classification algorithm to categorize short biographies into different classes.
### Reading a corpus of biographies, the classifier learns from the training set using normalization, stop word
removal, word and category counting, and Laplacian-corrected probabilities.
### Evaluated on the test set to output the category predictions, their probabilities, and the overall accuracy of
the model. The program handles varying numbers of categories and different input corpus sizes, making it adaptable to different datasets.
## Example output:
Lady gaga. Prediction: Music. Right.
Government: 0.29 Music: 0.41 Writer: 0.29 Scientist: 0.01 

Aung san suu kyi. Prediction: Government. Right.
Government: 0.88 Music: 0.04 Writer: 0.05 Scientist: 0.04 

John lewis. Prediction: Government. Right.
Government: 1.00 Music: 0.00 Writer: 0.00 Scientist: 0.00 

Naguid mahfouz. Prediction: Writer. Right.
Government: 0.03 Music: 0.01 Writer: 0.66 Scientist: 0.30 

Hilary mantel. Prediction: Writer. Right.
Government: 0.05 Music: 0.01 Writer: 0.88 Scientist: 0.07 

Barbara mcclintock. Prediction: Government. Wrong.
Government: 0.42 Music: 0.02 Writer: 0.42 Scientist: 0.14 

Toni morrison. Prediction: Writer. Right.
Government: 0.00 Music: 0.00 Writer: 1.00 Scientist: 0.00 

Jean perrin. Prediction: Scientist. Right.
Government: 0.12 Music: 0.02 Writer: 0.12 Scientist: 0.74 

Segolene royal. Prediction: Government. Right.
Government: 1.00 Music: 0.00 Writer: 0.00 Scientist: 0.00 

Erik satie. Prediction: Music. Right.
Government: 0.00 Music: 1.00 Writer: 0.00 Scientist: 0.00 

Sin itiro tomonaga. Prediction: Scientist. Right.
Government: 0.00 Music: 0.00 Writer: 0.02 Scientist: 0.98 

Overall accuracy: 10 out of 11 = 0.91.
