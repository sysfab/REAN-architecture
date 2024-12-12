# ***REAN*** architecture

most language models generate text by returning the next token predicted in a given sequence.

the next token returned is usually a 1hot / probability distribution around the desired word.

***REAN*** relies on returning the embeddings vector of the next token, not its idx.

if you want to just run the model and play around, go to the run_model.ipynb file, load the correct weights + word2vec, and run all.

main improvements:
  - lower weight compared to normal method
  - faster (bcs of lower weight)

disadvantages:
  - requires dedicated external word2vec model
  - still really bad performance

feel free to actually add a good readme, this is just a placeholder :)

more info in this vid:
https://www.youtube.com/watch?v=ECx2oLYXRms