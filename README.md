rbmdemo
=======

My restricted boltzmann machine demo. See richardweiss.org/blog for more details.

Running
=======
"python RBMWithBiases.py"

It takes as input movies.txt and matrices.txt are each taken from wikipedia articles on matrices and movies. They are chunked and split and then have the RBM fit to the data. The output is the errors of the system, then the values for the hidden nodes for a matrix article and a movie article.

Tests
=====
Sometimes the tests fail. This is a probabalistic issue. If it keeps happening then something is wrong, but it should go away when you run the system again.
