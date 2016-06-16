# You can run this by typing the commands at the interactive 
# R prompt, or from the command line by typing the following command
# (assuming that this script is called 'svd.R'):
#
# R CMD BATCH svd.R
#
# If you don't like the output that this write by default to 
# the file 'svd.Rout', give it a bogus output file name (in Linux/Unix):
#
# R CMD BATCH svd.R /dev/null
#
# I assume that the pre-SVD word vectors are written in a 
# space-delimited text file of the form
#
# word num num num ...
# word num num num ...
# ...
#

setwd("/Users/russellrichie/introcompling/FinalProject/BNC")

# Read in the data
#rawvectors <- read.table('cooc_matrix_without_stops.txt')
#rawvectors <- read.table('cooc_matrix_with_stops.txt')
#rawvectors <- read.table('cooc_matrix_with_stops_not_lemmatized.txt')
#rawvectors <- read.table('cooc_matrix_without_stops_not_lemmatized.txt')
rawvectors <- read.table('cooc_matrix_without_stops_lemmatized.txt')

# THIS NEXT SECTION WAS   MADE FOR STEFAN'S CODE. SINCE MY RAWVECTORS DOESN'T HAVE WORD LABELS
# The index '[,-1]' tells R to skip the first column, 
# which holds the words
# (very much unlike Python indexing syntax)
#SVD <- svd(rawvectors[,-1])

# THIS LINE BELOW IS APPROPRIATE FOR MY ORIGINAL CODE, BECAUSE THE FIRST COLUMN OF RAWVECTORS DOES NOT CONTAIN WORD LABELS, BUT ACTUAL COOCCURRENCE COUNTS
SVD <- svd(rawvectors)

# 'SVD' now holds the three matrices:
# SVD$u - left matrix
# SVD$d - singular values (an array)
# SVD$v - right matrix
#
# You can get the original matrix back as follows:
# (SVD$u %*% diag(SVD$d)) %*% t(SVD$v)

# The next step is optional. Standard LSA instructions tell you to do it,
# but there is some debate as to whether it is useful. If you skip it,
# just output SVD$u in the next step.
wordvectors <- SVD$u %*% diag(SVD$d)
# wordvectors <- SVD$u %*% diag(SVD$d)

# Write the result in the same format in which the input was saved.
#write.table(wordvectors, 'wordvectors_without_stops.csv', 
#			 quote=FALSE, sep=",",
#			 row.names=FALSE, col.names=FALSE)
#write.table(wordvectors, 'wordvectors_with_stops.csv', 
#            quote=FALSE, sep=",",
#            row.names=FALSE, col.names=FALSE)
#write.table(wordvectors, 'wordvectors_with_stops_not_lemmatized.csv', 
#            quote=FALSE, sep=",",
#            row.names=FALSE, col.names=FALSE)
#write.table(wordvectors, 'wordvectors_without_stops_not_lemmatized.csv', 
#            quote=FALSE, sep=",",
#            row.names=FALSE, col.names=FALSE)
write.table(wordvectors, 'wordvectors_without_stops_lemmatized.csv', 
            quote=FALSE, sep=",",
            row.names=FALSE, col.names=FALSE)