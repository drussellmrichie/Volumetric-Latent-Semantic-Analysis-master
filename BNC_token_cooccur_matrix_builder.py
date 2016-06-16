"""
README
This file reads a corpus (here, BNC), opens a +/- n (here, 15) window whenever
it comes across a vocab item, and then adds the context vectors for all the words
in that vocab items context, yielding a representation of the meaning of that 
word-token.

CRUCIAL -- for a number n of SVD'd dimensions, at least n + 1 tokens must 
extracted for convex hulls method of building volume.

ALSO CRITICAL -- NEED TO REMOVE ZEROS FROM MATRIX AT THE END.

This script is unoptimized. May want to consider using Cython or something to 
make it faster, etc?

This script can also either include or exclude stopwords, depending on which
portion of code is commented out.

Script can also do either first n_types, or do "logarithmic" sampling
"""

from __future__ import division
import numpy
import os, sys, subprocess
import nltk.corpus.reader.bnc
import cPickle as pickle
from nltk.corpus import stopwords

RCMD = r'/usr/bin/Rscript'
os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC')

bnc = nltk.corpus.reader.bnc.BNCCorpusReader(
        r'/Users/russellrichie/introcompling/FinalProject/BNC/2554/download/Texts',
        r'.*/.*/.*.xml')

RCMD = r'/usr/bin/Rscript'
ROWS = 20000
COLS = 1000
stopset = set(stopwords.words('english'))

# Load the dictionaries with the row and column names

"""
And here is where we load the LDT words...then need to reformat this in the same way as the old r2i
"""
r2i = pickle.load( open( "ldtFrequentWords.p", "rb" ) )
c2i = pickle.load( open( "c2iWithoutStopsLemmatized.p", "rb" ) )
#r2itest = pickle.load( open( "r2iWithoutStopsNotLemmatized.p", "rb" ) )
#c2i = pickle.load( open( "c2iWithoutStopsNotLemmatized.p", "rb" ) )
#r2i = pickle.load( open( "r2iWithStopsNotLemmatized.p", "rb" ) )
#c2i = pickle.load( open( "c2iWithStopsNotLemmatized.p", "rb" ) )
#r2i = pickle.load( open( "r2iWithStops.p", "rb" ) )
#c2i = pickle.load( open( "c2iWithStops.p", "rb" ) )
#r2i = pickle.load( open( "r2i.p", "rb" ) )
#c2i = pickle.load( open( "c2i.p", "rb" ) )
ROWS = len(r2i)
COLS = len(c2i)

# set a few parameters...number of svd dimensions, number of tokens per type,
# number of types

log_samp = False # if True, logarithmic sampling, if False, first n_types are used
specificWordSet = True # set this to true if using a predetermined word set, like the words for which there exist LDT or naming RT data

if log_samp == True: # THERE MIGHT BE SOMETHING ABOUT THIS VERSION OF THE PROGRAM THAT DOESN'T LIKE LOG SAMPLING???
    type_inds = numpy.logspace(0,4,1000)
    n_types = len(type_inds)
elif specificWordSet:
    n_types = len(r2i)
else:
    n_types = 1000
n_dim = 100 #should amp this up to 100, and then SVD the resulting matrices down for convex hull extraction
token_length = 500

# read the svd'd word vectors in...
print ("Reading in the SVD file")
svd_wordvectors = numpy.genfromtxt('wordvectors_without_stops_lemmatized.csv', delimiter=',')[:,1:n_dim] #DISCARD FIRST SVD'D COLUMN
#svd_wordvectors = numpy.genfromtxt('wordvectors_without_stops_not_lemmatized.csv', delimiter=',')[:,1:n_dim] #DISCARD FIRST SVD'D COLUMN
#svd_wordvectors = numpy.genfromtxt('wordvectors_with_stops_not_lemmatized.csv', delimiter=',')[:,:n_dim] #KEEP FIRST SVD'D COLUMN
#svd_wordvectors = numpy.genfromtxt('wordvectors_with_stops_not_lemmatized.csv', delimiter=',')[:,1:n_dim] #DISCARD FIRST SVD'D COLUMN
#svd_wordvectors = numpy.genfromtxt('wordvectors_with_stops.csv', delimiter=',')[:,1:n_dim] #DISCARD FIRST SVD'D COLUMN
#svd_wordvectors = numpy.genfromtxt('wordvectors_without_stops.csv', delimiter=',')[:,1:ndim] #DISCARD FIRST SVD'D COLUMN

if log_samp == True:
    vocab = []
    for ind in type_inds:
        vocab.append(r2i.keys()[ind]) # this also ain't right...need to sort    
else:
    vocab = sorted( r2i.keys(),
                    key=lambda x:r2i[x],
                    reverse=False)[:n_types] #this ain't right...need to sort
vocab_set = set(vocab)

"""
Start token matrix, which is a 3d matrix, type x token x SVD-dimension
(e.g., 'eat' x 4th occurrence of 'eat' x some SVD-dimension)
"""

#token_cooc_matrix = numpy.zeros(shape=(n_types, token_length, n_dim)) #3rd dim should be n_dim because we are KEEPING first SVD'd dimension (because it seems to matter...)
token_cooc_matrix = numpy.zeros(shape=(n_types, token_length, n_dim-1)) #3rd dim should be n_dim-1 because we are leaving out first SVD'd dimension since it is noise
token_count = [0] * n_types

"""
Fill in type x token x SVD-dimension matrix, file by file...this takes ~30 min
Figure out how to make single vector for sum of context vectors

Remember to keep track of token count, based on vocind
"""
sys.stderr.write('Doing co-occurrence counts...\n')
for fnum,fid in enumerate(bnc.fileids()):
    if fnum % 50 == 0:
        sys.stderr.write('Document {} out of {}...\n'.format(
                fnum,len(bnc.fileids())))
    words = [word for word in (w.lower() for w in bnc.words(fid,stem=False)) # BE CAREFUL WHETHER STEM IS FALSE OR TRUE
             if word not in stopset #uncomment this if want to exclude stop words
             if word.isalpha()]
    for index, word in enumerate(words): # why don't try making i the right-hand edge, rather than the middle?
        #if (word in vocab_set and token_count[r2i[word]] < token_length):
        if (word in vocab_set and token_count[vocab.index(word)] < token_length): # need this for LDT word list
            #vocind = r2i[word]
            vocind = vocab.index(word) # need this for LDT word list
            tokind = token_count[vocind] # for some reason this was embedded in the loop below...
            j = max(0,index-15)
            k = min(len(words),index+16)  # indexing is allowed to run off the right
            for x in range(j,k):
                if words[x] in c2i:
                    #if not x == index: # okay to keep this when just measuring volume, but when want to localize words, should remove this.
                    contextvec = svd_wordvectors[c2i[words[x]]]
                    token_cooc_matrix[vocind, tokind, : ] += contextvec # add the vector of the context word to this token's vector ... 
            token_count[vocind] += 1

sys.stderr.write('Done.\n')

print 'Saving token cooccurrence matrix to file'
numpy.save('token_cooc_matrix_without_stops_lemmatized_first_svd_dim_discarded.npy',token_cooc_matrix)
#numpy.save('token_cooc_matrix_without_stops_not_lemmatized_first_svd_dim_discarded.npy',token_cooc_matrix)
#numpy.save('token_cooc_matrix_with_stops_not_lemmatized_first_svd_dim_kept.npy',token_cooc_matrix)
#numpy.save('token_cooc_matrix_with_stops_not_lemmatized.npy',token_cooc_matrix)
#numpy.save('token_cooc_matrix_with_stops.npy',token_cooc_matrix)
#numpy.save('token_cooc_matrix_without_stops.npy',token_cooc_matrix)

'''
Usually, will run what's above, and then run what's below. If running all at once,
don't need to load token_cooc_matrix
'''

#token_cooc_matrix = numpy.load('token_cooc_matrix.npy')[:,:n_types,:] # n_types and other params might vary and need to be fixed here
#token_cooc_matrix = numpy.load('token_cooc_matrix_with_stops_not_lemmatized.npy')[:n_types,:token_length,:n_dim-1]
#token_cooc_matrix = numpy.load('token_cooc_matrix_with_stops_not_lemmatized_first_svd_dim_kept.npy')[:n_types,:token_length,:n_dim]
#token_cooc_matrix = numpy.load('token_cooc_matrix_with_stops_not_lemmatized_first_svd_dim_kept.npy')[:n_types,:token_length,:n_dim]
#token_cooc_matrix = numpy.load('token_cooc_matrix_without_stops_not_lemmatized_first_svd_dim_discarded.npy')
token_cooc_matrix = numpy.load('token_cooc_matrix_without_stops_lemmatized_first_svd_dim_discarded.npy')

print 'Now reshaping and SVD\'ing the entire token cooccurrence matrix. This takes 20-30 minutes'
reshaped_matrix = token_cooc_matrix.reshape(n_types*token_length,n_dim-1) #use this if 1st dim was discarded
#reshaped_matrix = token_cooc_matrix.reshape(n_types*token_length,n_dim) #use this one if 1st dim was kept...could change this to refer directly to dims of token_cooc_matrix

#numpy.savetxt('reshaped_token_cooc_matrix_without_stops.txt',reshaped_matrix)
#numpy.savetxt('reshaped_token_cooc_matrix_with_stops_not_lemmatized.txt',reshaped_matrix)
#numpy.savetxt('reshaped_token_cooc_matrix_with_stops_not_lemmatized_first_svd_dim_kept.txt',reshaped_matrix)
#numpy.savetxt('reshaped_token_cooc_matrix_without_stops_not_lemmatized_first_svd_dim_discarded.txt',reshaped_matrix)
numpy.savetxt('reshaped_token_cooc_matrix_without_stops_lemmatized_first_svd_dim_discarded.txt',reshaped_matrix)


sys.stderr.write('Calling R...\n')
subprocess.call([RCMD, 'bnc_token_svd.R'])

sys.stderr.write('Loading SVD\'d token cooccurrence matrix and reshaping it into final token cooc matrix...\n')
#svd_token_cooc_matrix = numpy.genfromtxt('svd_reshaped_token_cooccur_matrix.csv', delimiter=',')
#svd_token_cooc_matrix = numpy.genfromtxt('svd_reshaped_token_cooccur_matrix_without_stops.csv', delimiter=',')
#svd_token_cooc_matrix = numpy.genfromtxt('svd_reshaped_token_cooccur_matrix_with_stops_not_lemmatized.csv', delimiter=',')
#svd_token_cooc_matrix = numpy.genfromtxt('svd_reshaped_token_cooccur_matrix_with_stops_not_lemmatized_first_svd_dim_kept.csv', delimiter=',')
#svd_token_cooc_matrix = numpy.genfromtxt('svd_reshaped_token_cooccur_matrix_without_stops_not_lemmatized_first_svd_dim_discarded.csv', delimiter=',')
svd_token_cooc_matrix = numpy.genfromtxt('svd_reshaped_token_cooccur_matrix_without_stops_lemmatized_first_svd_dim_discarded.csv', delimiter=',')

final_token_cooc_matrix = svd_token_cooc_matrix.reshape(n_types, token_length, 8) # or however many dims...this also just might not work...

print 'Now removing 0\'s and saving token coocccurence matrices. This takes a minute.'
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files/')
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops/')
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_without_stops_with_one_svd/')
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops_and_log_sampling/')
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops_with_one_svd_not_lemmatized/')
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_with_stops_with_one_svd_not_lemmatized_first_svd_dim_kept/')
#os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_without_stops_with_one_svd_not_lemmatized_first_svd_dim_discarded/')
os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC/token_by_context_files_without_stops_with_one_svd_lemmatized_first_svd_dim_discarded/')

for x in range(0,n_types):
    sys.stderr.write('Writing file {} out of {}...\n'.format(
                x,n_types))
    currmat = final_token_cooc_matrix[x,:,:]
    nz = (currmat == 0).sum(1)
    cleaned_matrix = currmat[nz == 0, :] #"boolean indexing"...only takes those rows of currmat where corresponding row in nz == 0, i.e. have no zero's
    #filename = 'type_#_' + str(x) + '_token_cooc_matrix.txt'
    filename = 'type_#_' + "%04d" % x + '_token_cooc_matrix.txt'
    numpy.savetxt(filename,cleaned_matrix) # tokens should be rows, and (svd'd) contexts are columns
    #numpy.savetxt(filename,token_cooc_matrix[x,:,:]) # are tokens rows, and contexts columns?

os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC')

#make frequency dictionary, then extract vocab and context from that...would prefer to reload frequencydict csv, but that's giving me problems right now    
#DictOfFreqDists = pickle.load( open( "DictOfFreqDists.p", "rb" ) )
#
#docfreqs = ConditionalFreqDist()
#sys.stderr.write('Making the ConditionalFreqDist\n')
#for fnum, fid in enumerate(DictOfFreqDists.keys()):
#    if fnum % 100 == 0:
#        sys.stderr.write('Document {} out of {}...\n'.format(
#                fnum,len(bnc.fileids())))
#    for word in DictOfFreqDists[fid].keys():
#        docfreqs[word][fid] = DictOfFreqDists[fid][word] #mem error if do docfreqs[word][fid], i guess because that takes more mem than other way around
#        
#wordlist = [w for w in sorted( docfreqs.keys(),
#                               key=lambda x:docfreqs[x].N(),
#                               reverse=True)
#            if w not in stopset
#            if docfreqs[w].N() > 2]