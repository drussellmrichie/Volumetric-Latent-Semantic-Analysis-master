"""
README
This file builds a vocab x context words cooccurence matrix, for the ultimate 
purpose of doing some LSA analysis on a corpus.

The file first extracts the 20k (vocab) and 1k (context) most frequent nonstop words.
Then it reads each document in a corpus (here, the BNC corpus), and extracts
its words, leaving out stopwords and punctuation. It then goes through each word,
and, if the word is in the vocab list, it opens a +/- 15 word window around
the word. It then goes through each of those window-words, and, if the window-word
is in the context list, it increments the corresponding entry in the cooccurence
matrix. It then square roots all entries, length-normalizes vectors, and writes 
the matrix to a file (which is then SVD'd in R).

This script is unoptimized. May want to consider using Cython or something to 
make it faster, etc.

This script can also either include or exclude stopwords, depending on which
portion of code is commented out.

AS OF SEPT 7, 2014 -- we are redirecting attention from Manin/Zipf to word 
recognition, priming, and other lexical phenomena. Accordingly, we no longer
want to stop words. Also, I'm lemmatizing everything.
"""

from __future__ import division
import numpy as np
import os, sys, subprocess
from math import log
from nltk.corpus import stopwords
from nltk import ConditionalFreqDist, FreqDist
import nltk.corpus.reader.bnc
import cPickle as pickle

bnc = nltk.corpus.reader.bnc.BNCCorpusReader(
        r'/Users/russellrichie/introcompling/FinalProject/BNC/2554/download/Texts',
        r'.*/.*/.*.xml')

os.chdir('/Users/russellrichie/introcompling/FinalProject/BNC')

RCMD = r'/usr/bin/Rscript'
ROWS = 20000
COLS = 1000
stopset = set(stopwords.words('english'))

sys.stderr.write('Doing frequency counts...\n')
DictOfFreqDists = {}
for fnum,fid in enumerate(bnc.fileids()): # K/K2/K21.xml previously had an unmatched tag problem, but I think I fixed it.
    if fnum % 10 == 0:
        sys.stderr.write('Document {} out of {}...\n'.format(
                fnum,len(bnc.fileids())))
    docfreq = FreqDist(
    word
    for word in ( w.lower() for w in bnc.words(fid,stem=True) ) # stem=True makes it treat all words of same stem as same (e.g., cats and cat)
    if word.isalpha() 
    if word not in stopset) # (un)comment to include/leave out stops
    DictOfFreqDists[fid] = docfreq
    ## when run this loop in the future, try the below so can do counts and make CFD in one go THIS DOESN'T ACTUALLY WORK -- NEED TO START DOCFREQS OUTSIDE LOOP
    #docfreqs = ConditionalFreqDist()
    #for word in (w.lower() for w in bnc.words(fid) if w.isalpha() ):
    #    docfreqs[fid].inc(word)
    #
sys.stderr.write('Done.\n')

#pickle this since counts take forever...saving takes a minute or so
print "Pickling DictOfFreqDists"
pickle.dump( DictOfFreqDists, open( "DictOfFreqDistsWithoutStopsLemmatized.p", "wb" ) )
#pickle.dump( DictOfFreqDists, open( "DictOfFreqDistsWithoutStopsNotLemmatized.p", "wb" ) )
#pickle.dump( DictOfFreqDists, open( "DictOfFreqDistsWithStopsNotLemmatized.p", "wb" ) )
#pickle.dump( DictOfFreqDists, open( "DictOfFreqDistsWithStops.p", "wb" ) )
#pickle.dump( DictOfFreqDists, open( "DictOfFreqDistsWithoutStops.p", "wb" ) )
"""
Make conditional freq dist from dictionary of freq dists -- hopefully, in the 
future this won't be done, and instead, the CFD will be at the same time as the 
counts. Although apparently pickle and cPickle don't like to dump CFD's, 
suggesting I'll have to pickle dict of freq dists, then convert to CFD?

Also, you can't run the conversion process below if a lot of other stuff has been
saved to memory -- the docfreqs CFD needs a boatload of free memory (at least 
when the words are the conditions, as in docfreqs[word][fid]).
"""
#print "Loading DictOfFreqDists"
#DictOfFreqDists = pickle.load( open( "DictOfFreqDistsWithoutStops.p", "rb" ) )
#DictOfFreqDists = pickle.load( open( "DictOfFreqDistsWithStops.p", "rb" ) ) # loading dict takes a while
#DictOfFreqDists = pickle.load( open( "DictOfFreqDistsWithStopsNotLemmatized.p", "rb" ) )
#DictOfFreqDists = pickle.load( open( "DictOfFreqDistsWithoutStopsNotLemmatized.p", "rb" ) )
#DictOfFreqDists = pickle.load( open( "DictOfFreqDistsWithoutStopsLemmatized.p", "rb" ) )

docfreqs = ConditionalFreqDist()
sys.stderr.write('Making the ConditionalFreqDist\n')
for fnum, fid in enumerate(DictOfFreqDists.keys()):
    if fnum % 100 == 0:
        sys.stderr.write('Document {} out of {}...\n'.format(
                fnum,len(bnc.fileids())))
    for word in DictOfFreqDists[fid].keys():
        docfreqs[word][fid] = DictOfFreqDists[fid][word] #mem error if do docfreqs[word][fid], i guess because that takes more mem than other way around

"""
The below code is used to get the English lexicon project words which occur 
> 500 times in BNC, and put them into a frequency dictionary.
"""

ldtWords = pickle.load( open( "LexicalDecisionDataWords.p", "rb" ) )
ldtFrequentWords = dict()
for word in ldtWords:
    if docfreqs[word].N() > 500:
        ldtFrequentWords[word] = docfreqs[word].N()
pickle.dump(ldtFrequentWords, open('ldtFrequentWords.p','w'))

#pickle.dump( docfreqs, open( 'docfreqs.p','w' ) ) #apparently this doesn't work because docfreqs is honkin' big

def idf(w):
    return (log(len(bnc.fileids())+1) - log(docfreqs[w].B())) # docfreqs[w].B() is how many docs word occurs in
def tf_idf(w):
    return docfreqs[w].N() * idf(w) #docfreqs[w].N() is how often word occurs throughout entire BNC

wordlist = [w for w in sorted( docfreqs.keys(),
                               key=lambda x:docfreqs[x].N(),
                               reverse=True)
            if w not in stopset                            # comment this out if want to include stops
            if docfreqs[w].N() > 2]
r2i = dict( (w,i) for (i,w) in enumerate(wordlist[:ROWS]))
c2i = dict( (w,i) for (i,w) in enumerate(wordlist[50:COLS+50])) # leave out the 50 most frequent words from the context columns
#pickle.dump( r2i, open( 'r2iWithoutStops.p','w' ) )
#pickle.dump( c2i, open( 'c2iWithoutstops.p','w' ) )
#pickle.dump( r2i, open( 'r2iWithStops.p','w' ) )
#pickle.dump( c2i, open( 'c2iWithStops.p','w' ) )
#pickle.dump( r2i, open( 'r2iWithStopsNotLemmatized.p','w' ) )
#pickle.dump( c2i, open( 'c2iWithStopsNotLemmatized.p','w' ) )
#pickle.dump( r2i, open( 'r2iWithoutStopsNotLemmatized.p','w' ) )
#pickle.dump( c2i, open( 'c2iWithoutStopsNotLemmatized.p','w' ) )
pickle.dump( r2i, open( 'r2iWithoutStopsLemmatized.p','w' ) )
pickle.dump( c2i, open( 'c2iWithoutStopsLemmatized.p','w' ) )

# Adjust the dimensions in case there weren't enough items
ROWS = len(r2i)
COLS = len(c2i)
cooc_matrix = np.zeros(shape=(ROWS,COLS))

sys.stderr.write('Doing co-occurrence counts...\n')
for fnum,fid in enumerate(bnc.fileids()):
    if fnum % 50 == 0:
        sys.stderr.write('Document {} out of {}...\n'.format(
                fnum,len(bnc.fileids())))
    words = [word for word in (w.lower() for w in bnc.words(fid))
             if word not in stopset #comment this out if want to include stops
             if word.isalpha()]
    for i in range(len(words)):
        if words[i] in r2i:
            j = max(0,i-15)
            k = min(len(words),i+16)
            for x in range(j,k):
                if not x == i:
                    if words[x] in c2i:
                        cooc_matrix[r2i[words[i]],c2i[words[x]]] += 1
sys.stderr.write('Done.\n')

sys.stderr.write('Applying idf-weighting...')
for (c,i) in c2i.items():
    cooc_matrix[:,i] *= idf(c)
sys.stderr.write('Done.\n')

cooc_matrix = np.sqrt(cooc_matrix)

for i in range(len(cooc_matrix)):
    length = np.sqrt(np.dot(cooc_matrix[i],cooc_matrix[i]))
    if length:
        cooc_matrix[i] = cooc_matrix[i]/length

# save cooccurence matrix to file
sys.stderr.write('Saving data to disk...\n')
np.savetxt('cooc_matrix_without_stops_lemmatized.txt',cooc_matrix)
#np.savetxt('cooc_matrix_without_stops_not_lemmatized.txt',cooc_matrix)
#np.savetxt('cooc_matrix_with_stops_not_lemmatized.txt',cooc_matrix)
#np.savetxt('cooc_matrix_with_stops.txt',cooc_matrix)
#np.savetxt('cooc_matrix_without_stops.txt',cooc_matrix)

#apparently it doesn't want to save the rowlabels.txt because a weird char is in there
#np.savetxt('rowlabels.txt',
#           sorted(r2i.keys(), key=lambda x:r2i[x]),
#           fmt="%s")
#np.savetxt('rowlabelsWithStops.txt',
#           sorted(r2i.keys(), key=lambda x:r2i[x]),
#           fmt="%s")

# The following is just for curiosity; the file can be inspected but
# is not used later - SK
#outfile = open('wordinfo.txt', 'w')
#for word in sorted(r2i.keys(), key=lambda x:r2i[x]):
#    outfile.write("{:>20} {:>6} {:>5.2f} {:>5.2f}\n".format(
#            word, docfreqs[word].N(), tf_idf(word), idf(word)))
#outfile.close()
#sys.stderr.write('Done.\n')

# Then do the SVD in R!
sys.stderr.write('Calling R...\n')
subprocess.call([RCMD, 'bnc_svd.R'])
sys.stderr.write('Done.\n')