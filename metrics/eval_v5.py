""""Evaluates generated sentences.

Please, install NLTK: https://www.nltk.org/install.html
sudo pip install -U nltk


python eval.py <system-dir> <reference-dir>

e.g.
python bin/eval.py system_out_dev Finall4/Sentences/dev/

Author: Bernd Bohnet, bohnetbd@gmail.com, Simon Mille, simon.mille@upf.edu
"""

import codecs
import collections
import io
import os
import sys
import glob
try:
  from nltk.metrics import *
  import nltk.translate.nist_score as ns
  import nltk.translate.bleu_score as bs
except ImportError:
  print('Please install nltk (https://www.nltk.org/)')
  print("For instance: 'sudo pip install -U nltk\'")
  exit()


def read_corpus(filename, ref=False, normalize=True):
  """Reads a corpus

  Args:
    filename: Path and file name for the corpus.

  Returns:
    A list of the sentences.
  """
  data = []
  #print('Received filename: ', filename)
  with open(filename, 'r') as f:
    #for line in codecs.getreader('utf-8')(f, errors='ignore'):  # type: ignore
    for line in codecs.open(f.name, 'r', 'utf-8'):
      line = line.rstrip()
      if line.startswith(u'# text') or line.startswith(u'#text'):
        split = line.split(u'text = ')
        if len(split) > 1:
          text = split[1]
        else:
          #text = '# #'
          text = ''
        if normalize:
          text = text.lower()
        if ref:
          data.append([text.split()])
        else:
          data.append(text.split())
  return data
  print(split)


def main():
  arguments = sys.argv[1:]
  num_args = len(arguments)
  if num_args != 2:
    print('Wrong number few arguments.')
    print(str(sys.argv[0]), 'system-dir', 'reference-dir')
    exit()
  system_path = arguments[0]
  ref_path = arguments[1]
  fo = codecs.open(os.path.join(os.path.abspath(os.path.join(system_path, os.pardir)), 'eval.txt'), 'a', 'utf-8')

  # For all files in system path.
  for filename in os.listdir(system_path):
    print('Filename', str(filename))
    fo.write('Filename: '+filename+'\n')
    system_filename = os.path.join(system_path, filename)
    ref_filename = os.path.join(ref_path, filename)

    # read files
    ref = read_corpus(ref_filename, ref=True)
    hyp = read_corpus(system_filename, ref=False)

    # NIST score
    nist = ns.corpus_nist(ref, hyp, n=4)

    # BLEU score
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(ref, hyp, smoothing_function=chencherry.method2)
    print('BLEU', str(round(bleu, 4)))
    fo.write('BLEU '+str(round(bleu, 4)*100)+'\n')
    print('NIST', str(round(nist, 2)))
    fo.write('NIST '+str(round(nist, 2))+'\n')
    total_str_len = 0.0
    edits, total_word_edits = 0.0, 0.0
    micro_edits, macro_edits = 0.0, 0.0
    cnt = 0.0
    #first_word_edits, total_first = 0.0, 0.0
    word_edits = 0.0

    for r, h in zip(ref, hyp):

      cnt += 1

      # String edit distance.
      s1 = ' '.join(r[0])
      s2 = ' '.join(h)
      total_str_len += max(len(s1), len(s2))
      macro_edits += edit_distance(s2, s1)
      #micro_edits += 1.0-edit_distance(s2, s1)/float(max(len(s1), len(s2)))

      # Word edit distance.
      #total_word_edits += max(len(r[0]), len(h))
      word_edits += edit_distance(r[0], h)

      # First word edit distance.
      #first_word_edits += edit_distance(r[0][0], h[0])
      #total_first += max(len(r[0][0]), len(h[0]))
    print('Macro DIST %.4f' % float(1-macro_edits/total_str_len))
    fo.write('Macro_DIST %.2f' % float((1-macro_edits/total_str_len)*100)+'\n\n')
    #print('Micro DIST %.3f' % float(micro_edits/cnt))
    #print('Word DIST %.3f' % float(1-first_word_edits/total_word_edits))
    #print('First Word DIST %.3f' % float(1-first_word_edits/total_first))
    print ('')
    
  fo.close()

if __name__ == "__main__":
    main()
